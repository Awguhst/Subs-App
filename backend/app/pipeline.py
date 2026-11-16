"""Upload -> audio -> Scribe -> cues -> Claude -> WebVTT, with caching and de-duplication.

Results are keyed by the SHA-256 of the uploaded bytes. Uploading the same file
again returns the stored result instantly (no provider cost); identical uploads
arriving while the first is still processing share its work; a client that
loses its connection can fetch the finished result later by id.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable

from . import media, subtitles
from .config import Settings
from .costs import EMPTY_COST, CostTracker, Media, billed_audio_seconds
from .elevenlabs import MODEL as STT_MODEL, TranscriptionError
from .breaks import MODEL as BREAKS_MODEL, BreakDetectionError
from .translation import MODEL as TRANSLATION_MODEL, TranslationError
from .uploads import Upload

log = logging.getLogger("subtitles")
CACHE_VERSION = 1
Event = dict[str, Any]


class PipelineError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass
class Job:
    task: asyncio.Task | None = None
    started: float = field(default_factory=time.monotonic)
    last: Event = field(default_factory=lambda: {"stage": "queued"})
    listeners: set[asyncio.Queue] = field(default_factory=set)

    def emit(self, event: Event):
        self.last = event
        for queue in self.listeners:
            queue.put_nowait(event)

    def listen(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(self.last)
        if self.task is not None and self.task.done():
            queue.put_nowait(None)  # finished before anyone listened (e.g. a cache hit)
        self.listeners.add(queue)
        return queue


class ResultCache:
    def __init__(self, directory: Path, ttl_hours: int):
        self.directory, self.ttl = directory, ttl_hours * 3600

    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > self.ttl:
                path.unlink(missing_ok=True)
                return None
            entry = json.loads(path.read_text(encoding="utf-8"))
            return entry if entry.get("version") == CACHE_VERSION else None
        except (OSError, ValueError, AttributeError):
            return None

    def put(self, key: str, entry: dict):
        if not self.enabled:
            return
        if entry.get("en") is None or entry.get("breaks") is None:
            # Never let a write for one option discard what another request
            # already produced for the same transcript (the English track, or
            # the programme/break blocks).
            old = self.get(key)
            if old and old.get("bg") == entry.get("bg"):
                for name in ("en", "breaks"):
                    if entry.get(name) is None and old.get(name) is not None:
                        entry = {**entry, name: old[name], f"{name}_model": old.get(f"{name}_model")}
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self._path(key).with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self._path(key))

    def sweep(self):
        if not self.directory.is_dir():
            return
        for path in self.directory.glob("*.json"):
            try:
                if not self.enabled or time.time() - path.stat().st_mtime > self.ttl:
                    path.unlink()
            except OSError:
                pass


class SubtitleService:
    def __init__(self, settings: Settings, transcriber, translator, detector=None, costs=None):
        self.settings = settings
        self.transcriber = transcriber   # async (audio_path) -> Scribe JSON
        self.translator = translator     # .translate(texts, label, on_progress, media) -> list[str]
        self.detector = detector         # .detect(cues, silences, duration, label, media) -> list[block]
        self.costs = costs or CostTracker.from_settings(settings)   # what every provider request cost
        self.cache = ResultCache(settings.data_dir / "cache", settings.cache_ttl_hours)
        self.slots = asyncio.Semaphore(settings.max_concurrent_jobs)
        self.jobs: dict[str, Job] = {}                       # "<sha>:<translate>:<breaks>" -> request job
        self.transcriptions: dict[str, asyncio.Task] = {}    # "<sha>" -> shared transcription

    # ----- job management -------------------------------------------------
    def submit(self, upload: Upload, translate: bool, breaks: bool = False) -> Job:
        """Start processing, or join an identical upload that is already running."""
        key = f"{upload.sha256}:{int(translate)}:{int(breaks)}"
        if key in self.jobs:
            shutil.rmtree(upload.path.parent, ignore_errors=True)
            return self.jobs[key]
        job = Job()
        job.task = task = asyncio.create_task(self._process(upload, translate, breaks, job.emit))
        self.jobs[key] = job

        def finished(_):
            self.jobs.pop(key, None)
            for queue in job.listeners:
                queue.put_nowait(None)  # end-of-stream marker
            if not task.cancelled() and task.exception() is not None:
                error = task.exception()
                if not isinstance(error, PipelineError):
                    log.error("Processing failed unexpectedly", exc_info=error)
        task.add_done_callback(finished)
        return job

    def status(self, sha256: str) -> tuple[dict | None, Event | None]:
        """(stored result or None, progress of a running job for this file or None)."""
        progress = next(({**job.last, "elapsed_seconds": round(time.monotonic() - job.started, 1)}
                         for key, job in self.jobs.items() if key.startswith(sha256 + ":")), None)
        return self.cache.get(sha256), progress

    # ----- the pipeline ---------------------------------------------------
    async def _process(self, upload: Upload, translate: bool, breaks: bool,
                       emit: Callable[[Event], None]) -> dict:
        started = time.perf_counter()
        since = CostTracker.now()   # provider requests recorded from here on are this request's
        timings: dict[str, int] = {}

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000)
            return now

        try:
            entry = self.cache.get(upload.sha256)
            cached = entry is not None
            if entry is None:
                entry = await self._shared_transcription(upload, emit, lap)
            entry = dict(entry)  # this request's own copy
            translation = {"status": "skipped", "model": None, "error": None}
            if translate and entry["cues"]:
                if entry.get("en") is None:
                    translation = await self._translate(upload, entry, emit, lap)
                else:
                    translation = {"status": "complete", "model": entry.get("en_model"), "error": None}
            detection = {"status": "skipped", "model": None, "error": None}
            if breaks and entry["cues"]:
                if entry.get("breaks") is None:
                    detection = await self._detect_breaks(upload, entry, emit, lap)
                else:
                    detection = {"status": "complete", "model": entry.get("breaks_model"), "error": None}
            timings["total"] = round((time.perf_counter() - started) * 1000)
            cost = self.costs.cost_of(upload.sha256, since)
            result = build_result(entry, translate, translation, cached, timings, upload.filename,
                                  breaks, detection, cost)
            log.info("id=%s size=%d duration=%.1fs cues=%d cached=%s translation=%s breaks=%s "
                     "cost=$%.4f timings=%s", upload.sha256[:12], upload.size, entry["duration"] or 0,
                     len(entry["cues"]), cached, translation["status"], detection["status"],
                     cost["request"]["usd"], timings)
            return result
        finally:
            shutil.rmtree(upload.path.parent, ignore_errors=True)

    async def _shared_transcription(self, upload: Upload, emit, lap) -> dict:
        """Transcribe each file once, however many requests (with any options) want it."""
        sha = upload.sha256
        task = self.transcriptions.get(sha)
        if task is None:
            async def run() -> dict:
                emit({"stage": "queued"})
                async with self.slots:
                    entry = await self._transcribe(upload, emit, lap)
                self.cache.put(sha, entry)
                return entry

            task = self.transcriptions[sha] = asyncio.create_task(run())

            def done(finished: asyncio.Task):
                if self.transcriptions.get(sha) is finished:
                    del self.transcriptions[sha]
                if not finished.cancelled():
                    finished.exception()  # retrieved here; every waiter re-raises it
            task.add_done_callback(done)
        else:
            emit({"stage": "transcribing"})
        return await asyncio.shield(task)

    async def _transcribe(self, upload: Upload, emit, lap) -> dict:
        mark = time.perf_counter()
        emit({"stage": "probing"})
        limit_minutes = self.settings.max_duration_minutes
        too_long = PipelineError(413, f"The media is longer than the {limit_minutes}-minute limit.")
        try:
            info = await asyncio.to_thread(media.probe, upload.path)
            if info.duration and info.duration > limit_minutes * 60:
                raise too_long  # declared length already over: skip the extraction
            emit({"stage": "extracting"})
            audio = upload.path.with_name("audio.mp3")
            # The declared length can be missing or wrong (VBR/ADTS estimates), so the
            # limit is enforced on the measured audio, and extraction stops just past it.
            duration = await asyncio.to_thread(media.extract_audio, upload.path, audio, limit_minutes * 60)
            if duration > limit_minutes * 60:
                raise too_long
        except media.MediaError as exc:
            raise PipelineError(exc.status, str(exc)) from None
        upload.path.unlink(missing_ok=True)  # free disk before the long remote step
        mark = lap("extract", mark)

        emit({"stage": "transcribing", "duration_seconds": round(duration or 0, 1)})
        # The silences are measured while Scribe works (seconds of CPU, no cost)
        # and kept with the transcript, so break detection asked for later, even
        # on a cached result, never needs the media again.
        silence_scan = asyncio.create_task(asyncio.to_thread(self._silences, audio))
        try:
            transcript = await self.transcriber(audio)
        except TranscriptionError as exc:
            silence_scan.cancel()
            raise PipelineError(exc.status, self.settings.redact(str(exc))) from None
        silences = await silence_scan
        mark = lap("transcribe", mark)
        # Paid now, whatever happens next: recorded before the cues are built.
        seconds, source = billed_audio_seconds(transcript, duration)
        self.costs.record_transcription(Media(upload.sha256, upload.filename, duration), STT_MODEL, seconds, source)

        emit({"stage": "segmenting"})
        try:
            cues = subtitles.generate_cues(transcript, duration)
        except ValueError as exc:
            raise PipelineError(422, f"The transcript could not be turned into subtitles: {exc}") from None
        lap("segment", mark)
        return {"version": CACHE_VERSION, "id": upload.sha256, "filename": upload.filename,
                "media_type": info.media_type, "duration": duration, "cues": cues,
                "bg": subtitles.to_vtt(cues), "en": None, "en_model": None,
                "silences": silences, "breaks": None, "breaks_model": None}

    @staticmethod
    def _silences(audio: Path) -> list[list[float]]:
        try:
            return [list(pair) for pair in media.detect_silences(audio)]
        except media.MediaError as exc:
            log.warning("Silence detection skipped: %s", exc)
            return []

    async def _translate(self, upload: Upload, entry: dict, emit, lap) -> dict:
        mark = time.perf_counter()
        emit({"stage": "translating", "completed": 0, "total": None})
        try:
            english = await self.translator.translate(
                [cue["text"] for cue in entry["cues"]], label=upload.sha256[:16],
                on_progress=lambda done, total: emit({"stage": "translating", "completed": done, "total": total}),
                media=Media(upload.sha256, upload.filename, entry["duration"]))
        except TranslationError as exc:
            lap("translate", mark)
            return {"status": "failed", "model": TRANSLATION_MODEL, "error": self.settings.redact(str(exc))}
        except Exception:
            log.exception("Translation failed unexpectedly")
            lap("translate", mark)
            return {"status": "failed", "model": TRANSLATION_MODEL, "error": "English translation failed unexpectedly."}
        lap("translate", mark)
        en_cues = [{**cue, "text": subtitles.wrap_english(text)} for cue, text in zip(entry["cues"], english)]
        entry.update(en=subtitles.to_vtt(en_cues), en_model=TRANSLATION_MODEL)
        self.cache.put(upload.sha256, entry)
        return {"status": "complete", "model": TRANSLATION_MODEL, "error": None}

    async def _detect_breaks(self, upload: Upload, entry: dict, emit, lap) -> dict:
        mark = time.perf_counter()
        emit({"stage": "detecting_breaks"})
        failed = {"status": "failed", "model": BREAKS_MODEL, "error": None}
        try:
            found = await self.detector.detect(entry["cues"], entry.get("silences") or [],
                                               entry["duration"], label=upload.sha256[:16],
                                               media=Media(upload.sha256, upload.filename, entry["duration"]))
        except BreakDetectionError as exc:
            lap("detect_breaks", mark)
            return {**failed, "error": self.settings.redact(str(exc))}
        except Exception:
            log.exception("Break detection failed unexpectedly")
            lap("detect_breaks", mark)
            return {**failed, "error": "Programme/break detection failed unexpectedly."}
        lap("detect_breaks", mark)
        entry.update(breaks=found, breaks_model=BREAKS_MODEL)
        self.cache.put(upload.sha256, entry)
        return {"status": "complete", "model": BREAKS_MODEL, "error": None}


SKIPPED = {"status": "skipped", "model": None, "error": None}


def build_result(entry: dict, translate: bool, translation: dict, cached: bool,
                 timings: dict[str, int], filename: str | None = None,
                 breaks: bool = False, detection: dict | None = None, cost: dict | None = None) -> dict:
    if translate and not entry["cues"]:
        translation = dict(SKIPPED)
    english = entry.get("en") if translate else None
    if translate and not entry["cues"]:
        english = "WEBVTT\n"
    detection = dict(detection or SKIPPED)
    blocks = entry.get("breaks") if breaks else None
    if breaks and not entry["cues"]:
        blocks, detection = [], dict(SKIPPED)
    program = next((b["start"] for b in blocks or [] if b["kind"] == "program"), None)
    return {
        "id": entry["id"],
        "filename": filename or entry.get("filename") or "upload",
        "media_type": entry["media_type"],
        "duration_seconds": round(entry["duration"], 3) if entry["duration"] else None,
        "cue_count": len(entry["cues"]),
        "subtitles": {"bg": entry["bg"], "en": english},
        "translation": translation,
        "breaks": {**detection, "program_starts": program, "blocks": blocks},
        "cached": cached,
        "cost": cost or EMPTY_COST,
        "timings_ms": timings,
    }
