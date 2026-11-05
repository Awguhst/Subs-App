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
from .elevenlabs import TranscriptionError
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
    def __init__(self, settings: Settings, transcriber, translator):
        self.settings = settings
        self.transcriber = transcriber   # async (audio_path) -> Scribe JSON
        self.translator = translator     # .translate(texts, label, on_progress) -> list[str]
        self.cache = ResultCache(settings.data_dir / "cache", settings.cache_ttl_hours)
        self.slots = asyncio.Semaphore(settings.max_concurrent_jobs)
        self.jobs: dict[str, Job] = {}

    # ----- job management -------------------------------------------------
    def submit(self, upload: Upload, translate: bool) -> Job:
        """Start processing, or join an identical upload that is already running."""
        key = f"{upload.sha256}:{int(translate)}"
        if key in self.jobs:
            shutil.rmtree(upload.path.parent, ignore_errors=True)
            return self.jobs[key]
        job = Job()
        job.task = task = asyncio.create_task(self._process(upload, translate, job.emit))
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

    def status(self, sha256: str) -> tuple[str, Any]:
        for key, job in self.jobs.items():
            if key.startswith(sha256 + ":"):
                return "processing", {**job.last, "elapsed_seconds": round(time.monotonic() - job.started, 1)}
        entry = self.cache.get(sha256)
        if entry is None:
            return "unknown", None
        return "complete", entry

    # ----- the pipeline ---------------------------------------------------
    async def _process(self, upload: Upload, translate: bool, emit: Callable[[Event], None]) -> dict:
        started = time.perf_counter()
        timings: dict[str, int] = {}

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000)
            return now

        try:
            entry = self.cache.get(upload.sha256)
            cached = entry is not None
            if entry is None:
                emit({"stage": "queued"})
                async with self.slots:
                    entry = await self._transcribe(upload, emit, lap)
                self.cache.put(upload.sha256, entry)
            translation = {"status": "skipped", "model": None, "error": None}
            if translate and entry["cues"]:
                if entry.get("en") is None:
                    translation = await self._translate(upload, entry, emit, lap)
                else:
                    translation = {"status": "complete", "model": entry.get("en_model"), "error": None}
            timings["total"] = round((time.perf_counter() - started) * 1000)
            result = build_result(entry, translate, translation, cached, timings, upload.filename)
            log.info("id=%s size=%d duration=%.1fs cues=%d cached=%s translation=%s timings=%s",
                     upload.sha256[:12], upload.size, entry["duration"] or 0, len(entry["cues"]),
                     cached, translation["status"], timings)
            return result
        finally:
            shutil.rmtree(upload.path.parent, ignore_errors=True)

    async def _transcribe(self, upload: Upload, emit, lap) -> dict:
        mark = time.perf_counter()
        emit({"stage": "probing"})
        try:
            info = await asyncio.to_thread(media.probe, upload.path)
            limit = self.settings.max_duration_minutes * 60
            if info.duration and info.duration > limit:
                raise PipelineError(413, f"The media is {info.duration / 60:.0f} minutes long; "
                                         f"the limit is {self.settings.max_duration_minutes} minutes.")
            emit({"stage": "extracting"})
            audio = upload.path.with_name("audio.mp3")
            duration = await asyncio.to_thread(media.extract_audio, upload.path, audio, info.duration)
            if duration and duration > limit:
                raise PipelineError(413, f"The media is longer than {self.settings.max_duration_minutes} minutes.")
        except media.MediaError as exc:
            raise PipelineError(exc.status, str(exc)) from None
        upload.path.unlink(missing_ok=True)  # free disk before the long remote step
        mark = lap("extract", mark)

        emit({"stage": "transcribing", "duration_seconds": round(duration or 0, 1)})
        try:
            transcript = await self.transcriber(audio)
        except TranscriptionError as exc:
            raise PipelineError(exc.status, self.settings.redact(str(exc))) from None
        mark = lap("transcribe", mark)

        emit({"stage": "segmenting"})
        try:
            cues = subtitles.generate_cues(transcript, duration)
        except ValueError as exc:
            raise PipelineError(422, f"The transcript could not be turned into subtitles: {exc}") from None
        lap("segment", mark)
        return {"version": CACHE_VERSION, "id": upload.sha256, "filename": upload.filename,
                "media_type": info.media_type, "duration": duration, "cues": cues,
                "bg": subtitles.to_vtt(cues), "en": None, "en_model": None}

    async def _translate(self, upload: Upload, entry: dict, emit, lap) -> dict:
        mark = time.perf_counter()
        emit({"stage": "translating", "completed": 0, "total": None})
        try:
            english = await self.translator.translate(
                [cue["text"] for cue in entry["cues"]], label=upload.sha256[:16],
                on_progress=lambda done, total: emit({"stage": "translating", "completed": done, "total": total}))
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


def build_result(entry: dict, translate: bool, translation: dict, cached: bool,
                 timings: dict[str, int], filename: str | None = None) -> dict:
    if translate and not entry["cues"]:
        translation = {"status": "skipped", "model": None, "error": None}
    english = entry.get("en") if translate else None
    if translate and not entry["cues"]:
        english = "WEBVTT\n"
    return {
        "id": entry["id"],
        "filename": filename or entry.get("filename") or "upload",
        "media_type": entry["media_type"],
        "duration_seconds": round(entry["duration"], 3) if entry["duration"] else None,
        "cue_count": len(entry["cues"]),
        "subtitles": {"bg": entry["bg"], "en": english},
        "translation": translation,
        "cached": cached,
        "timings_ms": timings,
    }
