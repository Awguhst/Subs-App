"""Bulgarian Subtitle API: one upload in, Bulgarian + English WebVTT out."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
import secrets
import shutil
import time
from typing import Literal
from urllib.parse import quote
import uuid

from fastapi import FastAPI, HTTPException, Path, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from . import __version__, media
from .config import Settings
from .costs import CostTracker
from .elevenlabs import Transcriber
from .pipeline import Job, PipelineError, SubtitleService, build_result
from .breaks import MODEL as BREAKS_MODEL, BreakDetector
from .translation import MODEL as TRANSLATION_MODEL, Translator
from .uploads import Upload, receive_upload

log = logging.getLogger("subtitles")
HEARTBEAT_SECONDS = 15
SHA256 = r"^[0-9a-f]{64}$"

DESCRIPTION = """
Upload an audio or video recording with Bulgarian speech and receive, **in one request**:

* `subtitles.bg`: Bulgarian subtitles transcribed with ElevenLabs Scribe v2 (WebVTT)
* `subtitles.en`: English subtitles translated by Claude, with exactly the same cue timings (WebVTT)

```bash
curl -X POST "$BASE_URL/v1/subtitles" -H "X-API-Key: $API_KEY" -F "file=@recording.mp4"
```

**Ready-to-use .vtt files.** Add `?format=vtt&language=bg` or `?format=vtt&language=en` and the
response is that subtitle file itself. Send one request per language with the same file: the second
is served from the cache at no extra provider cost.

```bash
curl -X POST "$BASE_URL/v1/subtitles?format=vtt&language=bg" -F "file=@recording.mp4" -o recording.bg.vtt
curl -X POST "$BASE_URL/v1/subtitles?format=vtt&language=en" -F "file=@recording.mp4" -o recording.en.vtt
```

**Any device, any format.** Send `multipart/form-data` with a `file` field, or the raw bytes as the
request body. Containers and codecs are detected from the content (MP4, MOV, MKV, WebM, AVI, MP3,
M4A, AAC, WAV, FLAC, OGG/Opus, AMR, 3GP, CAF, WMA ...), not from the filename.

**Where the show starts and the commercial breaks are.** Add `?breaks=true` and the JSON result
also carries `breaks`: the recording split into `program` and `break` blocks with start/end times
(one extra Claude request on the transcript, about one cent per hour of video; no extra speech
recognition). Also available afterwards as `GET /v1/subtitles/{id}/breaks`.

**Long recordings.** Add `?stream=true` to receive progress events as NDJSON while processing; the
connection stays active, which proxies with idle timeouts require. Identical uploads are cached
by their SHA-256 (`id`), so retrying after a dropped connection costs nothing; a finished result
can also be fetched with `GET /v1/subtitles/{id}`.

**What it costs.** Every result carries `cost`: what this request's provider calls cost at the
published list prices (ElevenLabs per audio hour, Claude per token, from the usage the providers
report) and what the file has cost in total. `POST /v1/spending` gives the totals for the server,
per service, purpose, model and recording; `POST /v1/spending/media` lists every provider request
behind one recording. The records persist in the data directory.

**Authentication.** When the server sets `API_KEYS`, send a key as `X-API-Key: <key>` or
`Authorization: Bearer <key>`.
"""

ERRORS = {
    400: {"description": "Empty or malformed upload"},
    401: {"description": "Missing or invalid API key"},
    413: {"description": "File larger than MAX_UPLOAD_MB or longer than MAX_DURATION_MINUTES"},
    415: {"description": "Not a supported audio/video format"},
    422: {"description": "The file has no decodable audio track"},
    502: {"description": "The transcription provider failed"},
    503: {"description": "Transcription is not configured on the server"},
    504: {"description": "The transcription provider timed out"},
}

UPLOAD_BODY = {"requestBody": {"required": True, "content": {
    "multipart/form-data": {"schema": {
        "type": "object", "required": ["file"],
        "properties": {"file": {"type": "string", "format": "binary",
                                "description": "Audio or video file with Bulgarian speech"}}}},
    "application/octet-stream": {"schema": {"type": "string", "format": "binary",
                                            "description": "Raw file bytes; optionally name it with an X-Filename header"}},
}}}

VTT_EXAMPLE = "WEBVTT\n\n1\n00:00:01.200 --> 00:00:03.400\nЗдравейте на всички!\n"

NDJSON_EXAMPLE = "\n".join(json.dumps(line) for line in (
    {"event": "accepted", "id": "9f2c…", "filename": "clip.mp4", "size_bytes": 48211033},
    {"event": "progress", "stage": "extracting", "elapsed_seconds": 0.4},
    {"event": "progress", "stage": "transcribing", "duration_seconds": 312.4, "elapsed_seconds": 1.6},
    {"event": "progress", "stage": "translating", "completed": 3, "total": 5, "elapsed_seconds": 21.0},
    {"event": "progress", "stage": "detecting_breaks", "elapsed_seconds": 29.5},
    {"event": "result", "result": {"id": "9f2c…", "subtitles": {"bg": "WEBVTT…", "en": "WEBVTT…"},
                                   "breaks": {"status": "complete", "program_starts": 71.4,
                                              "blocks": [{"start": 0.0, "end": 71.4, "kind": "break", "label": "Adverts: Lidl"},
                                                         {"start": 71.4, "end": 312.4, "kind": "program", "label": "Игри на волята"}]}}},
))


# ----- response models ---------------------------------------------------------
class Tracks(BaseModel):
    bg: str = Field(description="Bulgarian subtitles transcribed from the speech (WebVTT).",
                    examples=["WEBVTT\n\n1\n00:00:01.200 --> 00:00:03.400\nЗдравейте, как сте?\n"])
    en: str | None = Field(description="English subtitles with exactly the same cue timings (WebVTT); "
                                       "null when translation was not requested or failed.",
                           examples=["WEBVTT\n\n1\n00:00:01.200 --> 00:00:03.400\nHi, how are you?\n"])


class Translation(BaseModel):
    status: Literal["complete", "skipped", "failed"]
    model: str | None = None
    error: str | None = Field(None, description="Why English is missing when status is `failed`.")


class Block(BaseModel):
    start: float = Field(description="Seconds from the start of the recording.")
    end: float
    kind: Literal["program", "break"] = Field(description="`program`: the show itself. `break`: adverts and "
                                                          "the channel's own promos/trailers.")
    label: str = Field(description="The programme's name or the advertisers, when they can be told from the speech.")
    transition: list[float] | None = Field(
        description="[end of the last speech before, start of the first speech after]: the window inside "
                    "which the item changes. `start` is the middle of the longest silence in it. Null for "
                    "the first block, which begins at 0.")


class Breaks(BaseModel):
    status: Literal["complete", "skipped", "failed"] = Field(
        description="`complete`: `blocks` is filled. `skipped`: not requested, or no speech. `failed`: see `error`.")
    model: str | None = None
    error: str | None = Field(None, description="Why `blocks` is missing when status is `failed`.")
    program_starts: float | None = Field(None, description="Where the show starts: the start of the first "
                                                           "`program` block, in seconds.")
    blocks: list[Block] | None = Field(None, description="Programme and commercial-break blocks covering the "
                                                         "whole recording, in order; null unless `breaks=true` "
                                                         "was given and detection completed.")


class CostTotals(BaseModel):
    usd: float = Field(description="At the providers' published list prices (see `POST /v1/spending`).")
    services: dict[str, float] = Field(description="The same amount split by provider: `elevenlabs`, `anthropic`.")
    requests: int = Field(description="Provider requests behind this amount.")


class Cost(BaseModel):
    currency: Literal["USD"] = "USD"
    request: CostTotals = Field(description="The provider requests this HTTP request caused; all zero when "
                                            "everything came from the cache.")
    media: CostTotals = Field(description="Everything ever spent on this file: transcription, translation, "
                                          "break detection, retries included.")


class SubtitleResult(BaseModel):
    id: str = Field(description="SHA-256 of the uploaded bytes; use with `GET /v1/subtitles/{id}`.")
    filename: str
    media_type: Literal["audio", "video"]
    duration_seconds: float | None
    cue_count: int
    subtitles: Tracks
    translation: Translation
    breaks: Breaks = Field(description="Where the show and the commercial breaks are (`?breaks=true`).")
    cached: bool = Field(description="True when the transcription came from an identical earlier upload "
                                     "(no provider cost).")
    cost: Cost = Field(description="What the provider requests cost, for this request and for this file.")
    timings_ms: dict[str, int] = Field(description="Server-side duration of each stage in milliseconds.")


class BreaksResult(Breaks):
    id: str
    filename: str
    duration_seconds: float | None


class Processing(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    status: Literal["processing"]
    stage: str = Field(description="queued, probing, extracting, transcribing, segmenting, translating "
                                   "or detecting_breaks")
    elapsed_seconds: float


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    transcription_configured: bool
    translation_configured: bool
    ffmpeg_available: bool
    auth_required: bool
    translation_model: str
    break_detection_model: str
    limits: dict[str, int]


class Usage(BaseModel):
    cap_usd: float
    charged_usd: float
    reserved_usd: float
    remaining_usd: float
    currency: str
    euro_budget: float


class SpendingQuery(BaseModel):
    since: datetime | None = Field(None, description="Count only provider requests at or after this moment "
                                                     "(ISO 8601; UTC when no offset is given).")
    until: datetime | None = Field(None, description="Count only provider requests at or before this moment.")
    media_id: str | None = Field(None, pattern=SHA256, description="Count only one recording (its `id`).")
    limit: int = Field(100, ge=1, le=1000, description="How many recordings to list in `media` "
                                                       "(most recently used first).")


class Bucket(BaseModel):
    model_config = ConfigDict(extra="allow")
    usd: float
    requests: int


class MediaSpending(BaseModel):
    id: str = Field(description="The recording's `id` (SHA-256 of the file).")
    filename: str | None
    duration_seconds: float | None
    usd: float = Field(description="Everything spent on this recording.")
    services: dict[str, float] = Field(description="Split by provider: `elevenlabs`, `anthropic`.")
    purposes: dict[str, float] = Field(description="Split by purpose: `transcription`, `translation`, `breaks`.")
    requests: int
    unpriced_requests: int = Field(description="Requests whose model or tier is not in the price list "
                                               "(recorded with their usage, counted as $0).")
    first_at: str
    last_at: str


class Spending(BaseModel):
    currency: Literal["USD"]
    total_usd: float = Field(description="Every provider request at the published list prices.")
    requests: int
    unpriced_requests: int
    unrecorded_requests: int = Field(description="Requests since the server started that could not be written "
                                                 "to the records (disk error); their cost is unknown.")
    unreadable_lines: int = Field(description="Damaged lines in the records file that were skipped.")
    media_count: int
    services: dict[str, Bucket] = Field(description="`elevenlabs` (with `audio_seconds`) and `anthropic` "
                                                    "(with the token counts).")
    purposes: dict[str, Bucket] = Field(description="`transcription`, `translation`, `breaks`.")
    models: dict[str, Bucket]
    media: list[MediaSpending] = Field(description="Per recording, most recently used first.")
    first_at: str | None
    last_at: str | None
    pricing: dict = Field(description="The rates applied: USD per audio hour and per million tokens.")
    filters: dict


class MediaSpendingQuery(BaseModel):
    id: str = Field(pattern=SHA256, description="The recording's `id` (SHA-256 of the file).")


class MediaSpendingDetail(MediaSpending):
    records: list[dict] = Field(description="Every provider request for this recording, in order, with its "
                                            "usage, the rates applied and `cost_usd`.")


def _auth_dependency(settings: Settings):
    if not settings.auth_required:
        async def open_access():
            return None
        return open_access
    header = APIKeyHeader(name="X-API-Key", auto_error=False, scheme_name="ApiKey")
    bearer = HTTPBearer(auto_error=False, scheme_name="Bearer")
    keys = [key.encode("utf-8") for key in settings.api_keys]

    async def require_api_key(api_key: str | None = Security(header),
                              credentials: HTTPAuthorizationCredentials | None = Security(bearer)):
        supplied = (api_key or (credentials.credentials if credentials else "")).encode("utf-8")
        valid = False
        for key in keys:  # compare against every key: timing reveals nothing
            valid |= secrets.compare_digest(supplied, key)
        if not valid:
            raise HTTPException(401, "A valid API key is required (X-API-Key header or Authorization: Bearer).",
                                headers={"WWW-Authenticate": "Bearer"})
    return require_api_key


def create_app(settings: Settings | None = None, *, transcriber=None, translator=None,
               detector=None, costs: CostTracker | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    transcriber = transcriber or Transcriber(settings)
    translator = translator or Translator(settings, costs=costs)
    # One record of provider spending, shared by every stage.
    costs = costs or translator.costs
    translator.costs = costs
    # Break detection shares the translator's Anthropic client and spending ledger.
    detector = detector or BreakDetector(settings, client=translator.client, ledger=translator.ledger, costs=costs)
    service = SubtitleService(settings, transcriber, translator, detector, costs)
    auth = Security(_auth_dependency(settings))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_uploads(settings.data_dir / "tmp")
        service.cache.sweep()
        _log_startup_warnings(settings)
        yield
        for client in (transcriber, translator):
            if hasattr(client, "aclose"):
                await client.aclose()

    app = FastAPI(title="Bulgarian Subtitle API", version=__version__, description=DESCRIPTION,
                  lifespan=lifespan, openapi_tags=[
                      {"name": "Subtitles", "description": "Speech-to-text and translation"},
                      {"name": "Service", "description": "Health, spending and the translation cap"}])
    app.state.settings, app.state.service = settings, service
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6,
                       exclude_content_types=("application/x-ndjson", "text/event-stream"))
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["Authorization", "X-API-Key", "Content-Type", "X-Filename"],
                       expose_headers=["Content-Disposition", "X-Subtitle-Id", "X-Cue-Count"])

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _exc: Exception):
        return JSONResponse({"detail": "Internal server error."}, status_code=500)

    @app.post("/v1/subtitles", tags=["Subtitles"], dependencies=[auth], response_model=SubtitleResult,
              summary="Create Bulgarian + English subtitles from audio or video",
              openapi_extra=UPLOAD_BODY,
              responses={**ERRORS, 200: {"description": "JSON with both subtitle tracks; one ready-to-use "
                                                        "`.vtt` file when `format=vtt`; or an NDJSON event "
                                                        "stream when `stream=true`.",
                                         "content": {"text/vtt": {"example": VTT_EXAMPLE},
                                                     "application/x-ndjson": {"example": NDJSON_EXAMPLE}}}})
    async def create_subtitles(
        request: Request,
        translate: bool = Query(True, description="Also return English subtitles (`false` = Bulgarian only)."),
        stream: bool = Query(False, description="Stream NDJSON progress events, ending with a `result` "
                                                "or `error` event. Recommended for recordings over ~20 minutes."),
        format: Literal["json", "vtt"] = Query("json", description="`vtt` returns one subtitle file "
                                                                     "(choose it with `language`) instead of JSON."),
        language: Literal["bg", "en"] | None = Query(None, description="Track to return when `format=vtt`."),
        breaks: bool = Query(False, description="Also find where the show and the commercial breaks "
                                                "start and end (`breaks` in the JSON result)."),
    ):
        """Transcribes the Bulgarian speech and translates every cue into English.

        **JSON (default):** both WebVTT tracks in one response. If only the translation fails, the
        request still succeeds with the Bulgarian track, `subtitles.en = null` and the reason in
        `translation.error`.

        **`format=vtt&language=bg|en`:** the response *is* that `.vtt` file, ready to save. Make one
        request per language with the same file; the second is answered from the cache (no extra
        provider cost), and simultaneous requests share one transcription.

        **`breaks=true`:** `breaks` in the JSON result also lists the `program` and `break` blocks of
        the recording with their start and end times, and `program_starts` says where the show begins
        (one extra Claude request on the transcript).
        """
        if format == "vtt":
            if language is None:
                raise HTTPException(400, "Add language=bg or language=en to choose which .vtt file to return.")
            if stream:
                raise HTTPException(400, "stream=true returns NDJSON; it cannot be combined with format=vtt.")
            if breaks:
                raise HTTPException(400, "breaks=true needs the JSON response; it cannot be combined with format=vtt.")
            if language == "en" and not translate:
                raise HTTPException(400, "language=en needs the translation; remove translate=false.")
            if language == "en" and not translator.available:
                raise HTTPException(503, "English translation is not configured: set ANTHROPIC_API_KEY on the server.")
        if breaks and not detector.available:
            raise HTTPException(503, "Break detection is not configured: set ANTHROPIC_API_KEY on the server.")
        if not settings.transcription_configured:
            raise HTTPException(503, "Transcription is not configured: set ELEVENLABS_API_KEY on the server.")
        directory = settings.data_dir / "tmp" / uuid.uuid4().hex
        try:
            upload = await receive_upload(request, directory, settings.max_upload_bytes)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        job = service.submit(upload, translate, breaks)
        if stream:
            return StreamingResponse(_events(job, upload), media_type="application/x-ndjson",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            # Shielded: if the client goes away the work still finishes and is cached.
            result = await asyncio.shield(job.task)
        except PipelineError as exc:
            raise HTTPException(exc.status, exc.detail) from None
        # A request that joined someone else's identical upload gets its own filename back.
        result = {**result, "filename": upload.filename}
        if format == "json":
            return result
        vtt = result["subtitles"][language]
        if vtt is None:  # only English can be missing: its translation failed
            raise HTTPException(502, f"The English track could not be created: {result['translation']['error']}")
        return _vtt_file(vtt, result["filename"], language,
                         {"X-Subtitle-Id": result["id"], "X-Cue-Count": str(result["cue_count"])})

    @app.get("/v1/subtitles/{id}", tags=["Subtitles"], dependencies=[auth], response_model=SubtitleResult,
             summary="Fetch a result by id",
             responses={202: {"model": Processing, "description": "Still processing"},
                        404: {"description": "Unknown or expired id"}})
    async def get_subtitles(id: str = Path(pattern=SHA256, description="The `id` (SHA-256 of the file)")):
        """Returns a finished result (kept for `CACHE_TTL_HOURS`), or 202 while it is processing.

        `breaks.blocks` is filled when the breaks were requested for this file before, otherwise null."""
        entry, progress = service.status(id)
        complete = entry and entry.get("en") and entry.get("breaks") is not None
        if progress and not complete:
            return JSONResponse({"id": id, "status": "processing", **progress}, status_code=202)
        if entry is None:
            raise HTTPException(404, "No result for this id. It may have expired; upload the file again.")
        translation = ({"status": "complete", "model": entry.get("en_model"), "error": None} if entry.get("en")
                       else {"status": "skipped", "model": None, "error": None})
        found = entry.get("breaks") is not None
        detection = ({"status": "complete", "model": entry.get("breaks_model"), "error": None} if found
                     else {"status": "skipped", "model": None, "error": None})
        cost = await asyncio.to_thread(costs.cost_of, id)
        return build_result(entry, True, translation, True, {}, breaks=found, detection=detection, cost=cost)

    @app.get("/v1/subtitles/{id}/breaks", tags=["Subtitles"], dependencies=[auth], response_model=BreaksResult,
             summary="Where the show and the commercial breaks are",
             responses={202: {"model": Processing, "description": "Still processing"},
                        404: {"description": "Not available"}})
    async def get_breaks(id: str = Path(pattern=SHA256)):
        """The `program` / `break` blocks of a recording processed with `breaks=true`, and where the show starts."""
        entry, progress = service.status(id)
        blocks = entry.get("breaks") if entry else None
        if blocks is not None:
            program = next((b["start"] for b in blocks if b["kind"] == "program"), None)
            return {"id": id, "filename": entry.get("filename") or "upload",
                    "duration_seconds": round(entry["duration"], 3) if entry.get("duration") else None,
                    "status": "complete", "model": entry.get("breaks_model"), "error": None,
                    "program_starts": program, "blocks": blocks}
        if progress:
            return JSONResponse({"id": id, "status": "processing", **progress}, status_code=202)
        raise HTTPException(404, "No breaks for this id. Create them with POST /v1/subtitles?breaks=true.")

    @app.get("/v1/subtitles/{id}/{language}.vtt", tags=["Subtitles"], dependencies=[auth],
             summary="Download one track as a .vtt file",
             responses={200: {"content": {"text/vtt": {}}}, 202: {"model": Processing, "description": "Still processing"},
                        404: {"description": "Not available"}},
             response_class=Response)
    async def download_track(id: str = Path(pattern=SHA256), language: Literal["bg", "en"] = Path()):
        entry, progress = service.status(id)
        vtt = entry.get(language) if entry else None
        if vtt:
            return _vtt_file(vtt, entry.get("filename"), language, {"X-Subtitle-Id": id})
        if progress:
            return JSONResponse({"id": id, "status": "processing", **progress}, status_code=202)
        raise HTTPException(404, "This track is not available. Create it with POST /v1/subtitles first.")

    @app.post("/v1/spending", tags=["Service"], dependencies=[auth], response_model=Spending,
              summary="What the provider requests have cost")
    async def spending(query: SpendingQuery | None = None):
        """Totals for this server at the published list prices: overall, per service (ElevenLabs,
        Anthropic), per purpose (transcription, translation, breaks), per model and per recording.

        Every ElevenLabs and Claude request is priced from the usage the provider reported (audio
        seconds; input, cached and output tokens) and stored in the data directory, so the totals
        survive restarts. The body is optional; use it to narrow the period or pick one recording.
        """
        query = query or SpendingQuery()
        return await asyncio.to_thread(costs.summary, query.since, query.until, query.media_id, query.limit)

    @app.post("/v1/spending/media", tags=["Service"], dependencies=[auth], response_model=MediaSpendingDetail,
              summary="What one recording cost, request by request",
              responses={404: {"description": "No provider request was recorded for this id"}})
    async def media_spending(query: MediaSpendingQuery):
        """Every provider request made for one recording (by its `id`), with the usage reported,
        the rates applied and the resulting amount, plus the totals per service and purpose."""
        found = await asyncio.to_thread(costs.media, query.id)
        if found is None:
            raise HTTPException(404, "No spending is recorded for this id.")
        return found

    @app.get("/v1/usage", tags=["Service"], dependencies=[auth], response_model=Usage,
             summary="Translation spending against the lifetime cap")
    async def usage():
        return await asyncio.to_thread(translator.budget_status)

    @app.get("/health", tags=["Service"], response_model=Health, summary="Readiness and configuration")
    async def health():
        ffmpeg = media.available()
        return {"status": "ok" if ffmpeg and settings.transcription_configured else "degraded",
                "version": __version__,
                "transcription_configured": settings.transcription_configured,
                "translation_configured": settings.translation_configured,
                "ffmpeg_available": ffmpeg, "auth_required": settings.auth_required,
                "translation_model": TRANSLATION_MODEL, "break_detection_model": BREAKS_MODEL,
                "limits": {"max_upload_mb": settings.max_upload_mb,
                           "max_duration_minutes": settings.max_duration_minutes,
                           "max_concurrent_jobs": settings.max_concurrent_jobs}}

    index = settings.frontend_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def home():
        if settings.web_ui and index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return RedirectResponse("/docs")

    return app


def _remove_stale_uploads(directory, max_age_hours: float = 3):
    """Delete upload folders left behind by a crash. Recent ones may belong to
    another process sharing this data directory, so only old ones are removed."""
    if not directory.is_dir():
        return
    cutoff = time.time() - max_age_hours * 3600
    for path in directory.iterdir():
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
        except OSError:
            pass


def _vtt_file(vtt: str, filename: str | None, language: str, headers: dict[str, str]) -> Response:
    """A downloadable ``<name>.<language>.vtt`` response.

    The filename is sent ASCII-safe plus RFC 5987 UTF-8, so Cyrillic names
    (e.g. ``запис.bg.vtt``) download correctly instead of breaking the header.
    """
    name = filename or "subtitles"
    stem = name.rsplit(".", 1)[0] if "." in name.strip(".") else name
    stem = "".join("_" if c in '\\/:*?"<>|' or not c.isprintable() else c for c in stem).strip(" .") or "subtitles"
    target = f"{stem[:120]}.{language}.vtt"
    fallback = "".join(c if c.isascii() and (c.isalnum() or c in "-_. ") else "_" for c in target)
    disposition = f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(target, safe='')}"
    return Response(vtt, media_type="text/vtt; charset=utf-8",
                    headers={"Content-Disposition": disposition, **headers})


async def _events(job: Job, upload: Upload):
    def line(payload: dict) -> bytes:
        return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    def elapsed() -> float:
        return round(time.monotonic() - job.started, 1)

    queue = job.listen()
    try:
        yield line({"event": "accepted", "id": upload.sha256, "filename": upload.filename,
                    "size_bytes": upload.size})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                event = job.last  # heartbeat: keeps proxies from closing an idle connection
            if event is None:
                break
            yield line({"event": "progress", **event, "elapsed_seconds": elapsed()})
        try:
            yield line({"event": "result", "result": {**job.task.result(), "filename": upload.filename}})
        except PipelineError as exc:
            yield line({"event": "error", "status": exc.status, "detail": exc.detail})
        except Exception:
            yield line({"event": "error", "status": 500, "detail": "Internal server error."})
    finally:
        job.listeners.discard(queue)


def _log_startup_warnings(settings: Settings):
    import os
    if not settings.auth_required:
        log.warning("API_KEYS is not set: anyone who can reach this server can spend your provider credits.")
    if not settings.transcription_configured:
        log.warning("ELEVENLABS_API_KEY is not set: /v1/subtitles will answer 503.")
    if not settings.translation_configured:
        log.warning("ANTHROPIC_API_KEY is not set: only Bulgarian subtitles will be produced.")
    if not media.available():
        log.warning("FFmpeg/FFprobe not found on PATH.")
    if os.getenv("RAILWAY_ENVIRONMENT") and not os.getenv("RAILWAY_VOLUME_MOUNT_PATH"):
        log.warning("No Railway volume attached: the translation spending ledger, the spending records "
                    "and the result cache reset on every deploy. Attach a volume (mount path /data).")
