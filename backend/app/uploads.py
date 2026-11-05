"""Stream an upload straight to disk while hashing it.

Accepts ``multipart/form-data`` (the ``file`` field, or the first file part) or
a raw request body of any other content type, e.g. ``curl --data-binary @clip.mp4``.
Bytes are written exactly once, never fully buffered in memory, and oversized
uploads are rejected as soon as they cross the limit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from urllib.parse import unquote

from fastapi import HTTPException, Request
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.concurrency import run_in_threadpool

FLUSH_BYTES = 1 << 20     # hand writes to a thread in ~1 MiB batches
MAX_FIELD_BYTES = 64 * 1024  # total size of any non-file form fields


@dataclass
class Upload:
    path: Path
    filename: str
    size: int
    sha256: str


def clean_filename(name: str) -> str:
    """A display name only; it is never used to build a path."""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if c.isprintable()).strip()
    return name[:200] or "upload"


class _Sink:
    def __init__(self, path: Path, limit: int):
        self.path, self.limit = path, limit
        self.size = 0
        self.digest = hashlib.sha256()
        self.pending: list[bytes] = []
        self.pending_size = 0
        self.handle = path.open("wb")

    def feed(self, data: bytes):
        self.size += len(data)
        if self.size > self.limit:
            raise HTTPException(413, f"The upload exceeds the {self.limit // (1024 * 1024)} MB limit.")
        self.pending.append(bytes(data))
        self.pending_size += len(data)

    def _write(self, chunks: list[bytes]):
        for chunk in chunks:
            self.digest.update(chunk)
            self.handle.write(chunk)

    async def flush(self, force: bool = False):
        if self.pending and (force or self.pending_size >= FLUSH_BYTES):
            chunks, self.pending, self.pending_size = self.pending, [], 0
            await run_in_threadpool(self._write, chunks)

    def close(self):
        self.handle.close()


async def receive_upload(request: Request, directory: Path, limit: int) -> Upload:
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit + MAX_FIELD_BYTES:
                raise HTTPException(413, f"The upload exceeds the {limit // (1024 * 1024)} MB limit.")
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length header.") from None
    directory.mkdir(parents=True, exist_ok=True)
    sink = _Sink(directory / "upload", limit)
    try:
        mime, options = parse_options_header(request.headers.get("content-type", ""))
        if mime == b"multipart/form-data":
            filename = await _receive_multipart(request, sink, options.get(b"boundary"))
        else:
            filename = unquote(request.headers.get("x-filename", ""))
            async for chunk in request.stream():
                sink.feed(chunk)
                await sink.flush()
        await sink.flush(force=True)
    finally:
        sink.close()
    if sink.size == 0:
        raise HTTPException(400, "The uploaded file is empty.")
    return Upload(sink.path, clean_filename(filename), sink.size, sink.digest.hexdigest())


async def _receive_multipart(request: Request, sink: _Sink, boundary: bytes | None) -> str:
    if not boundary:
        raise HTTPException(400, "The multipart request has no boundary.")
    part = {"headers": {}, "field": b"", "value": b"", "target": None}
    state = {"found": False, "filename": "", "other": 0}

    def on_part_begin():
        part.update(headers={}, field=b"", value=b"", target=None)

    def on_header_field(data, start, end):
        part["field"] += data[start:end]

    def on_header_value(data, start, end):
        part["value"] += data[start:end]

    def on_header_end():
        part["headers"][part["field"].lower()] = part["value"]
        part["field"], part["value"] = b"", b""

    def on_headers_finished():
        _, params = parse_options_header(part["headers"].get(b"content-disposition", b""))
        name, filename = params.get(b"name", b""), params.get(b"filename")
        if not state["found"] and (name == b"file" or filename is not None):
            state["found"] = True
            state["filename"] = (filename or b"").decode("utf-8", "replace")
            part["target"] = "file"

    def on_part_data(data, start, end):
        if part["target"] == "file":
            sink.feed(data[start:end])
        else:
            state["other"] += end - start
            if state["other"] > MAX_FIELD_BYTES:
                raise HTTPException(413, "Too much form data besides the file.")

    def on_part_end():
        part["target"] = None

    parser = MultipartParser(boundary, callbacks={
        "on_part_begin": on_part_begin, "on_header_field": on_header_field,
        "on_header_value": on_header_value, "on_header_end": on_header_end,
        "on_headers_finished": on_headers_finished, "on_part_data": on_part_data,
        "on_part_end": on_part_end})
    try:
        async for chunk in request.stream():
            parser.write(chunk)
            await sink.flush()
        parser.finalize()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "The multipart upload is malformed.") from None
    if not state["found"]:
        raise HTTPException(400, 'Send the media as a multipart form field named "file".')
    return state["filename"]
