"""FFprobe inspection and fast speech-audio extraction for untrusted uploads."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

# Only real audio/video container demuxers may open an upload. Playlist and
# indirection formats (HLS, DASH, concat, ffconcat, image sequences...) are
# excluded, so a crafted file cannot make FFmpeg read server files or URLs.
ALLOWED_DEMUXERS = ",".join((
    "mov", "mp4", "matroska", "webm", "avi", "flv", "asf", "mpegts", "mpeg", "mxf", "rm",
    "mp3", "aac", "wav", "w64", "ogg", "flac", "amr", "aiff", "caf", "au", "ac3", "eac3",
    "dts", "wv", "ape", "tta", "dsf", "mpc", "mpc8", "xwma", "voc",
))
SAFE_INPUT = ["-format_whitelist", ALLOWED_DEMUXERS, "-protocol_whitelist", "file"]


class MediaError(RuntimeError):
    def __init__(self, message: str, status: int = 415):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class MediaInfo:
    media_type: str            # "audio" or "video"
    duration: float | None     # None when the container does not record it
    has_audio: bool
    container: str


def _binary(name: str) -> str:
    return os.getenv(f"{name.upper()}_BIN", name)


def available() -> bool:
    return bool(shutil.which(_binary("ffmpeg")) and shutil.which(_binary("ffprobe")))


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, check=False,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except FileNotFoundError:
        raise MediaError("FFmpeg/FFprobe is not installed on the server.", status=500) from None
    except subprocess.TimeoutExpired:
        raise MediaError("Reading the media took too long. The file may be corrupt.", status=422) from None


def _duration(*values) -> float | None:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            return number
    return None


def probe(path: str | Path) -> MediaInfo:
    result = _run([_binary("ffprobe"), "-v", "error", *SAFE_INPUT, "-show_streams", "-show_format",
                   "-of", "json", str(Path(path).resolve())], timeout=60)
    if result.returncode:
        raise MediaError("This file is not a supported audio or video format.")
    try:
        data = json.loads(result.stdout)
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
    except (ValueError, AttributeError):
        raise MediaError("This file is not a supported audio or video format.") from None
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    # Cover art is stored as a one-frame "video" stream; it is not a video.
    video = [s for s in streams if s.get("codec_type") == "video"
             and not (s.get("disposition") or {}).get("attached_pic")]
    if not audio:
        raise MediaError("This file has no audio track to transcribe.", status=422)
    duration = _duration(fmt.get("duration"), audio[0].get("duration"))
    return MediaInfo("video" if video else "audio", duration, True, str(fmt.get("format_name", "")))


def extract_audio(source: str | Path, target: str | Path, duration: float | None = None) -> float | None:
    """Write 16 kHz mono MP3 (~29 MB per hour) for transcription; return its duration.

    Speech recognition needs neither video nor high-fidelity audio, so this keeps
    the upload to ElevenLabs small. LAME's faster algorithm (compression level 7)
    is ~2.5x quicker than the default with no meaningful loss for speech.
    """
    target = Path(target).resolve()
    timeout = 120 + (duration or 4 * 3600) / 4
    result = _run([_binary("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-v", "error", *SAFE_INPUT,
                   "-i", str(Path(source).resolve()), "-map", "0:a:0", "-vn", "-sn", "-dn",
                   "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "16000",
                   "-c:a", "libmp3lame", "-b:a", "64k", "-compression_level", "7",
                   "-map_metadata", "-1", str(target)], timeout=timeout)
    if result.returncode or not target.is_file() or target.stat().st_size == 0:
        raise MediaError("The audio track could not be decoded. " + result.stderr[-400:].strip(), status=422)
    if duration is not None:
        return duration
    # Browser/phone recordings (e.g. MediaRecorder WebM) often lack a duration
    # header; the freshly written MP3 always has one.
    return probe(target).duration
