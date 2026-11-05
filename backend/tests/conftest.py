import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="FFmpeg/FFprobe not installed")


def ffmpeg(*args: str):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                   check=True, capture_output=True, timeout=60)


@pytest.fixture(scope="session")
def media_files(tmp_path_factory):
    """Small synthetic recordings in the formats phones and browsers produce."""
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("FFmpeg/FFprobe not installed")
    folder = tmp_path_factory.mktemp("media")
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    picture = ["-f", "lavfi", "-i", "color=c=0x182335:s=320x240:r=25"]
    files = {
        "video": folder / "clip.mp4", "audio": folder / "memo.m4a", "silent_video": folder / "no-audio.mp4",
        "streamed_webm": folder / "recorder.webm",
    }
    ffmpeg(*picture, *tone, "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(files["video"]))
    ffmpeg(*tone, "-t", "3", "-c:a", "aac", str(files["audio"]))
    ffmpeg(*picture, "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(files["silent_video"]))
    # Written through a pipe, like browser MediaRecorder output: no duration header.
    with files["streamed_webm"].open("wb") as handle:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *tone, "-t", "3", "-c:a", "libopus",
                        "-f", "webm", "pipe:1"], check=True, stdout=handle, timeout=60)
    # MP3s whose declared length is a wrong estimate: a 320 kbit/s start followed
    # by 32 kbit/s audio, with no Xing header, so FFprobe guesses from the first
    # frames and reports far less than the real duration.
    start, rest = folder / "hi.mp3", folder / "lo.mp3"
    ffmpeg(*tone, "-t", "5", "-c:a", "libmp3lame", "-b:a", "320k", "-write_xing", "0", str(start))
    for name, seconds in (("lying_short", 25), ("lying_long", 70)):
        ffmpeg(*tone, "-t", str(seconds), "-c:a", "libmp3lame", "-b:a", "32k", "-write_xing", "0", str(rest))
        files[name] = folder / f"{name}.mp3"
        files[name].write_bytes(start.read_bytes() + rest.read_bytes())
    files["playlist"] = folder / "evil.m3u8"
    files["playlist"].write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:3\n#EXTINF:3,\n{files['audio'].name}\n#EXT-X-ENDLIST\n")
    files["text"] = folder / "notes.mp4"
    files["text"].write_text("not a video")
    return files
