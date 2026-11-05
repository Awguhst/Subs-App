import pytest

from app import media


def test_video_and_audio_are_detected_from_content(media_files):
    video = media.probe(media_files["video"])
    assert video.media_type == "video" and video.has_audio and video.duration == pytest.approx(3, abs=0.1)
    audio = media.probe(media_files["audio"])
    assert audio.media_type == "audio" and audio.duration == pytest.approx(3, abs=0.1)


def test_extracts_small_mono_speech_audio(media_files, tmp_path):
    target = tmp_path / "audio.mp3"
    duration = media.extract_audio(media_files["video"], target, 3.0)
    assert duration == 3.0 and target.stat().st_size > 1000
    info = media.probe(target)
    assert info.media_type == "audio" and info.duration == pytest.approx(3, abs=0.1)


def test_browser_recordings_work_and_unknown_durations_are_measured(media_files, tmp_path):
    # MediaRecorder-style WebM/Opus written through a pipe (no seekable header).
    info = media.probe(media_files["streamed_webm"])
    assert info.media_type == "audio" and "webm" in info.container
    # When a container has no duration, it is measured on the extracted audio.
    assert media.extract_audio(media_files["streamed_webm"], tmp_path / "a.mp3", None) == pytest.approx(3, abs=0.1)


def test_real_duration_is_measured_and_extraction_is_capped(media_files, tmp_path):
    declared = media.probe(media_files["lying_short"]).duration
    assert declared < 15  # the container's estimate is badly wrong...
    measured = media.extract_audio(media_files["lying_short"], tmp_path / "a.mp3")
    assert measured == pytest.approx(30, abs=0.5)  # ...the extracted audio is measured
    capped = media.extract_audio(media_files["lying_long"], tmp_path / "b.mp3", max_seconds=60)
    assert capped == pytest.approx(61, abs=0.5)  # never decodes (or bills) far past a limit


def test_video_without_audio_is_rejected(media_files):
    with pytest.raises(media.MediaError, match="no audio") as error:
        media.probe(media_files["silent_video"])
    assert error.value.status == 422


@pytest.mark.parametrize("name", ["playlist", "text"])
def test_playlists_and_non_media_are_refused(media_files, name):
    # A playlist could make FFmpeg read other server files; it must never be opened.
    with pytest.raises(media.MediaError, match="not a supported") as error:
        media.probe(media_files[name])
    assert error.value.status == 415


def test_extraction_also_refuses_playlists(media_files, tmp_path):
    with pytest.raises(media.MediaError):
        media.extract_audio(media_files["playlist"], tmp_path / "a.mp3", 3.0)
