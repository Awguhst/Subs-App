"""End to end through HTTP: real FFmpeg, fake ElevenLabs and Claude (no credits used)."""
import hashlib
import json
import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.elevenlabs import TranscriptionError
from app.main import create_app
from app.translation import Translator
from fakes import FakeAnthropic, FakeTranscriber, anthropic, api_error

TIMING = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$", re.M)


@pytest.fixture
def build(tmp_path):
    def factory(transcriber=None, claude=None, **overrides):
        options = {"elevenlabs_api_key": "el-key", "anthropic_api_key": "an-key", "data_dir": tmp_path / "data"}
        settings = Settings(**{**options, **overrides})
        transcriber = transcriber or FakeTranscriber()
        claude = claude or FakeAnthropic()
        app = create_app(settings, transcriber=transcriber, translator=Translator(settings, client=claude))
        return TestClient(app), transcriber, claude
    return factory


def upload(client, path, **params):
    with path.open("rb") as handle:
        return client.post("/v1/subtitles", params=params, files={"file": (path.name, handle, "video/mp4")})


def test_two_posts_return_ready_to_use_vtt_files(build, media_files):
    client, transcriber, claude = build()
    with client:
        bg = upload(client, media_files["video"], format="vtt", language="bg")
        en = upload(client, media_files["video"], format="vtt", language="en")
    for response, language in ((bg, "bg"), (en, "en")):
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "text/vtt; charset=utf-8"
        assert f'filename="clip.{language}.vtt"' in response.headers["content-disposition"]
        assert response.text.startswith("WEBVTT\n\n1\n")  # the body is the file itself
    assert "Здравейте" in bg.text and "EN Здравейте" in en.text
    assert TIMING.findall(bg.text) == TIMING.findall(en.text)
    assert bg.headers["x-cue-count"] == str(len(TIMING.findall(bg.text)))
    assert bg.headers["x-subtitle-id"] == en.headers["x-subtitle-id"]
    assert len(transcriber.calls) == 1 and len(claude.calls) == 1  # second request: cache, no cost


def test_vtt_downloads_keep_cyrillic_filenames(build, media_files):
    client, _, _ = build()
    with client, media_files["audio"].open("rb") as handle:
        response = client.post("/v1/subtitles", params={"format": "vtt", "language": "bg"},
                               files={"file": ("Запис от срещата.m4a", handle, "audio/mp4")})
        assert response.status_code == 200, response.text
        expected = "filename*=UTF-8''%D0%97%D0%B0%D0%BF%D0%B8%D1%81%20%D0%BE%D1%82%20"
        assert expected in response.headers["content-disposition"]
        again = client.get(f"/v1/subtitles/{response.headers['x-subtitle-id']}/bg.vtt")
    assert again.status_code == 200 and again.text == response.text
    assert again.headers["content-disposition"] == response.headers["content-disposition"]


@pytest.mark.parametrize("params, status", [
    ({"format": "vtt"}, 400),                                         # which language?
    ({"format": "vtt", "language": "en", "translate": "false"}, 400),
    ({"format": "vtt", "language": "bg", "stream": "true"}, 400),
    ({"format": "vtt", "language": "fr"}, 422),
    ({"format": "srt", "language": "bg"}, 422),
])
def test_vtt_mode_is_validated_before_any_work(build, media_files, params, status):
    client, transcriber, _ = build()
    with client:
        assert upload(client, media_files["video"], **params).status_code == status
    assert transcriber.calls == []


def test_vtt_mode_reports_a_failed_or_unavailable_translation(build, media_files, tmp_path):
    claude = FakeAnthropic(lambda request: api_error(anthropic.AuthenticationError, 401))
    client, _, _ = build(claude=claude)
    with client:
        assert upload(client, media_files["video"], format="vtt", language="bg").status_code == 200
        failed = upload(client, media_files["video"], format="vtt", language="en")
    assert failed.status_code == 502 and "ANTHROPIC_API_KEY" in failed.json()["detail"]

    settings = Settings(elevenlabs_api_key="el-key", data_dir=tmp_path / "no-translation")
    transcriber = FakeTranscriber()
    with TestClient(create_app(settings, transcriber=transcriber)) as client:
        missing = upload(client, media_files["video"], format="vtt", language="en")
        assert upload(client, media_files["video"], format="vtt", language="bg").status_code == 200
    assert missing.status_code == 503 and len(transcriber.calls) == 1


def test_one_request_returns_bulgarian_and_english_vtt(build, media_files, tmp_path):
    client, transcriber, claude = build()
    with client:
        response = upload(client, media_files["video"])
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == hashlib.sha256(media_files["video"].read_bytes()).hexdigest()
        assert body["media_type"] == "video" and body["filename"] == "clip.mp4"
        assert body["duration_seconds"] == pytest.approx(3, abs=0.1)
        bg, en = body["subtitles"]["bg"], body["subtitles"]["en"]
        assert bg.startswith("WEBVTT\n\n1\n") and "Здравейте" in bg
        assert en.startswith("WEBVTT\n\n1\n") and "EN Здравейте" in en
        assert TIMING.findall(bg) == TIMING.findall(en) and len(TIMING.findall(bg)) == body["cue_count"] > 1
        assert body["translation"] == {"status": "complete", "model": "claude-haiku-4-5-20251001", "error": None}
        assert body["cached"] is False and {"extract", "transcribe", "translate", "total"} <= set(body["timings_ms"])
        assert len(transcriber.calls) == 1 and len(claude.calls) == 1
        assert list((tmp_path / "data" / "tmp").iterdir()) == []  # upload and audio removed


def test_identical_upload_is_served_from_cache_without_provider_calls(build, media_files):
    client, transcriber, claude = build()
    with client:
        first = upload(client, media_files["audio"]).json()
        second = upload(client, media_files["audio"]).json()
    assert second["cached"] is True and second["subtitles"] == first["subtitles"]
    assert second["media_type"] == "audio"
    assert len(transcriber.calls) == 1 and len(claude.calls) == 1


def test_raw_body_upload_works_for_simple_clients(build, media_files):
    client, _, _ = build()
    with client:
        response = client.post("/v1/subtitles", content=media_files["video"].read_bytes(),
                               headers={"Content-Type": "application/octet-stream", "X-Filename": "phone%20clip.mov"})
    assert response.status_code == 200
    assert response.json()["filename"] == "phone clip.mov"


def test_bulgarian_only_and_later_translation_reuse_the_transcript(build, media_files):
    client, transcriber, claude = build()
    with client:
        only_bg = upload(client, media_files["video"], translate="false").json()
        assert only_bg["subtitles"]["en"] is None and only_bg["translation"]["status"] == "skipped"
        assert claude.calls == []
        both = upload(client, media_files["video"]).json()
    assert both["subtitles"]["en"] and both["cached"] is True
    assert len(transcriber.calls) == 1 and len(claude.calls) == 1


def test_stream_mode_reports_progress_then_result(build, media_files):
    client, _, _ = build()
    with client:
        response = upload(client, media_files["video"], stream="true")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["event"] == "accepted" and events[-1]["event"] == "result"
    stages = [e["stage"] for e in events if e["event"] == "progress"]
    for stage in ("extracting", "transcribing", "segmenting", "translating"):
        assert stage in stages
    assert events[-1]["result"]["subtitles"]["en"].startswith("WEBVTT")


def test_finished_results_can_be_fetched_by_id_and_downloaded(build, media_files):
    client, _, _ = build()
    with client:
        body = upload(client, media_files["video"]).json()
        fetched = client.get(f"/v1/subtitles/{body['id']}")
        assert fetched.status_code == 200 and fetched.json()["subtitles"] == body["subtitles"]
        vtt = client.get(f"/v1/subtitles/{body['id']}/en.vtt")
        assert vtt.status_code == 200 and vtt.headers["content-type"].startswith("text/vtt")
        assert vtt.text == body["subtitles"]["en"]
        assert 'filename="clip.en.vtt"' in vtt.headers["content-disposition"]
        assert client.get(f"/v1/subtitles/{'0' * 64}").status_code == 404
        assert client.get("/v1/subtitles/not-a-hash").status_code == 422


def test_api_keys_protect_everything_except_health_and_ui(build, media_files):
    client, transcriber, _ = build(api_keys=("secret-one", "secret-two"))
    with client:
        assert upload(client, media_files["video"]).status_code == 401
        assert client.get("/v1/usage").status_code == 401
        assert client.get("/v1/usage", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/v1/usage", headers={"X-API-Key": "secret-two"}).status_code == 200
        assert client.get("/v1/usage", headers={"Authorization": "Bearer secret-one"}).status_code == 200
        assert client.get("/health").json()["auth_required"] is True
        assert client.get("/").status_code == 200
    assert transcriber.calls == []


def test_translation_failure_still_returns_bulgarian(build, media_files):
    claude = FakeAnthropic(lambda request: api_error(anthropic.AuthenticationError, 401))
    client, _, _ = build(claude=claude)
    with client:
        body = upload(client, media_files["video"]).json()
    assert body["subtitles"]["bg"].startswith("WEBVTT") and body["subtitles"]["en"] is None
    assert body["translation"]["status"] == "failed" and "ANTHROPIC_API_KEY" in body["translation"]["error"]


def test_silent_recording_returns_empty_tracks(build, media_files):
    client, _, claude = build(transcriber=FakeTranscriber(result={"words": []}))
    with client:
        body = upload(client, media_files["video"]).json()
    assert body["cue_count"] == 0 and body["subtitles"] == {"bg": "WEBVTT\n", "en": "WEBVTT\n"}
    assert claude.calls == []


@pytest.mark.parametrize("name, status", [("text", 415), ("playlist", 415), ("silent_video", 422)])
def test_unusable_files_are_rejected_before_paid_calls(build, media_files, name, status):
    client, transcriber, _ = build()
    with client:
        response = upload(client, media_files[name])
    assert response.status_code == status and response.json()["detail"]
    assert transcriber.calls == []


def test_limits_are_enforced(build, media_files, tmp_path):
    client, transcriber, _ = build(max_upload_mb=1)
    with client:
        response = client.post("/v1/subtitles", content=b"x" * (1024 * 1024 + 1),
                               headers={"Content-Type": "application/octet-stream"})
        assert response.status_code == 413
        assert client.post("/v1/subtitles", content=b"").status_code == 400
        assert list((tmp_path / "data" / "tmp").iterdir()) == []
    client, transcriber, _ = build(max_duration_minutes=1, transcriber=transcriber)
    long_file = tmp_path / "long.m4a"
    from conftest import ffmpeg
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "61", "-c:a", "aac", str(long_file))
    with client:
        response = upload(client, long_file)
    assert response.status_code == 413 and "limit is 1 minutes" in response.json()["detail"]
    assert transcriber.calls == []


def test_provider_errors_map_to_gateway_statuses(build, media_files):
    failing = FakeTranscriber(error=TranscriptionError("ElevenLabs rejected el-key.", 502))
    client, _, _ = build(transcriber=failing)
    with client:
        response = upload(client, media_files["video"])
        assert response.status_code == 502
        assert "el-key" not in response.text and "[redacted]" in response.text
        streamed = upload(client, media_files["audio"], stream="true")
    last = json.loads(streamed.text.splitlines()[-1])
    assert last == {"event": "error", "status": 502, "detail": "ElevenLabs rejected [redacted]."}


def test_missing_transcription_key_is_reported_before_upload(build, media_files):
    client, _, _ = build(elevenlabs_api_key="")
    with client:
        assert upload(client, media_files["video"]).status_code == 503
        health = client.get("/health").json()
    assert health["status"] == "degraded" and health["transcription_configured"] is False


def test_health_docs_and_ui(build):
    client, _, _ = build()
    with client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["ffmpeg_available"] and health["auth_required"] is False
        spec = client.get("/openapi.json").json()
        body = spec["paths"]["/v1/subtitles"]["post"]["requestBody"]["content"]
        assert body["multipart/form-data"]["schema"]["properties"]["file"]["format"] == "binary"
        assert client.get("/docs").status_code == 200
        assert "<title>" in client.get("/").text
    client, _, _ = build(web_ui=False)
    with client:
        assert client.get("/", follow_redirects=False).headers["location"] == "/docs"


def test_concurrent_identical_uploads_share_one_transcription(build, media_files):
    import asyncio
    import threading

    gate = threading.Event()

    class SlowTranscriber(FakeTranscriber):
        async def __call__(self, audio_path):
            while not gate.is_set():
                await asyncio.sleep(0.01)
            return await super().__call__(audio_path)

    transcriber = SlowTranscriber()
    client, _, _ = build(transcriber=transcriber)
    results = []
    with client:
        threads = [threading.Thread(target=lambda: results.append(upload(client, media_files["video"])))
                   for _ in range(3)]
        for thread in threads:
            thread.start()
        import time
        deadline = time.monotonic() + 10
        while len(client.app.state.service.jobs) == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.3)  # let all three uploads arrive
        gate.set()
        for thread in threads:
            thread.join(20)
    assert [r.status_code for r in results] == [200, 200, 200]
    assert len(transcriber.calls) == 1
