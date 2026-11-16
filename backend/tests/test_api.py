"""End to end through HTTP: real FFmpeg, fake ElevenLabs and Claude (no credits used)."""
import asyncio
import hashlib
import json
import os
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.elevenlabs import TranscriptionError
from app.main import create_app
from app.translation import Translator
from fakes import BREAKS_MODEL, FakeAnthropic, FakeTranscriber, anthropic, api_error, echo, scribe, with_breaks

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
        assert client.post("/v1/spending").status_code == 401
        assert client.post("/v1/spending/media", json={"id": "0" * 64}).status_code == 401
        assert client.post("/v1/spending", headers={"X-API-Key": "secret-two"}).status_code == 200
        assert client.post("/v1/spending/media", json={"id": "0" * 64}, headers={"X-API-Key": "secret-two"}).status_code == 404
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
    assert response.status_code == 413 and "1-minute limit" in response.json()["detail"]
    assert transcriber.calls == []


def test_duration_limit_uses_the_real_length_not_the_files_claim(build, media_files):
    # Declares ~12 s but holds 75 s of audio: rejected before any paid transcription.
    client, transcriber, _ = build(max_duration_minutes=1)
    with client:
        response = upload(client, media_files["lying_long"], translate="false")
    assert response.status_code == 413 and transcriber.calls == []


def test_speech_past_a_wrong_declared_length_still_gets_subtitles(build, media_files):
    # Declares ~7 s but holds 30 s; speech continues to 28 s (e.g. VBR phone recordings).
    words = [{"type": "word", "text": f"дума{i}", "start": i * 2.0, "end": i * 2.0 + 0.5, "speaker_id": "s"}
             for i in range(15)]
    client, transcriber, _ = build(transcriber=FakeTranscriber(result={"words": words}))
    with client:
        response = upload(client, media_files["lying_short"], translate="false")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["duration_seconds"] == pytest.approx(30, abs=0.5) and "дума14" in body["subtitles"]["bg"]


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


class HeldTranscriber(FakeTranscriber):
    """Waits until released, so several requests can be in flight at once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = threading.Event()

    async def __call__(self, audio_path):
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        return await super().__call__(audio_path)


def send_together(client, transcriber, requests):
    """POST (path, filename, params) requests in parallel while transcription is held."""
    results = [None] * len(requests)

    def send(index, path, filename, params):
        with path.open("rb") as handle:
            results[index] = client.post("/v1/subtitles", params=params,
                                         files={"file": (filename, handle, "video/mp4")})
    threads = [threading.Thread(target=send, args=(i, *request)) for i, request in enumerate(requests)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 10
    while not client.app.state.service.transcriptions and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.4)  # let the other uploads arrive and join
    transcriber.release.set()
    for thread in threads:
        thread.join(20)
    return results


def test_concurrent_identical_uploads_share_one_transcription(build, media_files):
    transcriber = HeldTranscriber()
    client, _, _ = build(transcriber=transcriber)
    with client:
        results = send_together(client, transcriber, [(media_files["video"], "clip.mp4", {})] * 3)
    assert [r.status_code for r in results] == [200, 200, 200]
    assert len(transcriber.calls) == 1


def test_mixed_options_share_one_transcription_and_keep_english(build, media_files):
    transcriber = HeldTranscriber()
    client, _, claude = build(transcriber=transcriber)
    video = media_files["video"]
    with client:
        both, bg_only, en_file = send_together(client, transcriber, [
            (video, "clip.mp4", {}), (video, "clip.mp4", {"translate": "false"}),
            (video, "clip.mp4", {"format": "vtt", "language": "en"})])
        assert [r.status_code for r in (both, bg_only, en_file)] == [200, 200, 200]
        assert len(transcriber.calls) == 1  # not once per option combination
        assert bg_only.json()["subtitles"]["en"] is None and en_file.text.startswith("WEBVTT")
        # The Bulgarian-only request must not have erased the English track.
        later = client.get(f"/v1/subtitles/{both.json()['id']}/en.vtt")
    assert later.status_code == 200 and later.text == both.json()["subtitles"]["en"]
    assert len(claude.calls) == 1


def test_joined_requests_get_their_own_filename(build, media_files):
    transcriber = HeldTranscriber()
    client, _, _ = build(transcriber=transcriber)
    with client:
        alice, bob = send_together(client, transcriber, [
            (media_files["video"], "alice-private-interview.mp4", {"format": "vtt", "language": "bg"}),
            (media_files["video"], "bob.mp4", {"format": "vtt", "language": "bg"})])
    assert 'filename="alice-private-interview.bg.vtt"' in alice.headers["content-disposition"]
    assert 'filename="bob.bg.vtt"' in bob.headers["content-disposition"]
    assert "alice" not in bob.headers["content-disposition"]


def test_raw_upload_accepts_an_unencoded_utf8_filename(build, media_files):
    client, _, _ = build()
    with client:
        response = client.post("/v1/subtitles", content=media_files["video"].read_bytes(),
                               params={"translate": "false"},
                               headers={"Content-Type": "application/octet-stream",
                                        "X-Filename": "запис.mp4".encode("utf-8")})
    assert response.status_code == 200 and response.json()["filename"] == "запис.mp4"


def test_startup_only_removes_stale_temporary_uploads(build, tmp_path):
    work = tmp_path / "data" / "tmp"
    fresh, stale = work / "another-process-upload", work / "crash-leftover"
    for folder in (fresh, stale):
        folder.mkdir(parents=True)
        (folder / "upload").write_bytes(b"x")
    hours_ago = time.time() - 4 * 3600
    os.utime(stale, (hours_ago, hours_ago))
    client, _, _ = build()
    with client:
        pass
    assert fresh.is_dir() and not stale.exists()


BLOCKS = [{"first_line": 1, "kind": "break", "label": "Adverts: Lidl"},
          {"first_line": 2, "kind": "program", "label": "Игри на волята"}]


def test_breaks_come_with_the_subtitles_and_are_kept_for_later(build, media_files):
    client, transcriber, claude = build(claude=FakeAnthropic(with_breaks(BLOCKS)))
    with client:
        body = upload(client, media_files["gapped"], breaks="true").json()
        assert body["cue_count"] == 2
        assert body["breaks"]["status"] == "complete" and body["breaks"]["model"] == "claude-haiku-5-5"
        assert body["breaks"]["error"] is None
        found = body["breaks"]["blocks"]
        assert [s["kind"] for s in found] == ["break", "program"]
        assert found[0]["start"] == 0.0 and found[0]["transition"] is None
        assert found[0]["end"] == found[1]["start"] and found[1]["end"] == body["duration_seconds"]
        low, high = found[1]["transition"]
        assert low <= found[1]["start"] <= high
        assert body["breaks"]["program_starts"] == found[1]["start"]
        assert body["subtitles"]["en"] and body["timings_ms"]["detect_breaks"] >= 0
        models = [call["model"] for call in claude.calls]
        assert models.count("claude-haiku-5-5") == 1 and len(transcriber.calls) == 1

        # Stored with the transcript: no new provider calls for the same file.
        again = upload(client, media_files["gapped"], breaks="true").json()
        assert again["breaks"]["blocks"] == found and again["cached"] is True
        later = client.get(f"/v1/subtitles/{body['id']}/breaks")
        assert later.status_code == 200
        assert later.json()["blocks"] == found and later.json()["program_starts"] == found[1]["start"]
        assert later.json()["status"] == "complete" and later.json()["id"] == body["id"]
        assert client.get(f"/v1/subtitles/{body['id']}").json()["breaks"]["blocks"] == found
        assert len(claude.calls) == len(models)

        # Not asked for: not returned, and not computed.
        other = upload(client, media_files["video"]).json()
        assert other["breaks"] == {"status": "skipped", "model": None, "error": None, "program_starts": None, "blocks": None}
        assert client.get(f"/v1/subtitles/{other['id']}/breaks").status_code == 404
        assert client.get(f"/v1/subtitles/{'0' * 64}/breaks").status_code == 404


def test_break_detection_failure_still_returns_the_subtitles(build, media_files):
    def handler(request):
        if request["model"] == BREAKS_MODEL:
            return api_error(anthropic.InternalServerError, 500)
        return echo(request)
    client, _, _ = build(claude=FakeAnthropic(handler))
    with client:
        body = upload(client, media_files["video"], breaks="true").json()
    assert body["subtitles"]["en"].startswith("WEBVTT") and body["breaks"]["blocks"] is None
    assert body["breaks"]["status"] == "failed" and "server error" in body["breaks"]["error"]


def test_breaks_need_claude_and_the_json_response(build, media_files, tmp_path):
    client, transcriber, _ = build()
    with client:
        response = upload(client, media_files["video"], format="vtt", language="bg", breaks="true")
        assert response.status_code == 400 and "breaks=true" in response.json()["detail"]
    assert transcriber.calls == []
    settings = Settings(elevenlabs_api_key="el-key", data_dir=tmp_path / "no-claude")
    transcriber = FakeTranscriber()
    client = TestClient(create_app(settings, transcriber=transcriber, translator=Translator(settings)))
    with client:
        response = upload(client, media_files["video"], breaks="true")
        assert response.status_code == 503 and "ANTHROPIC_API_KEY" in response.json()["detail"]
        assert client.get("/health").json()["break_detection_model"] == BREAKS_MODEL
    assert transcriber.calls == []


def test_silent_recording_has_no_breaks_to_detect(build, media_files):
    client, _, claude = build(transcriber=FakeTranscriber(result={"words": []}),
                              claude=FakeAnthropic(with_breaks(BLOCKS)))
    with client:
        body = upload(client, media_files["video"], breaks="true").json()
    assert body["breaks"]["blocks"] == [] and body["breaks"]["status"] == "skipped" and claude.calls == []


# ----- what it cost -------------------------------------------------------------
def test_every_provider_request_is_priced_and_reported(build, media_files):
    transcriber = FakeTranscriber(result={**scribe(), "audio_duration_secs": 90.0})
    client, _, claude = build(transcriber=transcriber, claude=FakeAnthropic(with_breaks(BLOCKS)))
    transcription = 90 * 0.20 / 3600                 # Scribe v2: $0.20 per audio hour, pro rata
    translation = (500 * 1 + 100 * 5) / 1e6          # Haiku 4.5: $1 / $5 per MTok (the fake answers 500 in, 100 out)
    detection = (500 * 0.10 + 100 * 0.50) / 1e6      # Haiku 5.5: $0.10 / $0.50 per MTok
    with client:
        body = upload(client, media_files["gapped"], breaks="true").json()
        cost = body["cost"]
        assert cost["currency"] == "USD" and cost["request"]["requests"] == 3
        assert cost["request"]["usd"] == pytest.approx(transcription + translation + detection, abs=1e-6)
        assert cost["request"]["services"]["elevenlabs"] == pytest.approx(transcription, abs=1e-6)
        assert cost["request"]["services"]["anthropic"] == pytest.approx(translation + detection, abs=1e-6)
        assert cost["media"] == cost["request"]
        assert all(call["service_tier"] == "standard_only" for call in claude.calls)   # the standard rates always apply

        # Served from the cache: nothing new is spent, and the file's total is unchanged.
        again = upload(client, media_files["gapped"], breaks="true").json()
        assert again["cached"] is True and again["cost"]["request"]["usd"] == 0
        assert again["cost"]["media"] == cost["media"]
        fetched = client.get(f"/v1/subtitles/{body['id']}").json()
        assert fetched["cost"]["request"]["requests"] == 0 and fetched["cost"]["media"] == cost["media"]

        summary = client.post("/v1/spending").json()
        assert summary["total_usd"] == pytest.approx(cost["media"]["usd"], abs=1e-6)
        assert summary["requests"] == 3 and summary["media_count"] == 1 and summary["unpriced_requests"] == 0
        assert summary["services"]["elevenlabs"]["audio_seconds"] == 90.0
        assert summary["services"]["anthropic"]["input_tokens"] == 1000
        assert summary["services"]["anthropic"]["output_tokens"] == 200
        assert summary["purposes"]["breaks"]["usd"] == pytest.approx(detection, abs=1e-6)
        assert summary["models"]["scribe_v2"]["requests"] == 1 and summary["models"][BREAKS_MODEL]["requests"] == 1
        assert summary["pricing"]["elevenlabs"]["scribe_usd_per_audio_hour"] == 0.2
        row = summary["media"][0]
        assert row["id"] == body["id"] and row["filename"] == "gapped.m4a"
        assert row["duration_seconds"] == pytest.approx(3, abs=0.1)
        assert row["purposes"]["transcription"] == pytest.approx(transcription, abs=1e-6)
        assert row["services"]["anthropic"] == pytest.approx(translation + detection, abs=1e-6)

        detail = client.post("/v1/spending/media", json={"id": body["id"]}).json()
        assert [r["purpose"] for r in detail["records"]] == ["transcription", "translation", "breaks"]
        assert detail["records"][0]["usage"] == {"audio_seconds": 90.0, "source": "audio_duration_secs"}
        assert detail["records"][1]["usage"]["input_tokens"] == 500
        assert detail["records"][1]["cost_usd"] == pytest.approx(translation)
        assert client.post("/v1/spending/media", json={"id": "0" * 64}).status_code == 404
        assert client.post("/v1/spending/media", json={"id": "nope"}).status_code == 422
        assert client.post("/v1/spending", json={"media_id": "0" * 64}).json()["requests"] == 0
        assert client.post("/v1/spending", json={"limit": 0}).status_code == 422

    # The records are on disk: a restarted server reports the same totals.
    client, _, _ = build()
    with client:
        restarted = client.post("/v1/spending", json={}).json()
    assert restarted["total_usd"] == summary["total_usd"] and restarted["requests"] == 3
    assert restarted["media"] == summary["media"]


def test_transcription_is_billed_on_the_measured_length_when_scribe_reports_none(build, media_files):
    client, _, _ = build()
    with client:
        body = upload(client, media_files["audio"], translate="false").json()
        detail = client.post("/v1/spending/media", json={"id": body["id"]}).json()
    record = detail["records"][0]
    assert record["usage"]["source"] == "measured"
    assert record["usage"]["audio_seconds"] == pytest.approx(body["duration_seconds"], abs=1e-3)
    assert body["cost"]["request"]["usd"] == pytest.approx(body["duration_seconds"] * 0.20 / 3600, abs=1e-6)
    assert body["cost"]["request"]["services"]["anthropic"] == 0


def test_a_failed_translation_still_records_the_paid_transcription(build, media_files):
    claude = FakeAnthropic(lambda request: api_error(anthropic.AuthenticationError, 401))
    client, _, _ = build(claude=claude)
    with client:
        body = upload(client, media_files["video"]).json()
    assert body["translation"]["status"] == "failed"
    assert body["cost"]["request"]["requests"] == 1 and body["cost"]["request"]["services"]["anthropic"] == 0
    assert body["cost"]["request"]["services"]["elevenlabs"] > 0


def test_the_scribe_rate_is_configurable(build, media_files):
    client, _, _ = build(elevenlabs_usd_per_hour=0.40)
    with client:
        body = upload(client, media_files["audio"], translate="false").json()
        summary = client.post("/v1/spending").json()
    assert body["cost"]["request"]["usd"] == pytest.approx(body["duration_seconds"] * 0.40 / 3600, abs=1e-6)
    assert summary["pricing"]["elevenlabs"]["scribe_usd_per_audio_hour"] == 0.4
