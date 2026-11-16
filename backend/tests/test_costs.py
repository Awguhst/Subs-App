"""Provider spending: priced from reported usage at the list rates, kept across restarts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from app.costs import (ANTHROPIC_PRICES, CostError, CostTracker, Media, anthropic_cost,
                       billed_audio_seconds)

MEDIA = Media("a" * 64, "clip.mp4", 3.0)


def usage(**counts):
    base = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation": None, "service_tier": "standard", "speed": "standard"}
    return SimpleNamespace(**{**base, **counts})


# ----- pricing ----------------------------------------------------------------
def test_haiku_4_5_is_priced_per_token_with_cache_multipliers():
    amount, counts, rates = anthropic_cost("claude-haiku-4-5-20251001", usage(
        input_tokens=1000, output_tokens=200, cache_read_input_tokens=400, cache_creation_input_tokens=300))
    # $1 / $5 per MTok; cache reads at 0.1x, 5-minute cache writes at 1.25x the input price.
    assert amount == Decimal("0.002415")
    assert counts["cache_write_5m_tokens"] == 300 and counts["cache_write_1h_tokens"] == 0
    assert rates["input"] == 1 and rates["output"] == 5 and rates["cache_read"] == 0.1 and rates["cache_write_5m"] == 1.25


def test_one_hour_cache_writes_cost_twice_the_input_price():
    breakdown = SimpleNamespace(ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=200)
    amount, counts, _ = anthropic_cost("claude-haiku-4-5-20251001",
                                       usage(cache_creation_input_tokens=300, cache_creation=breakdown))
    assert counts["cache_write_5m_tokens"] == 100 and counts["cache_write_1h_tokens"] == 200
    assert amount == Decimal("0.000525")   # 100 x 1.25 + 200 x 2, per million


def test_haiku_5_5_moves_to_the_long_context_tier_above_100k_prompt_tokens():
    short, _, short_rates = anthropic_cost("claude-haiku-5-5", usage(input_tokens=99_000, cache_read_input_tokens=1000,
                                                                    output_tokens=1000))
    long, _, long_rates = anthropic_cost("claude-haiku-5-5", usage(input_tokens=99_001, cache_read_input_tokens=1000,
                                                                  output_tokens=1000))
    assert short == Decimal("0.01041") and short_rates["prompt_tokens_up_to"] == 100_000   # 9900 + 10 + 500 microdollars
    assert long == Decimal("0.0520505") and long_rates["prompt_tokens_up_to"] is None   # 49500.5 + 50 + 2500
    assert long_rates["input"] == 0.5 and long_rates["output"] == 2.5 and long_rates["cache_read"] == 0.05


def test_batch_tier_costs_half_and_unknown_models_or_tiers_are_unpriced():
    half, counts, _ = anthropic_cost("claude-haiku-4-5-20251001", usage(input_tokens=1000, service_tier="batch"))
    assert half == Decimal("0.0005") and counts["service_tier"] == "batch"
    for model, extra in (("claude-opus-5-5", {}), ("claude-haiku-4-5-20251001", {"service_tier": "priority"}),
                         ("claude-haiku-4-5-20251001", {"speed": "fast"})):
        amount, counts, rates = anthropic_cost(model, usage(input_tokens=1000, **extra))
        assert amount is None and rates == {} and counts["input_tokens"] == 1000


def test_bad_usage_values_count_as_zero():
    amount, counts, _ = anthropic_cost("claude-haiku-4-5-20251001", usage(input_tokens=None, output_tokens=-5,
                                                                          cache_read_input_tokens=True))
    assert amount == 0 and counts["input_tokens"] == 0 and counts["output_tokens"] == 0


def test_elevenlabs_bills_the_reported_audio_length_or_the_measured_one():
    assert billed_audio_seconds({"audio_duration_secs": 90.5, "words": []}, 88.0) == (90.5, "audio_duration_secs")
    for transcript in ({"words": []}, {"audio_duration_secs": None}, {"audio_duration_secs": "90"},
                       {"audio_duration_secs": 0}, {"audio_duration_secs": float("nan")}, ["not", "a", "dict"]):
        assert billed_audio_seconds(transcript, 88.0) == (88.0, "measured")


@pytest.mark.parametrize("rate", [-1, float("inf"), float("nan"), "free"])
def test_invalid_scribe_rates_are_rejected(tmp_path, rate):
    with pytest.raises(CostError):
        CostTracker(tmp_path / "costs.jsonl", rate)


# ----- records ----------------------------------------------------------------
def test_records_are_priced_stored_and_summarised_across_restarts(tmp_path):
    path = tmp_path / "costs.jsonl"
    tracker = CostTracker(path, 0.20)
    assert tracker.record_transcription(MEDIA, "scribe_v2", 90, "audio_duration_secs") == Decimal("0.005")
    assert tracker.record_anthropic(MEDIA, "translation", "claude-haiku-4-5-20251001",
                                    usage(input_tokens=500, output_tokens=100)) == Decimal("0.001")
    assert tracker.record_anthropic(MEDIA, "breaks", "claude-haiku-5-5",
                                    usage(input_tokens=500, output_tokens=100)) == Decimal("0.0001")
    other = Media("b" * 64, "memo.m4a", 600.0)
    tracker.record_transcription(other, "scribe_v2", 600, "measured")

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["purpose"] for line in lines] == ["transcription", "translation", "breaks", "transcription"]
    assert lines[0]["cost_usd"] == "0.005000000" and lines[0]["usage"] == {"audio_seconds": 90.0, "source": "audio_duration_secs"}
    assert lines[1]["usage"]["input_tokens"] == 500 and lines[1]["rates"]["output"] == 5
    assert lines[1]["media_id"] == MEDIA.id and lines[1]["filename"] == "clip.mp4" and lines[1]["duration_seconds"] == 3.0

    restarted = CostTracker(path, 0.20)   # nothing is kept in memory between processes
    summary = restarted.summary()
    assert summary["total_usd"] == pytest.approx(0.005 + 0.001 + 0.0001 + 600 * 0.20 / 3600, abs=1e-6)
    assert summary["requests"] == 4 and summary["media_count"] == 2 and summary["unpriced_requests"] == 0
    assert summary["services"]["elevenlabs"] == {"usd": pytest.approx(0.038333), "requests": 2, "audio_seconds": 690.0}
    assert summary["services"]["anthropic"] == {"usd": 0.0011, "requests": 2, "input_tokens": 1000, "output_tokens": 200,
                                                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    assert summary["purposes"]["breaks"] == {"usd": 0.0001, "requests": 1}
    assert summary["models"]["claude-haiku-4-5-20251001"] == {"usd": 0.001, "requests": 1}
    assert [entry["id"] for entry in summary["media"]] == [other.id, MEDIA.id]   # most recently used first
    clip = summary["media"][1]
    assert clip["filename"] == "clip.mp4" and clip["duration_seconds"] == 3.0 and clip["requests"] == 3
    assert clip["usd"] == pytest.approx(0.0061) and clip["services"] == {"elevenlabs": 0.005, "anthropic": 0.0011}
    assert clip["purposes"] == {"transcription": 0.005, "translation": 0.001, "breaks": 0.0001}
    assert summary["pricing"]["elevenlabs"]["scribe_usd_per_audio_hour"] == 0.2
    assert summary["pricing"]["anthropic_usd_per_mtok"]["claude-haiku-5-5"][1]["input"] == 0.5
    assert summary["first_at"] == lines[0]["at"] and summary["last_at"] == lines[3]["at"]

    detail = restarted.media(MEDIA.id)
    assert detail["usd"] == pytest.approx(0.0061) and len(detail["records"]) == 3
    assert detail["records"][2]["cost_usd"] == 0.0001 and detail["records"][2]["model"] == "claude-haiku-5-5"
    assert not any(key.startswith("_") for key in detail["records"][0])
    assert restarted.media("c" * 64) is None


def test_the_cost_of_one_request_is_what_was_recorded_since_it_started(tmp_path):
    tracker = CostTracker(tmp_path / "costs.jsonl", 0.20)
    tracker.record_transcription(MEDIA, "scribe_v2", 3600, "measured")
    started = tracker.now()
    tracker.record_anthropic(MEDIA, "translation", "claude-haiku-4-5-20251001", usage(input_tokens=1000))
    tracker.record_anthropic(Media("b" * 64), "translation", "claude-haiku-4-5-20251001", usage(input_tokens=1000))
    cost = tracker.cost_of(MEDIA.id, started)
    assert cost == {"currency": "USD",
                    "request": {"usd": 0.001, "services": {"elevenlabs": 0.0, "anthropic": 0.001}, "requests": 1},
                    "media": {"usd": 0.201, "services": {"elevenlabs": 0.2, "anthropic": 0.001}, "requests": 2}}
    assert tracker.cost_of(MEDIA.id)["request"]["requests"] == 0
    assert tracker.cost_of("c" * 64)["media"]["usd"] == 0


def test_summaries_can_be_narrowed_by_time_and_recording(tmp_path):
    tracker = CostTracker(tmp_path / "costs.jsonl", 0.20)
    tracker.record_transcription(MEDIA, "scribe_v2", 3600, "measured")
    middle = tracker.now()
    tracker.record_transcription(Media("b" * 64), "scribe_v2", 1800, "measured")
    assert tracker.summary(since=middle)["total_usd"] == 0.1
    assert tracker.summary(until=middle)["total_usd"] == 0.2
    naive = (middle - timedelta(0)).replace(tzinfo=None)   # naive moments are read as UTC
    assert tracker.summary(since=naive)["requests"] == 1
    assert tracker.summary(media_id=MEDIA.id)["media"][0]["id"] == MEDIA.id
    assert tracker.summary(since=datetime(2099, 1, 1, tzinfo=timezone.utc))["requests"] == 0
    assert tracker.summary(limit=1)["media_count"] == 2 and len(tracker.summary(limit=1)["media"]) == 1


def test_unpriced_and_damaged_records_never_hide_the_rest(tmp_path):
    path = tmp_path / "costs.jsonl"
    tracker = CostTracker(path, 0.20)
    tracker.record_anthropic(MEDIA, "translation", "claude-opus-5-5", usage(input_tokens=1000))
    tracker.record_transcription(MEDIA, "scribe_v2", 3600, "measured")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"v": 1, "service": "elevenlabs"\n')          # a write cut short by a crash
        handle.write('{"v": 2, "service": "elevenlabs", "purpose": "x", "model": "m", "media_id": "a", '
                     '"usage": {}, "at": "2026-01-01T00:00:00+00:00", "cost_usd": "1"}\n')   # unknown version
        handle.write('{"v": 1, "service": "anthropic", "purpose": "x", "model": "m", "media_id": "a", '
                     '"usage": {}, "at": "2026-01-01T00:00:00+00:00", "cost_usd": "-1"}\n')  # impossible amount
    summary = tracker.summary()
    assert summary["requests"] == 2 and summary["unpriced_requests"] == 1 and summary["unreadable_lines"] == 3
    assert summary["total_usd"] == 0.2
    assert summary["media"][0]["unpriced_requests"] == 1
    assert tracker.media(MEDIA.id)["records"][0]["cost_usd"] is None


def test_concurrent_recorders_lose_nothing(tmp_path):
    path = tmp_path / "costs.jsonl"

    def record(index):
        CostTracker(path, 0.20).record_transcription(Media(str(index)), "scribe_v2", 36, "measured")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, range(40)))
    summary = CostTracker(path, 0.20).summary()
    assert summary["requests"] == 40 and summary["unreadable_lines"] == 0
    assert summary["total_usd"] == pytest.approx(40 * 36 * 0.20 / 3600)


def test_new_records_are_seen_without_a_restart(tmp_path):
    path = tmp_path / "costs.jsonl"
    reader, writer = CostTracker(path, 0.20), CostTracker(path, 0.20)
    assert reader.summary()["requests"] == 0
    writer.record_transcription(MEDIA, "scribe_v2", 60, "measured")
    assert reader.summary()["requests"] == 1
    writer.record_transcription(MEDIA, "scribe_v2", 60, "measured")
    assert reader.summary()["requests"] == 2


def test_price_list_covers_the_pinned_models():
    from app.breaks import MODEL as BREAKS_MODEL
    from app.translation import MODEL as TRANSLATION_MODEL
    for model in (TRANSLATION_MODEL, BREAKS_MODEL):
        assert any(model.startswith(prefix) for prefix in ANTHROPIC_PRICES), model
