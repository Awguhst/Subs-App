"""Programme / commercial-break detection: line numbers in, exact times out, every request accounted."""
import asyncio

import pytest

from app import breaks as det
from app.config import Settings
from app.breaks import BreakDetector, BreakDetectionError
from fakes import FakeAnthropic, anthropic, api_error, claude_json, breaks_response


def cues(count: int, length: float = 2.0, gap: float = 1.0) -> list[dict]:
    """``count`` cues of ``length`` seconds with ``gap`` seconds of nothing between them."""
    return [{"start": round(i * (length + gap), 3), "end": round(i * (length + gap) + length, 3),
             "text": f"Реплика {i + 1}"} for i in range(count)]


def make(tmp_path, handler, budget=20.0):
    client = FakeAnthropic(handler)
    settings = Settings(anthropic_api_key="test-key", data_dir=tmp_path, translation_budget_usd=budget)
    return BreakDetector(settings, client=client), client


def detect(detector, items, silences=(), duration=None):
    return asyncio.run(detector.detect(items, list(silences), duration, label="test"))


def items_of(cue_list, silences=(), duration=None):
    return det.transcript_items(cue_list, list(silences), duration)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda seconds: real_sleep(0))


# ----- pure functions ----------------------------------------------------------
def test_transcript_lines_are_numbered_and_show_the_minute():
    items = items_of([{"start": 65.9, "end": 67.0, "text": "Здравейте  на\nвсички"}])
    # Nobody speaks for the first minute, so that stretch gets its own line before the cue.
    assert det.transcript_lines(items, first=12) == "12 [0:00] [no speech for 66 s]\n13 [1:05] Здравейте на всички"


def test_transcript_items_mark_long_gaps_and_broadcast_silences():
    spoken = [{"start": 2.0, "end": 4.0, "text": "Едно"}, {"start": 5.0, "end": 6.0, "text": "Две"},
              {"start": 9.0, "end": 10.0, "text": "Три"}, {"start": 40.0, "end": 41.0, "text": "Четири"}]
    silences = [(4.2, 4.6), (6.3, 8.1), (12.0, 15.0), (16.0, 16.4)]
    items = items_of(spoken, silences, duration=70.0)
    assert [i["kind"] for i in items] == ["cue", "cue", "gap", "cue", "gap", "cue", "gap"]
    assert items[2] == {"kind": "gap", "start": 6.0, "end": 9.0, "text": "[1.8 s of silence]"}
    assert items[4] == {"kind": "gap", "start": 10.0, "end": 40.0, "text": "[no speech for 30 s, including 3.0 s of silence]"}
    assert items[6] == {"kind": "gap", "start": 41.0, "end": 70.0, "text": "[no speech for 29 s]"}
    # Short gaps with only a brief pause are ordinary speech and get no line.
    assert [i["kind"] for i in items_of(spoken[:2], silences)] == ["cue", "cue"]
    # A recording that starts long before anyone speaks gets a leading marker.
    assert items_of([{"start": 20.0, "end": 21.0, "text": "Хей"}], [])[0]["text"] == "[no speech for 20 s]"
    assert "20 [0:10] [no speech for 30 s, including 3.0 s of silence]" in det.transcript_lines(items, first=16)


def test_parse_blocks_keeps_only_usable_blocks_in_cue_order():
    answer = [{"first_line": 30, "kind": "program", "label": "Шоу"},
              {"first_line": 5, "kind": "break", "label": "Lidl"},
              {"first_line": 5, "kind": "program", "label": "duplicate cue"},
              {"first_line": 99, "kind": "program", "label": "outside the chunk"},
              {"first_line": "7", "kind": "break", "label": "not an integer"},
              {"first_line": 10, "kind": "unknown", "label": "bad kind"},
              {"first_line": 20, "kind": "program", "label": "\x00bad\tlabel  " + "x" * 200}]
    blocks = det.parse_blocks(answer, first=1, last=40)
    # The first returned block is moved to the chunk's first cue, so nothing is left uncovered.
    assert [b["first_line"] for b in blocks] == [1, 20, 30]
    assert [b["kind"] for b in blocks] == ["break", "program", "program"]
    assert blocks[1]["label"] == ("bad label " + "x" * 200)[:120]
    for bad in ([], "nope", [{"first_line": 1}]):
        with pytest.raises(det._Retryable):
            det.parse_blocks(bad, first=1, last=40)


def test_merge_blocks_joins_neighbours_of_the_same_kind():
    merged = det.merge_blocks([{"first_line": 1, "kind": "break", "label": ""},
                               {"first_line": 4, "kind": "break", "label": "Lidl"},
                               {"first_line": 9, "kind": "program", "label": "Шоу"},
                               {"first_line": 12, "kind": "program", "label": "Шоу, part 2"}])
    assert merged == [{"first_line": 1, "kind": "break", "label": "Lidl"},
                      {"first_line": 9, "kind": "program", "label": "Шоу"}]


def test_cut_point_prefers_the_longest_silence_inside_the_window():
    silences = [(0.5, 0.9), (10.2, 10.5), (12.0, 13.6), (14.0, 14.1), (20.0, 25.0)]
    assert det.cut_point((10.0, 15.0), silences) == pytest.approx(12.8)   # middle of 12.0–13.6
    assert det.cut_point((13.0, 15.0), silences) == pytest.approx(13.3)   # a silence running into the window
    assert det.cut_point((15.0, 19.0), silences) == pytest.approx(17.0)   # no silence: the middle of the gap
    assert det.cut_point((30.0, 31.0), []) == pytest.approx(30.5)
    assert det.cut_point((5.0, 4.0), silences) == 5.0                     # overlapping cues: no gap


def test_build_blocks_covers_the_whole_recording_with_exact_boundaries():
    silences = [(14.4, 14.6)]  # inside the gap between cue 5 (ends 14.0) and cue 6 (starts 15.0)
    items = items_of(cues(10), silences)  # cue n spans [3(n-1), 3(n-1)+2]; no markers: the gaps are 1 s
    assert len(items) == 10
    blocks = [{"first_line": 1, "kind": "break", "label": "Lidl"},
              {"first_line": 6, "kind": "program", "label": "Шоу"},
              {"first_line": 9, "kind": "break", "label": "Promo"}]
    result = det.build_blocks(blocks, items, duration=31.0, silences=silences)
    assert [(s["start"], s["end"], s["kind"]) for s in result] == [
        (0.0, 14.5, "break"), (14.5, 23.5, "program"), (23.5, 31.0, "break")]
    assert result[0]["transition"] is None
    assert result[1]["transition"] == [14.0, 15.0] and result[2]["transition"] == [23.0, 24.0]
    # The recording never ends before its last cue, whatever the container claims.
    assert det.build_blocks(blocks, items, duration=None, silences=[])[-1]["end"] == items[-1]["end"]
    assert det.build_blocks([], items, 31.0, []) == [] and det.build_blocks(blocks, [], 31.0, []) == []


def test_a_block_can_start_at_a_marker_line_when_the_show_resumes_without_speech():
    spoken = cues(5) + [{"start": 60.0, "end": 61.0, "text": "Браво!"}]  # cue 5 ends at 14.0; 46 s of song
    silences = [(14.2, 16.0), (30.0, 30.5)]
    items = items_of(spoken, silences, duration=61.0)
    assert items[5]["text"] == "[no speech for 46 s, including 1.8 s of silence]"
    blocks = [{"first_line": 1, "kind": "break", "label": "Lidl"},
              {"first_line": 6, "kind": "program", "label": "Гласът на България"}]
    result = det.build_blocks(blocks, items, duration=61.0, silences=silences)
    assert [(s["start"], s["end"], s["kind"]) for s in result] == [(0.0, 15.1, "break"), (15.1, 61.0, "program")]
    assert result[1]["transition"] == [14.0, 60.0]
    # Starting at the cue after the marker gives the same window: the change lies between the two speeches.
    blocks[1]["first_line"] = 7
    assert det.build_blocks(blocks, items, duration=61.0, silences=silences)[1]["start"] == 15.1
    # A block at a leading marker starts at the beginning of the recording.
    late = items_of([{"start": 20.0, "end": 21.0, "text": "Хей"}, {"start": 22.0, "end": 23.0, "text": "Ти"}], [(1.0, 2.0)])
    result = det.build_blocks([{"first_line": 1, "kind": "program", "label": ""},
                                 {"first_line": 2, "kind": "break", "label": ""}], late, 23.0, [(1.0, 2.0)])
    assert [(s["start"], s["end"]) for s in result] == [(0.0, 1.5), (1.5, 23.0)]


# ----- the Claude requests -----------------------------------------------------
def test_detect_sends_numbered_cues_and_settles_the_real_cost(tmp_path):
    def handler(request):
        assert request["model"] == det.MODEL and request["thinking"] == {"type": "adaptive"}
        assert request["output_config"]["format"]["schema"] is det.SCHEMA
        lines = request["messages"][0]["content"].splitlines()
        assert lines[-1].startswith("12 [0:33] Реплика 12")
        assert "[" not in lines[-1].split("] ", 1)[1]  # no marker lines: every gap is 1 s of speech pause
        return breaks_response([{"first_line": 1, "kind": "break", "label": "Lidl"},
                                  {"first_line": 7, "kind": "program", "label": "Шоу"}],
                                 input_tokens=3000, output_tokens=400)
    detector, client = make(tmp_path, handler)
    result = detect(detector, cues(12), duration=40.0)
    assert [(s["kind"], s["start"]) for s in result] == [("break", 0.0), ("program", 17.5)]  # cue 6 ends 17, cue 7 starts 18
    status = detector.ledger.status()
    # 3000 * $0.10/MTok + 400 * $0.50/MTok = $0.0005, rounded up to the microdollar
    assert status["reserved_usd"] == 0 and status["charged_usd"] == pytest.approx(0.0005, abs=1e-6)
    assert len(client.calls) == 1 and client.calls[0]["service_tier"] == "standard_only"
    spending = detector.costs.summary()
    assert spending["requests"] == 1 and spending["total_usd"] == 0.0005
    assert spending["purposes"]["breaks"]["usd"] == 0.0005 and spending["models"][det.MODEL]["requests"] == 1


def test_long_transcripts_are_split_into_chunks_and_joined(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "CHUNK_CUES", 5)
    answers = {1: [{"first_line": 1, "kind": "break", "label": "Lidl"},
                   {"first_line": 4, "kind": "program", "label": "Шоу"}],
               6: [{"first_line": 6, "kind": "program", "label": "Шоу"}],   # continues: merged
               11: [{"first_line": 11, "kind": "break", "label": "Promo"}]}

    def handler(request):
        first = int(request["messages"][0]["content"].splitlines()[-1].split()[0]) - 4  # last line is cue first+4
        return breaks_response(answers[max(first, 1)])
    detector, client = make(tmp_path, handler)
    result = detect(detector, cues(15), duration=45.0)
    assert len(client.calls) == 3
    assert [(s["kind"], s["label"]) for s in result] == [("break", "Lidl"), ("program", "Шоу"), ("break", "Promo")]
    assert result[1]["start"] == pytest.approx(8.5) and result[2]["start"] == pytest.approx(29.5)


def test_bad_answers_are_retried_then_reported(tmp_path):
    attempts = []

    def flaky(request):
        attempts.append(1)
        if len(attempts) == 1:
            return api_error(anthropic.RateLimitError, 429)
        if len(attempts) == 2:
            return claude_json({"blocks": "garbage"}, model=det.MODEL)
        return breaks_response([{"first_line": 1, "kind": "program", "label": "Шоу"}])
    detector, _ = make(tmp_path, flaky)
    assert detect(detector, cues(3), duration=9.0)[0]["kind"] == "program"
    assert len(attempts) == 3

    detector, _ = make(tmp_path, lambda request: claude_json({"nope": 1}, model=det.MODEL))
    with pytest.raises(BreakDetectionError, match="incomplete"):
        detect(detector, cues(3))
    assert detector.ledger.status()["reserved_usd"] == 0  # every attempt was settled


@pytest.mark.parametrize("make_error, message", [
    (lambda: api_error(anthropic.AuthenticationError, 401), "ANTHROPIC_API_KEY"),
    (lambda: api_error(anthropic.BadRequestError, 400), "HTTP 400"),
    (lambda: breaks_response([{"first_line": 1, "kind": "program", "label": ""}], model="claude-opus-5-5"),
     "unexpected model"),
    (lambda: breaks_response([], stop_reason="refusal"), "declined"),
])
def test_failures_become_break_detection_errors(tmp_path, make_error, message):
    detector, _ = make(tmp_path, lambda request: make_error())
    with pytest.raises(BreakDetectionError, match=message):
        detect(detector, cues(3))


def test_spending_cap_stops_the_request_before_it_is_sent(tmp_path):
    detector, client = make(tmp_path, lambda request: breaks_response([]), budget=0.001)
    with pytest.raises(BreakDetectionError, match="spending limit"):
        detect(detector, cues(50))
    assert client.calls == []


def test_unconfigured_or_empty_transcripts(tmp_path):
    settings = Settings(data_dir=tmp_path)
    unconfigured = BreakDetector(settings, client=None)
    assert not unconfigured.available
    with pytest.raises(BreakDetectionError, match="not configured"):
        detect(unconfigured, cues(2))
    detector, client = make(tmp_path, lambda request: breaks_response([]))
    assert detect(detector, []) == [] and client.calls == []
