import math

import pytest

from app.subtitles import generate_cues, parse_words, timestamp, to_vtt, validate_cues, wrap_english, wrap_text


def response(text, step=0.36):
    return {"words": [{"text": word, "start": i * step, "end": i * step + 0.3, "type": "word", "speaker_id": "speaker_0"}
                      for i, word in enumerate(text.split())]}


def test_timestamp_rounding_carries_and_long_video():
    assert timestamp(59.9996) == "00:01:00.000"
    assert timestamp(7201.2) == "02:00:01.200"
    for bad in (-1, math.inf, math.nan):
        with pytest.raises(ValueError):
            timestamp(bad)


def test_cyrillic_word_timing_and_two_lines():
    data = response("Здравейте на всички! Днес ще играем една много интересна игра заедно с нашите приятели и ще победим.")
    cues = generate_cues(data, duration=20)
    assert all(len(cue["text"].splitlines()) <= 2 for cue in cues)
    assert all(len(line) <= 42 for cue in cues for line in cue["text"].splitlines())
    assert all(cues[i]["end"] <= cues[i+1]["start"] for i in range(len(cues)-1))
    starts = {round(word["start"], 3) for word in data["words"]}
    assert all(cue["start"] in starts for cue in cues)
    vtt = to_vtt(cues)
    assert vtt.startswith("WEBVTT\n\n")
    assert "Здравейте" in vtt and "--&gt;" not in vtt


def test_events_and_spacing_are_not_dialogue():
    data = response("Здравейте приятели!")
    data["words"].insert(1, {"type": "audio_event", "text": "(gunfire)", "start": 0, "end": 0.2})
    data["words"].insert(2, {"type": "spacing", "text": " "})
    assert "gunfire" not in to_vtt(generate_cues(data))
    assert len(parse_words(data)) == 2


def test_pauses_and_speaker_boundaries():
    data = {"words": [
        {"type": "word", "text": "Здравей!", "start": 1.2, "end": 1.7, "speaker_id": "a"},
        {"type": "word", "text": "Привет!", "start": 2.0, "end": 2.5, "speaker_id": "b"},
        {"type": "word", "text": "Хайде!", "start": 4.0, "end": 4.4, "speaker_id": "b"}]}
    cues = generate_cues(data, 5)
    assert [cue["start"] for cue in cues] == [1.2, 2.0, 4.0]
    assert cues[0]["end"] <= 2
    assert cues[-1]["end"] <= 5


def test_long_handle_and_invalid_text():
    assert len(wrap_text("x" * 70).splitlines()) == 2
    with pytest.raises(ValueError):
        wrap_text("x" * 85)
    with pytest.raises(ValueError):
        validate_cues([{"start": 0, "end": 1, "text": "  "}], 2)


def test_unusual_transcripts_never_fail_a_paid_job():
    def w(text, start, end):
        return {"type": "word", "text": text, "start": start, "end": end, "speaker_id": "s"}
    # A spoken URL (one 100-character token) wraps onto extra lines instead of failing.
    url = "https://example.com/" + "a" * 80
    cues = generate_cues({"words": [w("Посетете", 0, 0.4), w(url, 0.5, 2.0)]}, 5.0)
    assert "".join(cues[-1]["text"].split("\n")) == url
    assert all(len(line) <= 42 for cue in cues for line in cue["text"].splitlines())
    # A run of zero-length words is kept, however long the joined text becomes.
    words = [w("Добре", 0, 0.4)] + [w(f"дума{i}", 1 + i * 0.1, 1 + i * 0.1) for i in range(40)] + [w("край", 5.5, 5.9)]
    text = " ".join(cue["text"].replace("\n", " ") for cue in generate_cues({"words": words}, 7.0))
    assert "дума39" in text and text.endswith("край")
    # Decoded audio can be a few ms longer than declared: a word in that tail still gets a cue.
    cues = generate_cues({"words": [w("Здравейте", 0, 0.5), w("чао", 2.995, 3.05)]}, 2.99)
    assert cues[-1]["text"] == "чао" and cues[-1]["end"] > cues[-1]["start"]


def test_simultaneous_speakers_share_cue_without_overlap():
    data = {"words": [
        {"type": "word", "text": "Хайде!", "start": 1, "end": 1.7, "speaker_id": "a"},
        {"type": "word", "text": "Давай!", "start": 1, "end": 1.5, "speaker_id": "b"},
        {"type": "word", "text": "Напред!", "start": 1.4, "end": 2, "speaker_id": "a"}]}
    cues = generate_cues(data, 3)
    assert "Хайде! Давай!" == cues[0]["text"]
    assert "speaker_id" not in cues[0]
    assert cues[0]["end"] == cues[1]["start"] == 1.4


def test_zero_duration_speech_uses_neighboring_actual_word_boundaries():
    data = {"words": [
        {"type": "word", "text": "Аз", "start": 1, "end": 1},
        {"type": "word", "text": "ще", "start": 1.1, "end": 1.3},
        {"type": "word", "text": "победя", "start": 1.4, "end": 1.4},
        {"type": "word", "text": "сега", "start": 2, "end": 2.4},
        {"type": "word", "text": "!", "start": 2.4, "end": 2.4}]}
    words = parse_words(data)
    assert words[0]["start"] == 1 and words[0]["end"] == 1.4
    assert words[-1]["text"] == "сега!"
    cues = generate_cues(data, 3)
    assert "Аз ще победя сега!" == " ".join(cue["text"].replace("\n", " ") for cue in cues)


@pytest.mark.parametrize("data", [{}, {"words": None}, {"words": [{"type": "word", "text": "Здрасти", "start": None, "end": 2}]}, {"words": [{"type": "word", "text": "Здрасти", "start": 2, "end": 1}]}])
def test_invalid_api_response(data):
    with pytest.raises(ValueError):
        generate_cues(data)


def test_validate_overlap_and_out_of_bounds():
    with pytest.raises(ValueError):
        validate_cues([{"start": 0, "end": 2, "text": "Едно"}, {"start": 1, "end": 3, "text": "Две"}], 4)
    with pytest.raises(ValueError):
        validate_cues([{"start": 0, "end": 5, "text": "Едно"}], 4)


def test_output_escapes_user_markup():
    cues = [{"start": 0, "end": 2, "text": r"<b>Едно</b> {\p1} &"}]
    assert r"&lt;b&gt;Едно&lt;/b&gt; {\p1} &amp;" in to_vtt(cues)


def test_vtt_numbers_cues_and_rejects_overlap():
    vtt = to_vtt([{"start": 0, "end": 1.5, "text": "Едно"}, {"start": 1.5, "end": 3, "text": "Две\nТри"}])
    assert vtt == ("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.500\nЕдно\n\n"
                   "2\n00:00:01.500 --> 00:00:03.000\nДве\nТри\n")
    with pytest.raises(ValueError):
        to_vtt([{"start": 0, "end": 2, "text": "a"}, {"start": 1, "end": 3, "text": "b"}])


def test_silence_produces_an_empty_track_instead_of_an_error():
    assert generate_cues({"words": []}) == []
    assert generate_cues({"words": [{"type": "spacing", "text": " ", "start": 0, "end": 1}]}) == []
    assert to_vtt([]) == "WEBVTT\n"


def test_english_wrapping_prefers_two_lines_but_never_cuts_text():
    assert wrap_english("  Hello   there ") == "Hello there"
    two = wrap_english("This translation is a little too long, so it wraps in two.")
    assert len(two.splitlines()) == 2 and all(len(line) <= 42 for line in two.splitlines())
    long = "A much longer English sentence that cannot possibly fit on two subtitle lines of forty-two characters."
    wrapped = wrap_english(long)
    assert len(wrapped.splitlines()) == 3 and all(len(line) <= 42 for line in wrapped.splitlines())
    assert wrapped.replace("\n", " ") == long


