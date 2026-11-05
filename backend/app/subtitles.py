"""Convert Scribe word timestamps to readable, Unicode-safe subtitle cues."""

from __future__ import annotations

import html
import math
import re
import textwrap

MAX_LINE = 42
MAX_CUE_DURATION = 6.0


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _clean_text(text: str) -> str:
    # Remove control characters without discarding Cyrillic or punctuation.
    return " ".join("".join(c for c in text if c >= " " or c in "\n\t").split())


def _join_words(words: list[dict]) -> str:
    text = " ".join(word["text"] for word in words)
    text = re.sub(r"\s+([,.;:!?\)\]\}])", r"\1", text)
    return re.sub(r"([\(\[\{])\s+", r"\1", text)


def wrap_text(text: str) -> str:
    """Balance at most two lines, respecting words and phrase punctuation."""
    text = _clean_text(text)
    if not text:
        raise ValueError("Subtitle text cannot be empty.")
    if len(text) <= MAX_LINE:
        return text
    candidates = []
    for position, char in enumerate(text):
        if char != " ":
            continue
        left, right = text[:position], text[position + 1:]
        if len(left) <= MAX_LINE and len(right) <= MAX_LINE:
            score = abs(len(left) - len(right))
            if left.endswith((",", ";", ":", ".", "!", "?")):
                score -= 12
            candidates.append((score, left, right))
    if candidates:
        _, left, right = min(candidates)
        return left + "\n" + right
    # Very long game handles can exceed a line. Break a single token without
    # inventing new cue timestamps, retaining the exact original text.
    if " " not in text and len(text) <= MAX_LINE * 2:
        split = (len(text) + 1) // 2
        return text[:split] + "\n" + text[split:]
    raise ValueError("Subtitle text must fit within two lines of 42 characters. Split this cue.")


def parse_words(response: dict) -> list[dict]:
    if not isinstance(response, dict) or not isinstance(response.get("words"), list):
        raise ValueError("ElevenLabs returned no word-level timestamps.")
    words = []
    for entry in response["words"]:
        if not isinstance(entry, dict) or entry.get("type") != "word":
            continue
        text = _clean_text(str(entry.get("text", "")))
        if not text:
            continue
        start = _number(entry.get("start"), "Word start")
        end = _number(entry.get("end"), "Word end")
        if start < 0 or end < start:
            raise ValueError("ElevenLabs returned invalid word timestamps.")
        words.append({"start": start, "end": end, "text": text,
                      "speaker_id": entry.get("speaker_id")})
    if not words:
        return []
    words.sort(key=lambda word: word["start"])
    # Scribe can emit genuine tokens with equal start/end timestamps. Attach
    # those tokens to the nearest timed word, using only observed boundaries.
    # This keeps the token without manufacturing a speech duration for it.
    timed = [i for i, word in enumerate(words) if word["end"] > word["start"]]
    if not timed:
        raise ValueError("ElevenLabs returned no usable word durations. Retry transcription.")
    if len(timed) == len(words):
        return words
    parts = {i: [] for i in timed}
    cursor = 0
    for index, word in enumerate(words):
        while cursor + 1 < len(timed) and timed[cursor + 1] <= index:
            cursor += 1
        if word["end"] > word["start"]:
            anchor = index
        else:
            candidates = [timed[cursor]]
            if cursor + 1 < len(timed):
                candidates.append(timed[cursor + 1])
            anchor = min(candidates, key=lambda i: min(abs(word["start"] - words[i]["start"]),
                                                       abs(word["start"] - words[i]["end"])))
        parts[anchor].append(word)
    joined = []
    for index in timed:
        tokens = parts[index]
        joined.append({**words[index], "text": _join_words(tokens),
                       "start": min(token["start"] for token in tokens),
                       "end": max(token["end"] for token in tokens)})
    return joined


def generate_cues(response: dict, duration: float | None = None) -> list[dict]:
    """Segment real word timestamps by pauses, speakers, sentences, and length.

    Cue starts always come from a spoken word. End padding uses only available
    silence, is capped at one second, and never overlaps the following cue.
    """
    words = parse_words(response)
    if not words:
        return []
    if duration is not None:
        # Decoded audio can run a few milliseconds past the declared length
        # (encoder padding); never let that clamp a real word into a zero cue.
        duration = max(_number(duration, "Video duration"), max(word["end"] for word in words))
    groups: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        if current:
            previous = current[-1]
            gap = word["start"] - previous["end"]
            text = _join_words(current)
            elapsed = previous["end"] - current[0]["start"]
            speaker_change = (previous["speaker_id"] is not None and
                              word["speaker_id"] is not None and
                              previous["speaker_id"] != word["speaker_id"])
            split = (gap >= 0.65 or speaker_change or
                     word["end"] - current[0]["start"] > MAX_CUE_DURATION or
                     (text.endswith((".", "!", "?", "…")) and elapsed >= 1.2) or
                     (text.endswith((",", ";", ":")) and len(text) >= 55))
            try:
                wrap_text(_join_words(current + [word]))
            except ValueError:
                split = True
            # Two speakers may begin on the same timestamp. Keep their words
            # together; splitting here would create an impossible zero cue.
            if word["start"] <= current[0]["start"] + 0.001:
                split = False
            if split:
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    cues = []
    for index, group in enumerate(groups):
        start = group[0]["start"]
        spoken_end = max(word["end"] for word in group)
        next_start = groups[index + 1][0]["start"] if index + 1 < len(groups) else math.inf
        if duration is not None:
            next_start = min(next_start, _number(duration, "Video duration"))
        text = wrap_lines(_join_words(group))
        ideal_end = start + max(0.85, len(text.replace("\n", " ")) / 18)
        end = min(max(spoken_end, min(ideal_end, spoken_end + 1.0)), next_start)
        # Overlapping speakers are displayed in chronological order; impossible
        # zero-length intervals must be corrected rather than fabricated.
        if end - start < 0.001:
            raise ValueError("Overlapping word timestamps cannot form readable cues. Retry transcription.")
        cue = {"start": round(start, 3), "end": round(end, 3), "text": text}
        if group[0]["speaker_id"] is not None and len({word["speaker_id"] for word in group}) == 1:
            cue["speaker_id"] = str(group[0]["speaker_id"])
        cues.append(cue)
    return validate_cues(cues, duration)


def validate_cues(cues: list[dict], duration: float | None = None) -> list[dict]:
    if not isinstance(cues, list) or not cues:
        raise ValueError("At least one subtitle cue is required.")
    if len(cues) > 100_000:
        raise ValueError("Too many subtitle cues.")
    limit = _number(duration, "Video duration") if duration is not None else math.inf
    validated = []
    previous_end = 0.0
    for index, cue in enumerate(cues, 1):
        if not isinstance(cue, dict):
            raise ValueError(f"Cue {index} is invalid.")
        start = round(_number(cue.get("start"), f"Cue {index} start"), 3)
        end = round(_number(cue.get("end"), f"Cue {index} end"), 3)
        if start < 0 or end <= start or start < previous_end:
            raise ValueError(f"Cue {index}: times must be positive, ordered, and non-overlapping.")
        if end > limit + 0.001:
            raise ValueError(f"Cue {index} ends beyond the video duration.")
        if not isinstance(cue.get("text"), str):
            raise ValueError(f"Cue {index} text must be a string.")
        raw_lines = cue["text"].replace("\r\n", "\n").split("\n")
        if len(raw_lines) > 1 and all(_clean_text(line) and len(_clean_text(line)) <= MAX_LINE for line in raw_lines):
            text = "\n".join(_clean_text(line) for line in raw_lines)
        else:
            text = wrap_lines(cue["text"])
        if not text:
            raise ValueError(f"Cue {index} has no text.")
        clean = {"start": start, "end": end, "text": text}
        if cue.get("speaker_id") is not None:
            clean["speaker_id"] = str(cue["speaker_id"])
        validated.append(clean)
        previous_end = end
    return validated


def timestamp(seconds: float) -> str:
    total = round(_number(seconds, "Timestamp") * 1000)
    if total < 0:
        raise ValueError("Timestamp cannot be negative.")
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{millis:03}"


def to_vtt(cues: list[dict]) -> str:
    """Format cues as WebVTT. Text is escaped so it can never become markup."""
    blocks = ["WEBVTT"]
    previous_end = 0.0
    for index, cue in enumerate(cues, 1):
        start, end = _number(cue["start"], "Cue start"), _number(cue["end"], "Cue end")
        if start < previous_end or end <= start:
            raise ValueError(f"Cue {index}: times must be ordered and non-overlapping.")
        previous_end = end
        text = html.escape(cue["text"].replace("\r\n", "\n").strip(), quote=False)
        blocks.append(f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}")
    return "\n\n".join(blocks) + "\n"


def wrap_lines(text: str) -> str:
    """Two balanced lines of at most 42 characters when possible. Text that
    cannot fit (a long translation, or an unbroken token such as a spoken URL)
    wraps onto further lines rather than being cut or failing the job."""
    text = _clean_text(text)
    try:
        return wrap_text(text)
    except ValueError:
        return "\n".join(textwrap.wrap(text, MAX_LINE, break_long_words=True, break_on_hyphens=False))


wrap_english = wrap_lines
