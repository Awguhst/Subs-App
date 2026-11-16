"""What the provider requests actually cost, per request, at the published list prices.

Every ElevenLabs and Anthropic request that returns usage is appended, with that
usage, the rates applied and the resulting amount, to an append-only JSON Lines
file in the data directory (``costs.jsonl``). Totals are computed from it, so
they survive restarts and redeploys (on Railway: with a volume attached). This
is separate from the lifetime *cap* in ``budget.py``, which deliberately
over-reserves; the amounts here are what the price lists charge for the usage
the providers reported.

Prices (checked 2026-11-16):

- ElevenLabs Scribe v2: $0.20 per hour of audio, billed in US dollars pro rata
  (https://elevenlabs.io/pricing/api). The ``audio_duration_secs`` the API
  reports is the billed length; the measured length of the extracted audio is
  the fallback. ``ELEVENLABS_STT_USD_PER_HOUR`` overrides the rate.
- Claude Haiku 4.5: $1 / $5 per million input / output tokens; cache writes at
  1.25x (5-minute) or 2x (1-hour) the input price, cache reads at 0.1x.
- Claude Haiku 5.5: $0.10 / $0.50 for prompts up to 100k tokens, $0.50 / $2.50
  above that, with the same cache multipliers. Batch requests cost half. The
  translation and break-detection requests are pinned to standard capacity
  (``service_tier="standard_only"``), so priority-tier rates never apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import logging
import math
import os
from pathlib import Path
from threading import RLock

from .budget import exclusive_lock

log = logging.getLogger("subtitles")
VERSION = 1
MTOK = Decimal(1_000_000)
NANO = Decimal("0.000000001")   # amounts are stored to the nanodollar
MICRO = Decimal("0.000001")     # and reported to the microdollar
SERVICES = ("elevenlabs", "anthropic")
PURPOSES = ("transcription", "translation", "breaks")
TOKEN_COUNTS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
DEFAULT_SCRIBE_USD_PER_HOUR = Decimal("0.20")
ZERO = {"usd": 0.0, "services": {name: 0.0 for name in SERVICES}, "requests": 0}
EMPTY_COST = {"currency": "USD", "request": ZERO, "media": ZERO}


class CostError(ValueError):
    pass


@dataclass(frozen=True)
class Media:
    """The recording a provider request was made for."""
    id: str
    filename: str | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class TokenRates:
    """USD per million tokens at standard capacity."""
    input: Decimal
    output: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal

    def as_dict(self) -> dict[str, float]:
        return {"input": float(self.input), "output": float(self.output),
                "cache_write_5m": float(self.cache_write_5m), "cache_write_1h": float(self.cache_write_1h),
                "cache_read": float(self.cache_read)}


def _rates(input_price: str, output_price: str) -> TokenRates:
    base, output = Decimal(input_price), Decimal(output_price)
    return TokenRates(base, output, base * Decimal("1.25"), base * 2, base / 10)


# Model id prefix -> ((largest prompt size the rates apply to, rates), ...), smallest tier first.
ANTHROPIC_PRICES: dict[str, tuple[tuple[int | None, TokenRates], ...]] = {
    "claude-haiku-4-5": ((None, _rates("1", "5")),),
    "claude-haiku-5-5": ((100_000, _rates("0.10", "0.50")), (None, _rates("0.50", "2.50"))),
}
TIER_MULTIPLIERS = {"standard": Decimal(1), "batch": Decimal("0.5")}


def pricing(scribe_usd_per_hour: Decimal) -> dict:
    """The rates in effect, as reported by the spending endpoints."""
    return {"currency": "USD",
            "elevenlabs": {"scribe_usd_per_audio_hour": float(scribe_usd_per_hour)},
            "anthropic_usd_per_mtok": {
                prefix: [{"prompt_tokens_up_to": limit, **rates.as_dict()} for limit, rates in tiers]
                for prefix, tiers in ANTHROPIC_PRICES.items()}}


def anthropic_cost(model: str, usage) -> tuple[Decimal | None, dict, dict]:
    """Price one Messages response from the usage it reports.

    Returns (amount in USD, or None when the model or tier is not in the price
    list; the usage as recorded; the rates applied). ``input_tokens`` is the
    uncached part of the prompt; cached tokens are reported separately.
    """
    def count(value) -> int:
        return value if type(value) is int and value > 0 else 0

    counts = {name: count(getattr(usage, name, None)) for name in TOKEN_COUNTS}
    breakdown = getattr(usage, "cache_creation", None)
    one_hour = count(getattr(breakdown, "ephemeral_1h_input_tokens", None)) if breakdown is not None else 0
    one_hour = min(one_hour, counts["cache_creation_input_tokens"])
    five_minutes = counts["cache_creation_input_tokens"] - one_hour
    counts.update(cache_write_5m_tokens=five_minutes, cache_write_1h_tokens=one_hour)
    tier = getattr(usage, "service_tier", None) or "standard"
    counts["service_tier"] = tier
    speed = getattr(usage, "speed", None)
    if speed:
        counts["speed"] = speed
    prices = next((tiers for prefix, tiers in ANTHROPIC_PRICES.items() if str(model).startswith(prefix)), None)
    multiplier = TIER_MULTIPLIERS.get(tier)
    if prices is None or multiplier is None or (speed and speed != "standard"):
        return None, counts, {}
    prompt = counts["input_tokens"] + counts["cache_creation_input_tokens"] + counts["cache_read_input_tokens"]
    limit, rates = next((limit, rates) for limit, rates in prices if limit is None or prompt <= limit)
    amount = (counts["input_tokens"] * rates.input + counts["output_tokens"] * rates.output
              + five_minutes * rates.cache_write_5m + one_hour * rates.cache_write_1h
              + counts["cache_read_input_tokens"] * rates.cache_read) / MTOK * multiplier
    applied = {"prompt_tokens_up_to": limit, **rates.as_dict(), "service_tier_multiplier": float(multiplier)}
    return amount, counts, applied


def billed_audio_seconds(transcript: object, measured: float) -> tuple[float, str]:
    """The length ElevenLabs bills for: its own ``audio_duration_secs`` when the
    response carries a usable one, else the measured length of the audio sent."""
    reported = transcript.get("audio_duration_secs") if isinstance(transcript, dict) else None
    if type(reported) in (int, float) and math.isfinite(reported) and reported > 0:
        return float(reported), "audio_duration_secs"
    return float(measured), "measured"


def _usd(amount: Decimal) -> float:
    return float(amount.quantize(MICRO, ROUND_HALF_UP))


def _parse_line(line: str) -> dict | None:
    try:
        record = json.loads(line)
        if (record["v"] != VERSION or record["service"] not in SERVICES
                or not isinstance(record["media_id"], str) or not isinstance(record["usage"], dict)
                or not isinstance(record["purpose"], str) or not isinstance(record["model"], str)):
            return None
        at = datetime.fromisoformat(record["at"])
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        cost = record["cost_usd"]
        if cost is not None:
            cost = Decimal(str(cost))
            if not cost.is_finite() or cost < 0:
                return None
    except (ValueError, KeyError, TypeError, InvalidOperation):
        return None
    return {**record, "_at": at, "_cost": cost}


def _public(record: dict) -> dict:
    cost = record["_cost"]
    return {**{k: v for k, v in record.items() if not k.startswith("_")},
            "cost_usd": None if cost is None else float(cost)}


def _aware(moment: datetime | None) -> datetime | None:
    if moment is not None and moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


class CostTracker:
    """Append-only, cross-process record of what every provider request cost."""

    def __init__(self, path: Path, scribe_usd_per_hour=DEFAULT_SCRIBE_USD_PER_HOUR):
        self.path = Path(path)
        try:
            rate = Decimal(str(scribe_usd_per_hour))
            if not rate.is_finite() or rate < 0:
                raise ValueError
        except (ValueError, InvalidOperation):
            raise CostError("ELEVENLABS_STT_USD_PER_HOUR must be a number of at least 0.") from None
        self.scribe_usd_per_hour = rate
        self.write_failures = 0
        self.skipped_lines = 0
        self._guard = RLock()
        self._loaded: tuple[int, int] | None = None   # (size, mtime_ns) the parsed records belong to
        self._records: list[dict] = []

    @classmethod
    def from_settings(cls, settings) -> "CostTracker":
        return cls(settings.costs_path, settings.elevenlabs_usd_per_hour)

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    @property
    def pricing(self) -> dict:
        return pricing(self.scribe_usd_per_hour)

    # ----- recording -----------------------------------------------------
    def record_transcription(self, media: Media, model: str, audio_seconds: float, source: str) -> Decimal:
        seconds = Decimal(str(audio_seconds))
        if not seconds.is_finite() or seconds < 0:
            raise CostError("Invalid audio length.")
        amount = seconds * self.scribe_usd_per_hour / 3600
        self._append({"service": "elevenlabs", "purpose": "transcription", "model": model,
                      "usage": {"audio_seconds": float(seconds), "source": source},
                      "rates": {"usd_per_audio_hour": float(self.scribe_usd_per_hour)}}, media, amount)
        return amount

    def record_anthropic(self, media: Media, purpose: str, model: str, usage) -> Decimal | None:
        amount, counts, rates = anthropic_cost(model, usage)
        self._append({"service": "anthropic", "purpose": purpose, "model": str(model),
                      "usage": counts, "rates": rates}, media, amount)
        return amount

    def _append(self, record: dict, media: Media, amount: Decimal | None):
        record = {"v": VERSION, "at": self.now().isoformat(timespec="microseconds"), **record,
                  "media_id": media.id, "filename": media.filename, "duration_seconds": media.duration_seconds,
                  "cost_usd": None if amount is None else str(amount.quantize(NANO, ROUND_HALF_UP))}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        try:
            with exclusive_lock(self.path):
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
        except OSError:
            # The provider has already been paid; the result must still reach
            # the client. The summary reports how many requests went unrecorded.
            self.write_failures += 1
            log.exception("Could not record provider spending in %s", self.path)

    # ----- reading -------------------------------------------------------
    def records(self) -> list[dict]:
        with self._guard:
            try:
                stat = self.path.stat()
            except OSError:
                self._loaded, self._records = None, []
                return []
            key = (stat.st_size, stat.st_mtime_ns)
            if key != self._loaded:   # an append always changes the size
                self._records, self.skipped_lines = self._parse()
                self._loaded = key
            return self._records

    def _parse(self) -> tuple[list[dict], int]:
        with exclusive_lock(self.path):
            try:
                text = self.path.read_text(encoding="utf-8")
            except OSError:
                return [], 0
        records, skipped = [], 0
        for line in text.splitlines():
            if not line.strip():
                continue
            record = _parse_line(line)
            if record is None:
                skipped += 1
            else:
                records.append(record)
        if skipped:
            log.warning("%d unreadable line(s) in %s were ignored", skipped, self.path)
        return records, skipped

    def _select(self, since=None, until=None, media_id=None) -> list[dict]:
        since, until = _aware(since), _aware(until)
        return [r for r in self.records()
                if (media_id is None or r["media_id"] == media_id)
                and (since is None or r["_at"] >= since) and (until is None or r["_at"] <= until)]

    @staticmethod
    def _totals(records: list[dict]) -> dict:
        services = {name: Decimal(0) for name in SERVICES}
        for record in records:
            services[record["service"]] += record["_cost"] or 0
        return {"usd": _usd(sum(services.values(), Decimal(0))),
                "services": {name: _usd(amount) for name, amount in services.items()},
                "requests": len(records)}

    def cost_of(self, media_id: str, since: datetime | None = None) -> dict:
        """What one recording has cost: ``request`` covers the records since
        ``since`` (this request's own provider calls), ``media`` all of them."""
        mine = self._select(media_id=media_id)
        recent = self._select(since=since, media_id=media_id) if since is not None else []
        return {"currency": "USD", "request": self._totals(recent), "media": self._totals(mine)}

    @staticmethod
    def _media_entry(record: dict) -> dict:
        return {"id": record["media_id"], "filename": None, "duration_seconds": None,
                "usd": Decimal(0), "services": {name: Decimal(0) for name in SERVICES},
                "purposes": {name: Decimal(0) for name in PURPOSES},
                "requests": 0, "unpriced_requests": 0, "_first": record["_at"], "_last": record["_at"]}

    @staticmethod
    def _add(entry: dict, record: dict):
        amount = record["_cost"]
        entry["requests"] += 1
        if amount is None:
            entry["unpriced_requests"] += 1
            amount = Decimal(0)
        entry["usd"] += amount
        entry["services"][record["service"]] += amount
        entry["purposes"][record["purpose"]] = entry["purposes"].get(record["purpose"], Decimal(0)) + amount
        if record.get("filename"):
            entry["filename"] = record["filename"]
        if record.get("duration_seconds"):
            entry["duration_seconds"] = record["duration_seconds"]
        entry["_first"], entry["_last"] = min(entry["_first"], record["_at"]), max(entry["_last"], record["_at"])

    @staticmethod
    def _finish(entry: dict) -> dict:
        return {**{k: v for k, v in entry.items() if not k.startswith("_")},
                "usd": _usd(entry["usd"]),
                "services": {k: _usd(v) for k, v in entry["services"].items()},
                "purposes": {k: _usd(v) for k, v in entry["purposes"].items()},
                "first_at": entry["_first"].isoformat(), "last_at": entry["_last"].isoformat()}

    def media(self, media_id: str) -> dict | None:
        """One recording's spending with every provider request behind it."""
        mine = self._select(media_id=media_id)
        if not mine:
            return None
        entry = self._media_entry(mine[0])
        for record in mine:
            self._add(entry, record)
        return {**self._finish(entry), "records": [_public(r) for r in mine]}

    def summary(self, since: datetime | None = None, until: datetime | None = None,
                media_id: str | None = None, limit: int = 100) -> dict:
        """Totals, per service, purpose, model and recording."""
        selected = self._select(since, until, media_id)
        services: dict[str, dict] = {name: {"usd": Decimal(0), "requests": 0} for name in SERVICES}
        services["elevenlabs"]["audio_seconds"] = 0.0
        services["anthropic"].update({name: 0 for name in TOKEN_COUNTS})
        purposes: dict[str, dict] = {name: {"usd": Decimal(0), "requests": 0} for name in PURPOSES}
        models: dict[str, dict] = {}
        media: dict[str, dict] = {}
        total, unpriced = Decimal(0), 0
        for record in selected:
            amount = record["_cost"]
            if amount is None:
                unpriced += 1
                amount = Decimal(0)
            total += amount
            service = services[record["service"]]
            service["usd"] += amount
            service["requests"] += 1
            if record["service"] == "elevenlabs":
                service["audio_seconds"] += float(record["usage"].get("audio_seconds") or 0)
            else:
                for name in TOKEN_COUNTS:
                    service[name] += int(record["usage"].get(name) or 0)
            for bucket, key in ((purposes, record["purpose"]), (models, record["model"])):
                item = bucket.setdefault(key, {"usd": Decimal(0), "requests": 0})
                item["usd"] += amount
                item["requests"] += 1
            self._add(media.setdefault(record["media_id"], self._media_entry(record)), record)

        def money(bucket: dict) -> dict:
            return {name: {**item, "usd": _usd(item["usd"])} for name, item in bucket.items()}

        ordered = sorted(media.values(), key=lambda entry: entry["_last"], reverse=True)
        first = min((r["_at"] for r in selected), default=None)
        last = max((r["_at"] for r in selected), default=None)
        since, until = _aware(since), _aware(until)
        return {"currency": "USD", "total_usd": _usd(total), "requests": len(selected),
                "unpriced_requests": unpriced, "unrecorded_requests": self.write_failures,
                "unreadable_lines": self.skipped_lines, "media_count": len(media),
                "services": money(services), "purposes": money(purposes), "models": money(models),
                "media": [self._finish(entry) for entry in ordered[:limit]],
                "first_at": first.isoformat() if first else None,
                "last_at": last.isoformat() if last else None,
                "pricing": self.pricing,
                "filters": {"since": since.isoformat() if since else None,
                            "until": until.isoformat() if until else None,
                            "media_id": media_id, "limit": limit}}
