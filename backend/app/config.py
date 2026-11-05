"""Environment-driven settings. Only the provider keys have no usable default."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDERS = {"", "your_api_key_here", "your_anthropic_secret_here", "replace_me", "changeme"}


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _int(name: str, default: int, low: int, high: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a whole number.") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


def _list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    elevenlabs_api_key: str = field(default="", repr=False)
    anthropic_api_key: str = field(default="", repr=False)
    api_keys: tuple[str, ...] = field(default=(), repr=False)
    data_dir: Path = ROOT / "data"
    max_upload_mb: int = 2048
    max_duration_minutes: int = 240
    max_concurrent_jobs: int = 2
    translation_concurrency: int = 6
    translation_budget_usd: float = 20.0
    cache_ttl_hours: int = 168
    elevenlabs_timeout_seconds: int = 1800
    anthropic_timeout_seconds: int = 120
    cors_origins: tuple[str, ...] = ("*",)
    web_ui: bool = True
    frontend_dir: Path = ROOT / "frontend"

    @classmethod
    def from_env(cls) -> "Settings":
        # A local .env wins over stale shell variables; deployments (Docker,
        # Railway) have no .env file and use their environment directly.
        load_dotenv(ROOT / ".env", override=True)
        data_dir = os.getenv("DATA_DIR") or os.getenv("RAILWAY_VOLUME_MOUNT_PATH") or "./data"
        try:
            budget = float(os.getenv("TRANSLATION_BUDGET_USD", "20") or "20")
        except ValueError:
            raise ValueError("TRANSLATION_BUDGET_USD must be a number.") from None
        if not math.isfinite(budget) or not 0 < budget <= 20:
            raise ValueError("TRANSLATION_BUDGET_USD must be above 0 and at most the app's $20 safety cap.")
        return cls(
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            api_keys=_list(os.getenv("API_KEYS", "") or os.getenv("API_KEY", "")),
            data_dir=_resolve(data_dir),
            max_upload_mb=_int("MAX_UPLOAD_MB", 2048, 1, 20_480),
            max_duration_minutes=_int("MAX_DURATION_MINUTES", 240, 1, 600),
            max_concurrent_jobs=_int("MAX_CONCURRENT_JOBS", 2, 1, 32),
            translation_concurrency=_int("TRANSLATION_CONCURRENCY", 6, 1, 32),
            translation_budget_usd=budget,
            cache_ttl_hours=_int("CACHE_TTL_HOURS", 168, 0, 24 * 365),
            elevenlabs_timeout_seconds=_int("ELEVENLABS_TIMEOUT_SECONDS", 1800, 30, 7200),
            anthropic_timeout_seconds=_int("ANTHROPIC_TIMEOUT_SECONDS", 120, 10, 600),
            cors_origins=_list(os.getenv("CORS_ORIGINS", "*")) or ("*",),
            web_ui=_bool("ENABLE_WEB_UI", True),
        )

    @property
    def transcription_configured(self) -> bool:
        return self.elevenlabs_api_key not in PLACEHOLDERS

    @property
    def translation_configured(self) -> bool:
        return self.anthropic_api_key not in PLACEHOLDERS

    @property
    def auth_required(self) -> bool:
        return bool(self.api_keys)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def ledger_path(self) -> Path:
        # Earlier versions kept the lifetime spending ledger in jobs/. Keep
        # using it when present so a redeploy can never reset recorded spend.
        legacy = self.data_dir / "jobs" / ".translation-budget.json"
        return legacy if legacy.exists() else self.data_dir / "translation-budget.json"

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self.elevenlabs_api_key, self.anthropic_api_key, *self.api_keys) if s)

    def redact(self, message: str) -> str:
        for secret in self.secrets:
            message = message.replace(secret, "[redacted]")
        return message
