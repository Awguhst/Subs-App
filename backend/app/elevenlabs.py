"""Async client for ElevenLabs Scribe v2 speech-to-text.

Reference: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from .config import Settings

ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
MODEL = "scribe_v2"

STATUS_MESSAGES = {
    401: "ElevenLabs rejected ELEVENLABS_API_KEY. Use the secret key (starting with sk_) with Speech-to-Text access.",
    402: "ElevenLabs has insufficient credits. Check the account balance.",
    403: "ElevenLabs denied access. Check the key's Speech-to-Text permission.",
    413: "ElevenLabs rejected the audio size. Use a shorter recording.",
    422: "ElevenLabs could not transcribe this audio.",
    429: "ElevenLabs rate or concurrency limit reached. Retry shortly.",
}


class TranscriptionError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class Transcriber:
    """Callable: ``await transcriber(audio_path) -> Scribe JSON with word timestamps``."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.elevenlabs_timeout_seconds, connect=15.0, write=300.0),
            follow_redirects=False,  # never forward the key to another host
        )

    async def aclose(self):
        await self.client.aclose()

    async def __call__(self, audio_path: Path) -> dict:
        if not self.settings.transcription_configured:
            raise TranscriptionError("Transcription is not configured: set ELEVENLABS_API_KEY on the server.", 503)
        fields = {"model_id": MODEL, "language_code": "bul", "timestamps_granularity": "word",
                  "diarize": "true", "tag_audio_events": "false"}
        response = None
        for attempt in range(2):
            try:
                with Path(audio_path).open("rb") as audio:
                    response = await self.client.post(
                        ENDPOINT, headers={"xi-api-key": self.settings.elevenlabs_api_key},
                        data=fields, files={"file": ("audio.mp3", audio, "audio/mpeg")})
                break
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Nothing was sent, so one retry cannot double-bill.
                if attempt:
                    raise TranscriptionError("Could not connect to ElevenLabs. Retry shortly.") from None
                await asyncio.sleep(1)
            except httpx.TimeoutException:
                raise TranscriptionError("ElevenLabs timed out. The request may still have been billed; "
                                         "check ElevenLabs history before retrying.", 504) from None
            except httpx.HTTPError:
                raise TranscriptionError("The connection to ElevenLabs failed during transcription.") from None
        assert response is not None
        if response.status_code != 200:
            detail = {}
            try:
                detail = response.json().get("detail") or {}
            except (ValueError, AttributeError):
                pass
            if isinstance(detail, dict) and detail.get("status") == "api_key_id_used_as_api_key":
                raise TranscriptionError("ELEVENLABS_API_KEY holds an API key ID, not the secret key. "
                                         "Copy the secret shown when the key is created (starts with sk_).")
            message = STATUS_MESSAGES.get(response.status_code,
                                          f"ElevenLabs returned HTTP {response.status_code}. Retry later.")
            raise TranscriptionError(message, 502)
        try:
            result = response.json()
        except ValueError:
            raise TranscriptionError("ElevenLabs returned an unreadable response.") from None
        if not isinstance(result, dict) or not isinstance(result.get("words"), list):
            raise TranscriptionError("ElevenLabs returned no word timestamps.")
        return result
