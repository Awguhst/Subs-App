"""Offline stand-ins for ElevenLabs and Anthropic: tests never spend credits."""
import asyncio
import json
from types import SimpleNamespace

import anthropic
import httpx2

from app.translation import MODEL

BULGARIAN = "Здравейте на всички! Днес ще играем една много интересна игра. Хайде да започваме."


def scribe(text: str = BULGARIAN, step: float = 0.2) -> dict:
    return {"language_code": "bul", "words": [
        {"type": "word", "text": word, "start": round(i * step, 3), "end": round(i * step + 0.15, 3),
         "speaker_id": "speaker_0"} for i, word in enumerate(text.split())]}


class FakeTranscriber:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result if result is not None else scribe(), error, []

    async def __call__(self, audio_path):
        self.calls.append(audio_path)
        assert audio_path.stat().st_size > 0
        if self.error:
            raise self.error
        return self.result


def claude_response(items, *, model=MODEL, stop_reason="end_turn", input_tokens=500, output_tokens=100):
    return SimpleNamespace(
        model=model, stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_creation_input_tokens=None, cache_read_input_tokens=None),
        content=[SimpleNamespace(type="text", text=json.dumps({"translations": items}, ensure_ascii=False))])


def cues_of(request: dict) -> list[dict]:
    return json.loads(request["messages"][0]["content"])["cues"]


def echo(request: dict):
    return claude_response([{"id": c["id"], "text": f"EN {c['text']}"} for c in cues_of(request)])


def api_error(cls, status: int):
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    return cls("simulated", response=response, body=None)


class FakeAnthropic:
    """Mimics ``AsyncAnthropic().messages.create``; ``handler(request)`` returns or raises.

    ``latency`` makes each call wait like a real network request, so concurrency
    and cancellation behave as they do in production.
    """

    def __init__(self, handler=echo, latency: float = 0.0):
        self.handler, self.latency, self.calls = handler, latency, []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **request):
        self.calls.append(request)
        if self.latency:
            await asyncio.sleep(self.latency)
        result = self.handler(request)
        if isinstance(result, Exception):
            raise result
        return result

    async def close(self):
        pass


__all__ = ["BULGARIAN", "FakeAnthropic", "FakeTranscriber", "anthropic", "api_error", "claude_response",
           "cues_of", "echo", "scribe"]
