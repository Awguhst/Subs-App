import asyncio
import json

import pytest

from app.budget import BudgetLedger
from app.config import Settings
from app.translation import BATCH_CUES, MODEL, SCHEMA, Translator, TranslationError
from fakes import FakeAnthropic, anthropic, api_error, claude_response, cues_of, echo


def make(tmp_path, handler=echo, budget=20.0, **settings):
    client = FakeAnthropic(handler)
    config = Settings(anthropic_api_key="test-key", data_dir=tmp_path, translation_budget_usd=budget, **settings)
    return Translator(config, client=client), client


def run(translator, texts, **kwargs):
    return asyncio.run(translator.translate(texts, label="test", **kwargs))


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda seconds: real_sleep(0))


def test_batches_run_in_order_with_context_and_exact_accounting(tmp_path):
    translator, client = make(tmp_path)
    texts = [f"Реплика номер {i}\nвтори ред" for i in range(95)]
    progress = []
    english = run(translator, texts, on_progress=lambda done, total: progress.append((done, total)))
    assert english == [f"EN Реплика номер {i} втори ред" for i in range(95)]
    assert len(client.calls) == 3 and progress[-1] == (3, 3)
    request = client.calls[0]
    assert request["model"] == MODEL and "temperature" not in request
    assert request["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    later = json.loads(client.calls[1]["messages"][0]["content"])
    assert later["context_before"][-1] == f"Реплика номер {BATCH_CUES - 1} втори ред"
    assert all("-->" not in call["messages"][0]["content"] for call in client.calls)
    status = translator.budget_status()
    assert status["charged_usd"] == pytest.approx(3 * (500 + 5 * 100) / 1e6)
    assert status["reserved_usd"] == 0


def test_only_missing_lines_are_requested_again(tmp_path):
    def handler(request):
        cues = cues_of(request)
        keep = cues[1:] if len(cues) > 1 else cues  # first answer omits a line
        return claude_response([{"id": c["id"], "text": f"EN {c['text']}"} for c in keep])
    translator, client = make(tmp_path, handler)
    assert run(translator, ["едно", "две", "три"]) == ["EN едно", "EN две", "EN три"]
    assert [len(cues_of(call)) for call in client.calls] == [3, 1]


def test_overload_is_retried_and_not_charged(tmp_path):
    answers = [api_error(anthropic.OverloadedError, 529), api_error(anthropic.RateLimitError, 429)]
    translator, client = make(tmp_path, lambda request: answers.pop(0) if answers else echo(request))
    assert run(translator, ["Здравей"]) == ["EN Здравей"]
    assert len(client.calls) == 3
    assert translator.budget_status()["charged_usd"] == pytest.approx(0.001)


def test_server_errors_keep_their_maximum_cost_counted(tmp_path):
    translator, _ = make(tmp_path, lambda request: api_error(anthropic.InternalServerError, 500))
    with pytest.raises(TranslationError, match="server error"):
        run(translator, ["Здравей"])
    status = translator.budget_status()
    assert status["reserved_usd"] == 0 and status["charged_usd"] > 0.05  # three uncertain attempts


def test_bad_key_fails_fast_without_retry(tmp_path):
    translator, client = make(tmp_path, lambda request: api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(TranslationError, match="ANTHROPIC_API_KEY"):
        run(translator, ["Здравей"])
    assert len(client.calls) == 1


def test_failure_cancels_other_batches_and_accounts_for_them(tmp_path):
    def handler(request):
        if cues_of(request)[0]["id"] == 0:
            return api_error(anthropic.PermissionDeniedError, 403)
        return echo(request)
    translator, _ = make(tmp_path, handler)
    with pytest.raises(TranslationError):
        run(translator, [f"ред {i}" for i in range(BATCH_CUES * 4)])
    assert translator.budget_status()["reserved_usd"] == 0  # nothing left dangling


def test_budget_limit_stops_before_any_request(tmp_path):
    translator, client = make(tmp_path, budget=0.01)
    with pytest.raises(TranslationError, match="spending limit"):
        run(translator, ["Здравей"])
    assert client.calls == []


def test_unexpected_model_and_refusals_are_errors(tmp_path):
    translator, _ = make(tmp_path, lambda r: claude_response([], model="claude-opus-5-5"))
    with pytest.raises(TranslationError, match="unexpected model"):
        run(translator, ["Здравей"])
    translator, _ = make(tmp_path / "b", lambda r: claude_response([], stop_reason="refusal"))
    with pytest.raises(TranslationError, match="declined"):
        run(translator, ["Здравей"])


def test_markup_and_control_characters_are_cleaned(tmp_path):
    translator, _ = make(tmp_path, lambda r: claude_response([{"id": 0, "text": " Hello\n\tthere\x07 "}]))
    assert run(translator, ["Здравей"]) == ["Hello there"]


def test_unconfigured_translation_is_reported(tmp_path):
    translator = Translator(Settings(data_dir=tmp_path))
    assert not translator.available
    with pytest.raises(TranslationError, match="ANTHROPIC_API_KEY"):
        run(translator, ["Здравей"])


def test_legacy_ledger_location_is_kept(tmp_path):
    legacy = tmp_path / "jobs" / ".translation-budget.json"
    BudgetLedger(legacy, cap_usd=20).reserve(1_000_000, job_id="old")
    translator, _ = make(tmp_path)
    assert translator.ledger.path == legacy
    assert translator.budget_status()["reserved_usd"] == 1.0
