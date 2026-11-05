"""The spending cap is enforced before any paid provider request can begin."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.budget import BudgetError, BudgetLedger


def test_reservations_count_against_cap_before_settlement(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", cap_usd=1)
    token = ledger.reserve(750_000, job_id="job", revision=1)
    assert ledger.status()["reserved_usd"] == 0.75
    with pytest.raises(BudgetError, match="exceed"):
        ledger.reserve(250_001, job_id="next", revision=1)
    ledger.settle(token, 250_000)
    status = ledger.status()
    assert status["charged_usd"] == 0.25
    assert status["reserved_usd"] == 0
    assert status["remaining_usd"] == 0.75
    with pytest.raises(BudgetError, match="already"):
        ledger.settle(token, 0)


def test_uncertain_and_unfinished_requests_survive_restart(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path, cap_usd=1)
    unknown = ledger.reserve(600_000, job_id="job", revision=1)
    ledger.settle(unknown, None)
    ledger.reserve(400_000, job_id="unfinished", revision=1)
    restarted = BudgetLedger(path, cap_usd=20)
    status = restarted.status()
    assert status["cap_usd"] == 1
    assert status["charged_usd"] == 0.6
    assert status["reserved_usd"] == 0.4
    assert status["remaining_usd"] == 0
    with pytest.raises(BudgetError, match="exceed"):
        restarted.reserve(1, job_id="retry", revision=1)


def test_concurrent_ledgers_cannot_overbook_cap(tmp_path):
    path = tmp_path / "budget.json"

    def reserve(index):
        try:
            return BudgetLedger(path, cap_usd=1).reserve(200_000, job_id=str(index), revision=1)
        except BudgetError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        tokens = list(executor.map(reserve, range(16)))
    assert len([token for token in tokens if token is not None]) == 5
    assert BudgetLedger(path, cap_usd=1).status()["remaining_usd"] == 0


def test_lowered_cap_is_retained_and_cannot_be_raised(tmp_path):
    path = tmp_path / "budget.json"
    BudgetLedger(path, cap_usd=20).status()
    assert BudgetLedger(path, cap_usd=0.25).status()["cap_usd"] == 0.25
    assert BudgetLedger(path, cap_usd=20).status()["cap_usd"] == 0.25


def test_rejected_reservation_also_retains_a_lowered_cap(tmp_path):
    path = tmp_path / "budget.json"
    ledger = BudgetLedger(path, cap_usd=20)
    ledger.reserve(500_000, job_id="old", revision=1)
    with pytest.raises(BudgetError):
        BudgetLedger(path, cap_usd=0.25).reserve(1, job_id="new", revision=1)
    assert BudgetLedger(path, cap_usd=20).status()["cap_usd"] == 0.25


@pytest.mark.parametrize("bad", [
    "not json",
    {"version": 2, "cap": 1_000_000, "requests": {}},
    {"version": 1, "cap": 0, "requests": {}},
    {"version": 1, "cap": 21_000_000, "requests": {}},
    {"version": 1, "cap": 1_000_000, "requests": []},
    {"version": 1, "cap": 1_000_000, "requests": {"x": {"state": "settled", "amount": -10}}},
    {"version": 1, "cap": 1_000_000, "requests": {"x": {"state": "settled", "amount": True}}},
    {"version": 1, "cap": 1_000_000, "requests": {"x": {"state": "unknown", "amount": 10}}},
])
def test_corrupt_ledger_blocks_spending(tmp_path, bad):
    path = tmp_path / "budget.json"
    path.write_text(bad if isinstance(bad, str) else json.dumps(bad), encoding="utf-8")
    ledger = BudgetLedger(path)
    with pytest.raises(BudgetError, match="saved spending ledger"):
        ledger.reserve(1, job_id="job", revision=1)


def test_optimized_python_still_rejects_corrupt_ledger(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"version": 1, "cap": 1_000_000,
                               "requests": {"bad": {"state": "settled", "amount": -2_000_000}}}), encoding="utf-8")
    code = """from pathlib import Path
import sys
from app.budget import BudgetLedger, BudgetError
try:
    BudgetLedger(Path(sys.argv[1]), cap_usd=1).reserve(1_500_000, job_id='new', revision=1)
except BudgetError:
    sys.exit(0)
sys.exit('Optimized Python incorrectly accepted a corrupt ledger')
"""
    result = subprocess.run([sys.executable, "-O", "-c", code, str(path)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_provider_usage_above_reserved_maximum_blocks_further_spend(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", cap_usd=1)
    token = ledger.reserve(100_000, job_id="job", revision=1)
    with pytest.raises(BudgetError, match="above the reserved maximum"):
        ledger.settle(token, 100_001)
    assert ledger.status()["remaining_usd"] == 0
    with pytest.raises(BudgetError):
        ledger.reserve(1, job_id="next", revision=1)


@pytest.mark.parametrize("value", [0, -1, 21, float("inf"), float("nan"), "invalid"])
def test_invalid_caps_are_rejected(tmp_path, value):
    with pytest.raises(BudgetError):
        BudgetLedger(tmp_path / "budget.json", cap_usd=value)
