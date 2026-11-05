"""Persistent, conservative accounting for this installation's Claude requests.

Amounts are integer microdollars. Reservations survive process crashes, and an
unconfirmed request keeps its full reservation. There is no automatic reset.
The file lock also serialises separate processes or containers sharing a volume.
"""
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
from threading import RLock
import uuid


MAX_CAP = 20_000_000
_locks: dict[str, RLock] = {}
_locks_guard = RLock()


class BudgetError(ValueError):
    pass


class BudgetLedger:
    def __init__(self, path: Path, cap_usd=20):
        self.path = Path(path)
        try:
            amount = Decimal(str(cap_usd)) * 1_000_000
            if not amount.is_finite() or amount <= 0 or amount > MAX_CAP:
                raise ValueError
            self.cap = int(amount)
        except (ValueError, InvalidOperation, OverflowError):
            raise BudgetError("The translation limit must be between $0.000001 and $20.") from None
        if self.cap < 1:
            raise BudgetError("The translation limit is too small.")
        with _locks_guard:
            self.lock = _locks.setdefault(str(self.path.resolve()), RLock())

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            with self.path.with_suffix(".lock").open("a+b") as handle:
                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self):
        if not self.path.exists():
            return {"version": 1, "cap": self.cap, "requests": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (data["version"] != 1 or type(data["cap"]) is not int
                    or not 0 < data["cap"] <= MAX_CAP or not isinstance(data["requests"], dict)):
                raise ValueError
            for entry in data["requests"].values():
                if (entry["state"] not in {"reserved", "settled", "uncertain"}
                        or type(entry["amount"]) is not int or entry["amount"] < 0):
                    raise ValueError
            data["cap"] = min(data["cap"], self.cap)
            return data
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            raise BudgetError("Translation is stopped because its saved spending ledger cannot be read. Restore the ledger from a backup; do not reset it.") from None

    def _write(self, data):
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def reserve(self, amount: int, *, job_id: str, revision: int = 0) -> str:
        if type(amount) is not int or amount <= 0:
            raise BudgetError("Invalid translation cost reservation.")
        with self._locked():
            data = self._read()
            self._write(data)
            used = sum(e["amount"] for e in data["requests"].values())
            if used + amount > data["cap"]:
                raise BudgetError("The translation spending limit is reached (TRANSLATION_BUDGET_USD, at most $20): the next Claude request would exceed it, so it was not sent.")
            token = uuid.uuid4().hex
            data["requests"][token] = {"state": "reserved", "amount": amount,
                                       "job_id": job_id, "revision": revision}
            self._write(data)
            return token

    def settle(self, token: str, amount: int | None):
        with self._locked():
            data = self._read()
            entry = data["requests"][token]
            if entry["state"] != "reserved":
                raise BudgetError("This translation request has already been accounted for.")
            if amount is None:
                entry["state"] = "uncertain"
            else:
                if type(amount) is not int or amount < 0:
                    raise BudgetError("Invalid translation usage.")
                # If a provider ever exceeds the reserved token ceiling, stop
                # all subsequent requests rather than silently undercounting.
                if amount > entry["amount"]:
                    entry.update(state="uncertain", amount=data["cap"])
                    self._write(data)
                    raise BudgetError("Claude reported usage above the reserved maximum. Translation is stopped; check the provider's usage report.")
                entry.update(state="settled", amount=amount)
            self._write(data)

    def status(self) -> dict:
        with self._locked():
            data = self._read()
            # Save even the initial/empty ledger, and permanently retain a
            # lowered cap. Increasing an environment value cannot reset spend.
            self._write(data)
            reserved = sum(e["amount"] for e in data["requests"].values() if e["state"] == "reserved")
            charged = sum(e["amount"] for e in data["requests"].values() if e["state"] != "reserved")
            return {"cap_usd": data["cap"] / 1_000_000,
                    "charged_usd": charged / 1_000_000,
                    "reserved_usd": reserved / 1_000_000,
                    "remaining_usd": max(0, data["cap"] - charged - reserved) / 1_000_000,
                    "currency": "USD", "euro_budget": 28.0}
