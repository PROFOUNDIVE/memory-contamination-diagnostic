from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    __package__ = "tests"

from .phase13_budget_fixtures import budget_api, fixture, operation, proof, save, signed


@pytest.mark.parametrize("hold", [0, 2_000_000], ids=["settled", "ambiguous"])
@pytest.mark.parametrize("identity", ["exact", "request", "attempt"])
def test_imported_identity_refuses_in_fresh_process(tmp_path: Path, hold: int, identity: str) -> None:
    api = budget_api()
    binding = fixture(tmp_path, hold=hold)
    request = operation()
    history_path = tmp_path / "reconciliation.json"
    history = json.loads(history_path.read_bytes())
    historical = history["operations"][0]
    if identity in {"exact", "request"}:
        historical.update(run_id=request["run_id"], request_id=request["request_id"])
    if identity in {"exact", "attempt"}:
        historical["attempt_id"] = request["attempt_id"]
    raw = json.loads(binding.read_bytes())
    raw["reconciliation"] = save(history_path, history)
    save(binding, raw)
    api.SharedProviderBudget.create(binding)

    child = subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline", __file__],
        env={**os.environ, "HISTORY_BINDING": str(binding), "HISTORY_RESERVE": json.dumps(request)},
        capture_output=True, text=True, timeout=30)

    assert child.returncode == 3, child.stdout + child.stderr
    assert "BUDGET_RESERVATION_REPLAY" in child.stdout
    status = api.SharedProviderBudget.open(binding).status()
    assert (status.reservations, status.known_usd_micros, status.held_usd_micros) == (0, 5_000_000, hold)


@pytest.mark.parametrize("mutation", ["delete_reserved", "delete_intent", "rewind_intent", "delete_released"])
def test_projection_reset_refuses_on_fresh_reopen(tmp_path: Path, mutation: str) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    intent = budget.reserve(api.BudgetOperation(**operation()))
    if mutation != "delete_reserved":
        budget.mark_intent(intent)
    if mutation == "delete_released":
        budget.reconcile(api.FilePin(**proof(tmp_path, budget, intent, kind="no_charge", amount=0)))
    with sqlite3.connect(tmp_path / "aggregate.sqlite3") as connection:
        if mutation == "rewind_intent":
            connection.execute("UPDATE reservations SET state='reserved'")
        else:
            connection.execute("DELETE FROM reservations")

    child = subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline", __file__],
        env={**os.environ, "HISTORY_BINDING": str(binding)}, capture_output=True, text=True, timeout=30)

    assert child.returncode == 3, child.stdout + child.stderr
    assert "BUDGET_HISTORY" in child.stdout
    with pytest.raises(ValueError, match="BUDGET_HISTORY"):
        budget.mark_intent(intent)


@pytest.mark.parametrize("statement", ["UPDATE history SET snapshot='[]'", "DELETE FROM history"])
def test_history_is_append_only(tmp_path: Path, statement: str) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    budget.reserve(api.BudgetOperation(**operation()))

    with sqlite3.connect(tmp_path / "aggregate.sqlite3") as connection:
        with pytest.raises(sqlite3.IntegrityError, match="BUDGET_HISTORY_APPEND_ONLY"):
            connection.execute(statement)

    assert api.SharedProviderBudget.open(binding).status().held_usd_micros == 27_000_000


@pytest.mark.parametrize("mutation", ["snapshot", "tail", "sequence"])
def test_history_corruption_refuses_even_without_triggers(tmp_path: Path, mutation: str) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    intent = budget.reserve(api.BudgetOperation(**operation()))
    budget.mark_intent(intent)
    statements = {"snapshot": "UPDATE history SET snapshot='[]' WHERE sequence=1",
                  "tail": "DELETE FROM history WHERE sequence=1",
                  "sequence": "UPDATE history SET sequence=2 WHERE sequence=1"}
    with sqlite3.connect(tmp_path / "aggregate.sqlite3") as connection:
        connection.execute("DROP TRIGGER history_no_update")
        connection.execute("DROP TRIGGER history_no_delete")
        connection.execute(statements[mutation])

    with pytest.raises(ValueError, match="BUDGET_HISTORY"):
        api.SharedProviderBudget.open(binding)


@pytest.mark.parametrize("stage", ["reserve", "intent", "reconcile"])
def test_history_and_projection_commit_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    intent = None if stage == "reserve" else budget.reserve(api.BudgetOperation(**operation()))
    before = budget.status()
    append = api.append_history

    def fail_after_append(connection: sqlite3.Connection, reservation_id: str, binding_sha256: str) -> None:
        append(connection, reservation_id, binding_sha256)
        raise api.BudgetError("INJECTED_TRANSACTION_FAILURE")

    monkeypatch.setattr(api, "append_history", fail_after_append)
    with pytest.raises(ValueError, match="INJECTED_TRANSACTION_FAILURE"):
        if stage == "reserve":
            budget.reserve(api.BudgetOperation(**operation()))
        elif stage == "intent":
            budget.mark_intent(intent)
        else:
            budget.reconcile(api.FilePin(**proof(tmp_path, budget, intent, kind="no_request", amount=0)))

    assert api.SharedProviderBudget.open(binding).status() == before
    with sqlite3.connect(tmp_path / "aggregate.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM history").fetchone()[0] == (0 if stage == "reserve" else 1)
        assert connection.execute("SELECT COUNT(*) FROM proofs").fetchone()[0] == 0


def test_ambiguous_to_paid_reopen_authenticates_entire_history(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    intent = budget.reserve(api.BudgetOperation(**operation()))
    budget.mark_intent(intent)
    first = proof(tmp_path, budget, intent, kind="ambiguous", amount=3_000_000)
    budget.reconcile(api.FilePin(**first))
    raw = json.loads(Path(first["path"]).read_bytes())
    raw.pop("signature")
    raw.update(kind="charge", realized_usd_micros=4_000_000)
    second = save(tmp_path / "settled.json", signed(raw))

    api.SharedProviderBudget.open(binding).reconcile(api.FilePin(**second))

    status = api.SharedProviderBudget.open(binding).status()
    assert (status.known_usd_micros, status.held_usd_micros) == (9_000_000, 2_000_000)
    Path(first["path"]).unlink()
    with pytest.raises(OSError):
        api.SharedProviderBudget.open(binding)


def test_rehashed_intent_rewind_still_violates_transition_history(tmp_path: Path) -> None:
    api = budget_api()
    binding = fixture(tmp_path)
    budget = api.SharedProviderBudget.create(binding)
    intent = budget.reserve(api.BudgetOperation(**operation()))
    budget.mark_intent(intent)
    with sqlite3.connect(tmp_path / "aggregate.sqlite3") as connection:
        sequence, snapshot, previous = connection.execute(
            "SELECT sequence,snapshot,previous_hash FROM history WHERE sequence=1").fetchone()
        row = json.loads(snapshot)
        row[5] = "reserved"
        snapshot = json.dumps(row, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(json.dumps([sequence, snapshot, previous],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        connection.execute("DROP TRIGGER history_no_update")
        connection.execute("UPDATE history SET snapshot=?,event_hash=? WHERE sequence=1", (snapshot, digest))
        connection.execute("UPDATE reservations SET state='reserved'")

    with pytest.raises(ValueError, match="BUDGET_HISTORY_TRANSITION_INVALID"):
        api.SharedProviderBudget.open(binding)


if __name__ == "__main__":
    api = budget_api()
    try:
        budget = api.SharedProviderBudget.open(Path(os.environ["HISTORY_BINDING"]))
        request = os.environ.get("HISTORY_RESERVE")
        if request is not None:
            budget.reserve(api.BudgetOperation.model_validate_json(request))
        print(budget.status().model_dump_json())
    except ValueError as error:
        print(str(error))
        raise SystemExit(3)
