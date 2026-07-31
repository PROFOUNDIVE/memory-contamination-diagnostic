from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    BudgetLedger,
    LedgerError,
    ResourceBudget,
)


def _reserve_screening_in_child(path: str, result: multiprocessing.Queue[str]) -> None:
    BudgetLedger(Path(path)).reserve_process("screening", "screen-002")
    result.put("reserved")


def test_ledger_retains_unresolved_wall_reservations_across_restart(tmp_path) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    ledger = BudgetLedger(path)
    ledger.reserve_process("screening", "screen-001")

    restarted = BudgetLedger(path)

    assert restarted.remaining_wall_seconds == 7200
    restarted.reserve_process("screening", "screen-002")
    with pytest.raises(LedgerError, match="WALL_TIME_CAP_EXCEEDED"):
        restarted.reserve_process("bct", "bct-001")


def test_ledger_durably_reserves_all_resources_and_settlement_releases_only_unused(tmp_path) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    ledger = BudgetLedger(path)

    reservation = ledger.reserve_process("screening", "screen-001")
    reserve_record = json.loads(path.read_text(encoding="utf-8"))

    assert reservation.resources == ResourceBudget(90, 368_640, 57_600, 2_000_000, 3_600)
    assert reserve_record["resources"] == {
        "calls": 90,
        "input_tokens": 368_640,
        "microusd": 2_000_000,
        "output_tokens": 57_600,
        "wall_seconds": 3_600,
    }
    ledger.settle(reservation, ResourceBudget(1, 600, 60, 10_000, 600))

    assert BudgetLedger(path).remaining_resources == ResourceBudget(
        569, 2_334_120, 364_740, 9_990_000, 10_200
    )


def test_unresolved_reservations_from_another_process_consume_shared_capacity(tmp_path) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    BudgetLedger(path).reserve_process("screening", "screen-001")
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    child = context.Process(target=_reserve_screening_in_child, args=(str(path), result))

    child.start()
    child.join(timeout=5)

    assert child.exitcode == 0
    assert result.get(timeout=1) == "reserved"
    with pytest.raises(LedgerError, match="WALL_TIME_CAP_EXCEEDED"):
        BudgetLedger(path).reserve_process("bct", "bct-001")


def test_ledger_rejects_rehashed_previous_record_tampering(tmp_path) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    BudgetLedger(path).reserve_process("screening", "screen-001")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["previous_hash"] = "f" * 64
    unsigned = {key: value for key, value in record.items() if key != "record_hash"}
    record["record_hash"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(LedgerError, match="LEDGER_CHAIN_INVALID"):
        BudgetLedger(path).head()
