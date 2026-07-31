from __future__ import annotations

from memcontam.experiment.phase12.filter_challenge.bct_archive import BudgetLedger, ResourceBudget


def test_settled_wall_time_reconstructs_bct_deadline_slice(tmp_path) -> None:
    ledger = BudgetLedger(tmp_path / "budget-ledger.jsonl")
    reservation = ledger.reserve_process("screening", "screen-001")
    ledger.settle(reservation, ResourceBudget(1, 600, 60, 10_000, 600))

    bct = BudgetLedger(ledger.path).reserve_process("bct", "bct-001")

    assert BudgetLedger(ledger.path).remaining_wall_seconds == 3000
    assert bct.reserved_wall_seconds == 7200


def test_deadline_clamps_requests_and_timeout_invalidation_retains_reservation(tmp_path) -> None:
    ledger = BudgetLedger(tmp_path / "budget-ledger.jsonl")
    screening = ledger.reserve_process("screening", "screen-001")
    ledger.settle(screening, ResourceBudget(1, 600, 60, 10_000, 600))
    bct = ledger.reserve_process("bct", "bct-001")
    deadline = ledger.deadline_for(bct, started_at=10.0)

    assert deadline.clamp_timeout(9_000, monotonic_now=10.0) == 7_200
    assert deadline.clamp_timeout(30, monotonic_now=7_211.0) == 0
    ledger.invalidate_timeout(bct)

    assert BudgetLedger(ledger.path).remaining_resources.wall_seconds == 3_000
    assert '"kind":"invalidate"' in ledger.path.read_text(encoding="utf-8")
