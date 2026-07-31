from __future__ import annotations

from memcontam.experiment.phase12.filter_challenge.bct_archive import BudgetLedger


def test_settled_wall_time_reconstructs_bct_deadline_slice(tmp_path) -> None:
    ledger = BudgetLedger(tmp_path / "budget-ledger.jsonl")
    reservation = ledger.reserve_process("screening", "screen-001")
    ledger.settle_process(reservation, 600)

    bct = BudgetLedger(ledger.path).reserve_process("bct", "bct-001")

    assert BudgetLedger(ledger.path).remaining_wall_seconds == 3000
    assert bct.reserved_wall_seconds == 7200
