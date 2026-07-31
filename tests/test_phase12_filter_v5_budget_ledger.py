from __future__ import annotations

import pytest

from memcontam.experiment.phase12.filter_challenge.bct_archive import BudgetLedger, LedgerError


def test_ledger_retains_unresolved_wall_reservations_across_restart(tmp_path) -> None:
    path = tmp_path / "budget-ledger.jsonl"
    ledger = BudgetLedger(path)
    ledger.reserve_process("screening", "screen-001")

    restarted = BudgetLedger(path)

    assert restarted.remaining_wall_seconds == 7200
    restarted.reserve_process("screening", "screen-002")
    with pytest.raises(LedgerError, match="WALL_TIME_CAP_EXCEEDED"):
        restarted.reserve_process("bct", "bct-001")
