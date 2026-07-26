from __future__ import annotations

import warnings

import pytest

from memcontam.clients.cost_guard import CostGuard, CostLimitExceeded, MissingUsageError


def test_cost_guard_warns_once_when_run_cost_reaches_three_usd() -> None:
    guard = CostGuard()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        guard.record_usage({"input_tokens": 1_000_000, "output_tokens": 50_000})
        guard.record_usage({"input_tokens": 1, "output_tokens": 1})

    assert guard.spent_usd == pytest.approx(3.0000125)
    assert len(captured) == 1
    assert "USD 3" in str(captured[0].message)


def test_cost_guard_rejects_projected_cost_over_hard_ceiling_before_dispatch() -> None:
    guard = CostGuard(warning_usd=6)
    guard.record_usage({"input_tokens": 1_000_000, "output_tokens": 240_000})

    with pytest.raises(CostLimitExceeded, match="USD 5"):
        guard.check_before_dispatch(0.11)

    assert guard.spent_usd == pytest.approx(4.9)


def test_cost_guard_fails_closed_for_missing_or_empty_usage() -> None:
    guard = CostGuard()

    with pytest.raises(MissingUsageError, match="usage"):
        guard.record_usage(None)
    with pytest.raises(MissingUsageError, match="usage"):
        guard.record_usage({})
