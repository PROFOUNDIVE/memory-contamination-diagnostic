from __future__ import annotations

from decimal import Decimal
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-ELIGIBILITY-001.json"


def _timing_module():
    assert importlib.util.find_spec("memcontam.experiment.phase12.timing") is not None
    return importlib.import_module("memcontam.experiment.phase12.timing")


def test_uses_exact_finite_set_quantiles_for_early_base_and_late() -> None:
    timing = _timing_module()
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    eligible = fixture["joint"]

    assert timing.select_timing_checkpoint(eligible, "early") == fixture["quantiles"]["0.25"]
    assert timing.select_timing_checkpoint(eligible, "base") == fixture["quantiles"]["0.5"]
    assert timing.select_timing_checkpoint(eligible, "late") == fixture["quantiles"]["0.75"]


def test_returns_not_estimable_for_empty_support_and_rejects_invalid_quantiles() -> None:
    timing = _timing_module()

    assert timing.select_timing_checkpoint((), "base") is None
    with pytest.raises(timing.EligibilityError, match="INVALID_QUANTILE"):
        timing.select_lower_quantile_checkpoint((3, 5), Decimal("1.1"))
