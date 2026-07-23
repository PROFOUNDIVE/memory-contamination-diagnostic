from __future__ import annotations

from decimal import Decimal
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-ELIGIBILITY-001.json"


def _eligibility_module():
    assert importlib.util.find_spec("memcontam.experiment.phase12.eligibility") is not None
    return importlib.import_module("memcontam.experiment.phase12.eligibility")


def _maturity_module():
    assert importlib.util.find_spec("memcontam.experiment.phase12.maturity") is not None
    return importlib.import_module("memcontam.experiment.phase12.maturity")


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _decisions(family: str, indices: list[int], *, horizon: int = 3):
    maturity = _maturity_module()
    return tuple(
        maturity.MaturityDecision(
            condition_id=family,
            baseline_family=family,
            checkpoint_id=f"{family}-checkpoint-{index}",
            checkpoint_index=index,
            horizon=horizon,
            eligible=True,
        )
        for index in indices
    )


def test_selects_registered_common_checkpoint_from_eligible_panel() -> None:
    eligibility = _eligibility_module()
    fixture = _fixture()
    decisions = (
        *_decisions("fh", fixture["baseline_eligible"]["fh"]),
        *_decisions("rag_frozen", fixture["baseline_eligible"]["rag_frozen"]),
        *_decisions("bot", fixture["baseline_eligible"]["bot"]),
        *_decisions("reflexion", fixture["baseline_eligible"]["reflexion"]),
        *_decisions("dc", [5]),
    )

    result = eligibility.compute_joint_eligibility(decisions, horizon=fixture["horizon"])

    assert result.baseline_eligible == fixture["baseline_eligible"]
    assert result.primary_intersection == tuple(fixture["joint"])
    assert result.joint_eligible_indices == tuple(fixture["joint"])
    assert result.not_estimable is fixture["not_estimable"]
    assert result.estimability_counts == {"eligible_checkpoints": 2, "primary_baselines": 4}
    assert eligibility.select_lower_quantile_checkpoint(
        result.joint_eligible_indices, Decimal("0.5")
    ) == 3


def test_rejects_noncommon_relaxed_or_short_horizon_selection() -> None:
    eligibility = _eligibility_module()
    decisions = (
        *_decisions("fh", [3]),
        *_decisions("rag", [4]),
        *_decisions("bot", [3]),
        *_decisions("reflexion", [3]),
    )

    result = eligibility.compute_joint_eligibility(decisions, horizon=3)

    assert result.not_estimable is True
    assert result.joint_eligible_indices == ()
    assert eligibility.select_lower_quantile_checkpoint(result.joint_eligible_indices, Decimal("0.5")) is None
    with pytest.raises(eligibility.EligibilityError, match="RELAXED_MATURITY_THRESHOLD"):
        eligibility.compute_joint_eligibility((*decisions, *_decisions("fh", [5], horizon=2)), 3)
    with pytest.raises(eligibility.EligibilityError, match="INVALID_HORIZON"):
        eligibility.compute_joint_eligibility(decisions, 0)


def test_excludes_nomem_aliases_from_all_eligibility_results() -> None:
    eligibility = _eligibility_module()
    decisions = (
        *_decisions("fh", [3]),
        *_decisions("rag", [3]),
        *_decisions("bot", [3]),
        *_decisions("reflexion", [3]),
        *_decisions("nomem", [3]),
        *_decisions("no_memory", [4]),
        *_decisions("NoMem", [5]),
        *_decisions("dc", [99]),
    )

    result = eligibility.compute_joint_eligibility(decisions, horizon=3)

    assert result.baseline_eligible == {"fh": [3], "rag": [3], "bot": [3], "reflexion": [3]}
    assert result.optional_eligible == {"dc": [99]}
    assert result.primary_intersection == (3,)
    assert result.estimability_counts == {"eligible_checkpoints": 1, "primary_baselines": 4}
