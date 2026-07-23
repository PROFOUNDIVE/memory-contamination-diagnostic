from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pytest

from memcontam.behavior.directional import BehaviorEvaluationError, evaluate_direction
from memcontam.behavior.invariance import SeedInterval
from memcontam.behavior.registry import load_behavior_registry_bundle
from memcontam.evaluation.bootstrap import Interval, IntervalSet


ROOT = Path(__file__).parents[1]
REGISTRIES = ROOT / "data" / "phase12" / "registries"


def test_rejects_unregistered_threshold_or_trial_level_interval() -> None:
    bundle = load_behavior_registry_bundle(REGISTRIES)
    row = next(row for row in bundle.behavior_tests.rows if row.test_id == "DIR-01")
    invalid_threshold = row.model_copy(update={"equivalence_tolerance_or_interval_rule": "delta=9"})

    with pytest.raises(BehaviorEvaluationError, match="UNREGISTERED_BEHAVIOR_THRESHOLD"):
        evaluate_direction(SeedInterval(0.2, 0.1, 0.3), invalid_threshold)

    trial_interval = IntervalSet(
        {"DIR-01": Interval(0.2, 0.1, 0.3)},
        resampling_unit=cast(Literal["seed"], "trial"),
    )
    with pytest.raises(BehaviorEvaluationError, match="INVALID_RESAMPLING_UNIT"):
        evaluate_direction(trial_interval, row)
