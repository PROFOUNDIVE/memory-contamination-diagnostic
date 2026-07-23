from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from memcontam.experiment.phase12.contracts import BehaviorTestRow

from memcontam.behavior.invariance import (
    BehaviorEvaluationError,
    BehaviorTestResult,
    _THRESHOLDS,
    _coerce_interval,
    _not_applicable,
    _numeric_interval,
    _registered_spec,
    _result,
)


def evaluate_direction(
    interval: Any,
    spec: BehaviorTestRow | Mapping[str, Any] | str,
) -> BehaviorTestResult:
    test_id, rule, canonical_id = _registered_spec(spec)
    if canonical_id not in {"DIR-01", "DIR-02", "DIR-03", "DIR-04"}:
        raise BehaviorEvaluationError("UNREGISTERED_BEHAVIOR_THRESHOLD")
    if _not_applicable(interval):
        return _result(test_id, "not_applicable", rule, "NOT_APPLICABLE", evidence=interval)
    parsed = _coerce_interval(interval, test_id)
    values = _numeric_interval(parsed)
    if values is None:
        return _result(test_id, "inconclusive", rule, "INTERVAL_NOT_ESTIMABLE", evidence=interval)
    estimate, lower, upper = values
    direction, threshold = _THRESHOLDS[canonical_id]
    classification: Literal["supported", "violated", "inconclusive", "not_applicable"]
    if direction == "lower":
        if lower >= threshold:
            classification = "supported"
            reason = "LOWER_BOUND_MEETS_DIRECTION"
        elif upper < threshold:
            classification = "violated"
            reason = "UPPER_BOUND_FAILS_DIRECTION"
        else:
            classification = "inconclusive"
            reason = "INTERVAL_CROSSES_DIRECTION_THRESHOLD"
    else:
        if upper <= threshold:
            classification = "supported"
            reason = "UPPER_BOUND_MEETS_DIRECTION"
        elif lower > threshold:
            classification = "violated"
            reason = "LOWER_BOUND_FAILS_DIRECTION"
        else:
            classification = "inconclusive"
            reason = "INTERVAL_CROSSES_DIRECTION_THRESHOLD"
    return _result(
        test_id,
        classification,
        rule,
        reason,
        estimate=estimate,
        lower=lower,
        upper=upper,
        evidence=interval,
    )


__all__ = ["BehaviorEvaluationError", "BehaviorTestResult", "evaluate_direction"]
