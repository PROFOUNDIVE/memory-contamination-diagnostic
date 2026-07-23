from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from typing import Any, Literal, cast

from memcontam.evaluation.bootstrap import Interval, IntervalSet
from memcontam.experiment.phase12.contracts import BehaviorTestRow, canonical_json_hash


Classification = Literal["supported", "violated", "inconclusive", "not_applicable"]


class BehaviorEvaluationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class SeedInterval:
    estimate: float | str | None
    lower: float | str | None
    upper: float | str | None
    status: str = "estimable"
    resampling_unit: str = "seed"


@dataclass(frozen=True)
class BehaviorTestResult:
    test_id: str
    classification: Classification
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    threshold_rule: str = ""
    reason_code: str = ""
    evidence_hash: str = ""
    evidence: Mapping[str, Any] | None = None

    @property
    def status(self) -> Classification:
        return self.classification


_THRESHOLDS: dict[str, tuple[str, float]] = {
    "INV-01": ("equivalence", 0.1),
    "INV-02": ("equivalence", 0.1),
    "INV-03": ("equivalence", 0.1),
    "DIR-01": ("lower", 0.1),
    "DIR-02": ("upper", 0.0),
    "DIR-03": ("lower", 0.0),
    "DIR-04": ("lower", 0.0),
}


def evaluate_invariance(
    paired_metrics: Any,
    spec: BehaviorTestRow | Mapping[str, Any] | str,
) -> BehaviorTestResult:
    test_id, rule, canonical_id = _registered_spec(spec)
    if canonical_id not in {"INV-01", "INV-02", "INV-03"}:
        raise BehaviorEvaluationError("UNREGISTERED_BEHAVIOR_THRESHOLD")
    if _not_applicable(paired_metrics):
        return _result(test_id, "not_applicable", rule, "NOT_APPLICABLE", evidence=paired_metrics)

    if canonical_id == "INV-03":
        mechanical = _inv03_mechanical_status(paired_metrics)
        if mechanical == "not_applicable":
            return _result(
                test_id,
                "inconclusive",
                rule,
                "INV03_CANONICAL_CONTENT_UNAVAILABLE",
                evidence=paired_metrics,
            )
        if mechanical == "violated":
            return _result(
                test_id,
                "violated",
                rule,
                "INV03_CANONICAL_CONTENT_MISMATCH",
                evidence=paired_metrics,
            )

    interval = _coerce_interval(paired_metrics, test_id)
    values = _numeric_interval(interval)
    if values is None:
        return _result(test_id, "inconclusive", rule, "INTERVAL_NOT_ESTIMABLE", evidence=paired_metrics)
    estimate, lower, upper = values
    threshold = _THRESHOLDS[canonical_id][1]
    if lower >= -threshold and upper <= threshold:
        classification: Classification = "supported"
        reason = "INTERVAL_WITHIN_EQUIVALENCE_BAND"
    elif upper < -threshold or lower > threshold:
        classification = "violated"
        reason = "INTERVAL_OUTSIDE_EQUIVALENCE_BAND"
    else:
        classification = "inconclusive"
        reason = "INTERVAL_CROSSES_EQUIVALENCE_BAND"
    return _result(
        test_id,
        classification,
        rule,
        reason,
        estimate=estimate,
        lower=lower,
        upper=upper,
        evidence=paired_metrics,
    )


def _registered_spec(spec: BehaviorTestRow | Mapping[str, Any] | str) -> tuple[str, str, str]:
    if isinstance(spec, BehaviorTestRow):
        test_id = spec.test_id
        rule = spec.equivalence_tolerance_or_interval_rule
    elif isinstance(spec, Mapping):
        test_id = spec.get("test_id", spec.get("id"))
        rule = spec.get("equivalence_tolerance_or_interval_rule", spec.get("threshold_rule"))
    elif isinstance(spec, str):
        test_id = spec
        rule = None
    else:
        raise BehaviorEvaluationError("UNREGISTERED_BEHAVIOR_THRESHOLD")
    if not isinstance(test_id, str):
        raise BehaviorEvaluationError("UNREGISTERED_BEHAVIOR_THRESHOLD")
    canonical_id = _canonical_test_id(test_id)
    if canonical_id is None or canonical_id not in _THRESHOLDS:
        raise BehaviorEvaluationError("UNREGISTERED_BEHAVIOR_THRESHOLD")
    expected_kind, expected_value = _THRESHOLDS[canonical_id]
    if rule is not None and not _registered_rule(canonical_id, str(rule), expected_kind, expected_value):
        raise BehaviorEvaluationError("UNREGISTERED_BEHAVIOR_THRESHOLD")
    return test_id, "registered" if rule is None else str(rule), canonical_id


def _canonical_test_id(test_id: str) -> str | None:
    if test_id in _THRESHOLDS:
        return test_id
    for candidate in _THRESHOLDS:
        if test_id.startswith(f"{candidate}-"):
            return candidate
    return None


def _registered_rule(test_id: str, rule: str, kind: str, value: float) -> bool:
    normalized = rule.lower().replace(" ", "")
    if test_id.startswith("INV-"):
        if normalized == "interval-zero":
            return True
        if "epsilon=0.1" in normalized:
            return True
        return test_id == "INV-03" and normalized == "exact"
    if normalized == "directional-interval":
        return True
    if test_id == "DIR-01":
        return "delta_01=0.1" in normalized or "delta=0.1" in normalized
    suffix = test_id[-2:]
    return f"delta_{suffix}=0" in normalized or normalized == "delta=0"


def _not_applicable(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("not_applicable") is True
        or value.get("applicable") is False
        or value.get("status") == "not_applicable"
    )


def _coerce_interval(value: Any, test_id: str) -> SeedInterval:
    if isinstance(value, IntervalSet):
        _require_seed(value.resampling_unit)
        selected = value.intervals.get(test_id)
        if selected is None and len(value.intervals) == 1:
            selected = next(iter(value.intervals.values()))
        if selected is None:
            raise BehaviorEvaluationError("INVALID_BEHAVIOR_INTERVAL")
        return _coerce_interval(selected, test_id)
    if isinstance(value, Mapping):
        unit = value.get("resampling_unit", "seed")
        _require_seed(unit)
        selected = value.get("interval")
        if selected is None:
            canonical_id = _canonical_test_id(test_id)
            selected = value.get(test_id) or value.get(canonical_id) if canonical_id else None
        if selected is not None:
            return _coerce_interval(selected, test_id)
        if not {"estimate", "lower", "upper"}.issubset(value):
            raise BehaviorEvaluationError("INVALID_BEHAVIOR_INTERVAL")
        return SeedInterval(
            estimate=value.get("estimate"),
            lower=value.get("lower"),
            upper=value.get("upper"),
            status=str(value.get("status", "estimable")),
            resampling_unit=str(unit),
        )
    if isinstance(value, SeedInterval):
        _require_seed(value.resampling_unit)
        return value
    if isinstance(value, Interval):
        _require_seed(getattr(value, "resampling_unit", "seed"))
        return SeedInterval(value.estimate, value.lower, value.upper, value.status)
    if all(hasattr(value, name) for name in ("estimate", "lower", "upper")):
        unit = getattr(value, "resampling_unit", "seed")
        _require_seed(unit)
        return SeedInterval(
            getattr(value, "estimate"),
            getattr(value, "lower"),
            getattr(value, "upper"),
            str(getattr(value, "status", "estimable")),
        )
    raise BehaviorEvaluationError("INVALID_BEHAVIOR_INTERVAL")


def _require_seed(unit: Any) -> None:
    if unit != "seed":
        raise BehaviorEvaluationError("INVALID_RESAMPLING_UNIT")


def _numeric_interval(interval: SeedInterval) -> tuple[float, float, float] | None:
    if interval.status != "estimable":
        return None
    values = (interval.estimate, interval.lower, interval.upper)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return None
    numbers = tuple(float(cast(float, value)) for value in values)
    if any(not math.isfinite(value) for value in numbers):
        return None
    estimate, lower, upper = numbers
    if lower > upper:
        return None
    return estimate, lower, upper


def _inv03_mechanical_status(value: Any) -> Literal["supported", "violated", "not_applicable"]:
    if not isinstance(value, Mapping):
        return "not_applicable"
    correspondence = value.get("id_correspondence")
    canonical_ids = value.get("canonical_content_ids")
    canonical_content_id = value.get("canonical_content_id")
    reference_id = value.get("reference_canonical_content_id", value.get("reference_content_id"))
    variant_id = value.get("variant_canonical_content_id", value.get("variant_content_id"))
    if isinstance(correspondence, Mapping):
        ids = tuple(correspondence.values())
    elif isinstance(canonical_ids, (tuple, list)):
        ids = tuple(canonical_ids)
    elif reference_id is not None or variant_id is not None:
        ids = (reference_id, variant_id)
    else:
        ids = ()
    if not ids or any(not isinstance(item, str) or not item for item in ids):
        return "violated" if value.get("mechanical_gate_status") == "pass" else "not_applicable"
    if len(set(ids)) != 1:
        return "violated"
    if canonical_content_id is not None and ids[0] != canonical_content_id:
        return "violated"
    if value.get("mechanical_gate_status", "pass") != "pass":
        return "violated"
    return "supported"


def _result(
    test_id: str,
    classification: Classification,
    threshold_rule: str,
    reason_code: str,
    *,
    estimate: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    evidence: Any = None,
) -> BehaviorTestResult:
    normalized_evidence = _jsonable(evidence)
    payload = {
        "classification": classification,
        "estimate": estimate,
        "evidence": normalized_evidence,
        "lower": lower,
        "reason_code": reason_code,
        "test_id": test_id,
        "threshold_rule": threshold_rule,
        "upper": upper,
    }
    return BehaviorTestResult(
        test_id=test_id,
        classification=classification,
        estimate=estimate,
        lower=lower,
        upper=upper,
        threshold_rule=threshold_rule,
        reason_code=reason_code,
        evidence_hash=canonical_json_hash(payload),
        evidence=normalized_evidence if isinstance(normalized_evidence, Mapping) else None,
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in value.__dict__.items()}
    return str(value)


__all__ = [
    "BehaviorEvaluationError",
    "BehaviorTestResult",
    "Classification",
    "SeedInterval",
    "evaluate_invariance",
]
