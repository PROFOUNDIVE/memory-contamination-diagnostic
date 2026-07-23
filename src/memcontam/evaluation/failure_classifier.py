from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias


FailureClassId: TypeAlias = str


@dataclass(frozen=True)
class FailureClassifier:
    class_id: FailureClassId
    classify: Callable[[Any, Any, Any], bool | None]

    def __post_init__(self) -> None:
        if not self.class_id:
            raise ValueError("FAILURE_CLASS_ID_REQUIRED")


def classify_failure(
    q: Any,
    y: Any,
    v: Any,
    registry: Mapping[str, Sequence[FailureClassifier]],
) -> FailureClassId | None:
    """Return one registered task-specific failure class, never a generic error label."""
    task_name = _task_name(q)
    classifiers = registry.get(task_name, ())
    matches: list[FailureClassId] = []
    for classifier in classifiers:
        matched = classifier.classify(q, y, v)
        if not isinstance(matched, (bool, type(None))):
            raise ValueError("INVALID_FAILURE_CLASSIFIER_RESULT")
        if matched:
            matches.append(classifier.class_id)
    if len(matches) > 1:
        raise ValueError("AMBIGUOUS_FAILURE_CLASSIFICATION")
    return matches[0] if matches else None


def _task_name(query: Any) -> str:
    if isinstance(query, Mapping):
        for key in ("task_name", "task_family"):
            value = query.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("task_name", "task_family"):
        value = getattr(query, key, None)
        if isinstance(value, str) and value:
            return value
    raise ValueError("TASK_FAILURE_CLASSIFIER_REQUIRED")


__all__ = ["FailureClassId", "FailureClassifier", "classify_failure"]
