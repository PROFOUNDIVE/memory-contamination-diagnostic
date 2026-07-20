from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from memcontam.baselines.contracts import (
    BaselineExecutionOutcome,
    ErrorType,
    FailureDisposition,
    ReflexionAttemptOutcome,
    ReflexionReflectionEvent,
    ScientificIneligibilityReason,
    validate_failure_triple,
)


def normalize_direct_parent_ids(metadata: Mapping[str, Any]) -> list[str]:
    """Return recorded direct parents without promoting source or context evidence."""
    direct_parents = _string_list(metadata.get("direct_parent_ids"))
    return direct_parents or _string_list(metadata.get("parent_entry_ids"))


def validate_memory_item_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject metadata that claims source evidence as an exact direct parent."""
    direct_parents = normalize_direct_parent_ids(metadata)
    source_ids = _string_list(metadata.get("source_entry_ids"))
    if "direct_parent_ids" not in metadata and not _string_list(metadata.get("parent_entry_ids")):
        if direct_parents or source_ids and direct_parents:
            raise ValueError("source evidence cannot define direct parents")


def validate_failure_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate the closed failure triple persisted in trial metadata."""
    error_type = metadata.get("error_type")
    disposition = metadata.get("failure_disposition")
    reason = metadata.get("scientific_ineligibility_reason")
    if not all(isinstance(value, str) and value for value in (error_type, disposition, reason)):
        raise ValueError("failed evidence requires one complete failure triple")
    validate_failure_triple(
        cast(ErrorType, error_type),
        cast(FailureDisposition, disposition),
        cast(ScientificIneligibilityReason, reason),
    )


def validate_outcome_metadata(
    outcome: BaselineExecutionOutcome, metadata: Mapping[str, Any]
) -> None:
    """Require canonical scientific eligibility evidence only for failed outcomes."""
    failure_keys = {"failure_disposition", "scientific_ineligibility_reason"}
    if outcome.status == "succeeded":
        if failure_keys.intersection(metadata):
            raise ValueError("succeeded outcome cannot serialize failure evidence")
        return

    if metadata.get("scientific_ineligibility_reason") is None:
        raise ValueError("failed outcome requires scientific_ineligibility_reason metadata")
    if outcome.error_type is None or outcome.failure_disposition is None:
        raise ValueError("failed outcome requires one complete failure triple")
    persisted = dict(metadata)
    persisted["error_type"] = outcome.error_type
    validate_failure_metadata(persisted)
    if persisted["failure_disposition"] != outcome.failure_disposition:
        raise ValueError("failure_disposition metadata must match outcome")
    if persisted["scientific_ineligibility_reason"] != outcome.scientific_ineligibility_reason:
        raise ValueError("scientific_ineligibility_reason metadata must match outcome")


def validate_reflexion_attempt_records(
    attempts: Sequence[ReflexionAttemptOutcome],
    method_calls: Sequence[Any],
    *,
    retry_count: int | None = None,
) -> None:
    """Validate semantic Reflexion attempts against recorded calls, not retries."""
    if retry_count is not None:
        raise ValueError("attempt_index must not be inferred from retry_count")
    call_ids = _record_ids(method_calls, "call_id")
    attempt_ids: set[str] = set()
    attempt_indices: set[int] = set()
    for attempt in attempts:
        if attempt.attempt_index < 0 or attempt.attempt_index in attempt_indices:
            raise ValueError("attempt_index must be a unique non-negative semantic index")
        if attempt.attempt_id in attempt_ids:
            raise ValueError("attempt_id must be unique")
        if not attempt.answer_call_id or attempt.answer_call_id not in call_ids:
            raise ValueError("attempt answer_call_id must reference a recorded method call")
        attempt_ids.add(attempt.attempt_id)
        attempt_indices.add(attempt.attempt_index)


def validate_reflexion_reflection_events(
    attempts: Sequence[ReflexionAttemptOutcome],
    events: Sequence[ReflexionReflectionEvent],
    method_calls: Sequence[Any],
    memory_items: Sequence[Any],
) -> None:
    """Require reflections to join an authenticated incorrect attempt and evidence."""
    attempts_by_id = {attempt.attempt_id: attempt for attempt in attempts}
    call_ids = _record_ids(method_calls, "call_id")
    entry_ids = _record_ids(memory_items, "entry_id")
    for event in events:
        attempt = attempts_by_id.get(event.attempt_id)
        if attempt is None:
            raise ValueError("reflection event references an unknown attempt")
        if not _is_incorrect(attempt.outcome.verifier_result):
            raise ValueError("reflection event requires an authenticated incorrect attempt")
        if event.reflection_call_id not in call_ids:
            raise ValueError("reflection event references an unresolved reflection call")
        if event.reflection_entry_id not in entry_ids:
            raise ValueError("reflection event references an unresolved reflection entry")


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _record_ids(records: Sequence[Any], field: str) -> set[str]:
    ids: set[str] = set()
    for record in records:
        value = record.get(field) if isinstance(record, Mapping) else getattr(record, field, None)
        if isinstance(value, str) and value:
            ids.add(value)
    return ids


def _is_incorrect(verifier_result: Any) -> bool:
    if isinstance(verifier_result, bool):
        return not verifier_result
    if isinstance(verifier_result, Mapping):
        return verifier_result.get("is_correct") is False
    return getattr(verifier_result, "is_correct", None) is False
