from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from memcontam.evaluation.phase13_observability_models import (
    MetricValue,
    Phase13ObservabilityError,
    Phase13TrialAnalysis,
    Phase13TrialEvidence,
)
from .phase13_observability_registration import classify_registered_failure


def has_registered_recurrence(classes: Sequence[str | None], lookback: int) -> bool:
    current = classes[-1]
    return current is not None and current in classes[max(0, len(classes) - lookback - 1) : -1]


def first_continuous_episode(
    values: Sequence[bool], endpoint: Literal["core_prefix_50"]
) -> MetricValue:
    try:
        onset = values.index(True)
    except ValueError:
        return MetricValue(status="not_estimable", reason="NO_FIRST_EPISODE_ONSET")
    loss = next((index for index in range(onset + 1, len(values)) if not values[index]), None)
    if loss is not None:
        return MetricValue(
            status="supported",
            value=loss - onset,
            reason="FIRST_CONTINUOUS_EPISODE",
            censoring_status="OBSERVED_END",
        )
    return MetricValue(
        status="supported",
        value=len(values) - onset,
        reason="FIRST_CONTINUOUS_EPISODE",
        censoring_status="RIGHT_CENSORED",
        censoring_endpoint_analysis_window_id=endpoint,
    )


def is_exact_root_eviction(
    root_id: str,
    before_ids: Sequence[str],
    removed_ids: Sequence[str],
    after_ids: Sequence[str],
) -> bool:
    return root_id in before_ids and root_id in removed_ids and root_id not in after_ids


def reconstruct_registered_sequence(
    evidence_rows: tuple[Phase13TrialEvidence, ...],
    rows: tuple[Phase13TrialAnalysis, ...],
    lookback: int,
) -> tuple[Phase13TrialAnalysis, ...]:
    if len(evidence_rows) != len(rows) or not rows:
        raise Phase13ObservabilityError("SEQUENCE_EVIDENCE_MISMATCH")
    _validate_continuity(evidence_rows)
    failures = tuple(
        classify_registered_failure(row.task, row.verified_outcome, evidence.trial.failure_class)
        for evidence, row in zip(evidence_rows, rows, strict=True)
    )
    classes = tuple(metric.value if isinstance(metric.value, str) else None for metric in failures)
    exposed_roots = tuple(_exposed_roots(evidence) for evidence in evidence_rows)
    roots = tuple(_root_ids(evidence) for evidence in evidence_rows)
    descendants_by_row = _cumulative_descendants(rows)
    registered_roots = {root_id for row_roots in roots for root_id in row_roots}
    if len(registered_roots) != 1:
        raise Phase13ObservabilityError("FIXTURE_EXACTLY_ONE_REGISTERED_ROOT_REQUIRED")
    registered_root = next(iter(registered_roots))
    evictions: dict[str, int] = {}
    for index, evidence in enumerate(evidence_rows):
        if is_exact_root_eviction(
            registered_root,
            evidence.memory_before_ids,
            evidence.removed_entry_ids,
            evidence.memory_after_ids,
        ):
            evictions.setdefault(registered_root, index)
    root_retention = first_continuous_episode(
        tuple(registered_root in evidence.memory_after_ids for evidence in evidence_rows),
        rows[0].analysis_window_id,
    )
    exposure_retention = first_continuous_episode(
        tuple(row.theory_exposure.value is True for row in rows), rows[0].analysis_window_id
    )
    descendant_ids = {entry_id for row in descendants_by_row for entry_id in row}
    descendant_retention = first_continuous_episode(
        tuple(bool(descendant_ids & set(evidence.memory_after_ids)) for evidence in evidence_rows),
        rows[0].analysis_window_id,
    )
    result = []
    for index, (evidence, row) in enumerate(zip(evidence_rows, rows, strict=True)):
        generic = has_registered_recurrence(classes[: index + 1], lookback)
        prior_start = max(0, index - lookback)
        exact = any(
            classes[prior] == classes[index]
            and bool(exposed_roots[prior] & exposed_roots[index])
            for prior in range(prior_start, index)
        ) if classes[index] is not None else False
        post_eviction = _post_eviction(index, classes, exposed_roots, evictions, lookback)
        result.append(
            row.model_copy(
                update={
                    "failure_class": failures[index],
                    "descendant_entry_ids": descendants_by_row[index],
                    "descendant_storage_persistence": _descendant_metric(
                        descendants_by_row[index], evidence.memory_after_ids
                    ),
                    "descendant_prompt_visibility": _descendant_metric(
                        descendants_by_row[index],
                        () if evidence.context is None else tuple(evidence.context.final_entry_ids),
                    ),
                    "generic_recurrence": _indicator(generic, classes[index]),
                    "exact_lineage_recurrence": _indicator(exact, classes[index]),
                    "exposure_conditioned_recurrence": _exposure_indicator(row, generic, classes[index]),
                    "post_eviction_recurrence": post_eviction,
                    "root_retention_duration": root_retention,
                    "prompt_retention_duration": exposure_retention,
                    "descendant_retention_duration": descendant_retention,
                }
            )
        )
    return tuple(result)


def _cumulative_descendants(
    rows: tuple[Phase13TrialAnalysis, ...],
) -> tuple[tuple[str, ...], ...]:
    known: set[str] = set()
    result = []
    for row in rows:
        known.update(row.descendant_entry_ids)
        result.append(tuple(sorted(known)))
    return tuple(result)


def _descendant_metric(
    descendants: tuple[str, ...], container: Sequence[str]
) -> MetricValue:
    if not descendants:
        return MetricValue(status="not_applicable", reason="NO_RECORDED_DESCENDANT")
    return MetricValue(
        status="supported",
        value=bool(set(descendants).intersection(container)),
        reason="RECORDED_IDENTITY_INTERSECTION",
    )


def _validate_continuity(rows: tuple[Phase13TrialEvidence, ...]) -> None:
    if any(
        row.trial.analysis_inclusion != "included"
        or row.trial.parse_status != "parsed"
        or row.trial.execution_status != "completed"
        for row in rows
    ):
        raise Phase13ObservabilityError("ORDINARY_SEQUENCE_CONTINUITY_MISMATCH")
    first = rows[0]
    identity = (
        first.task,
        first.baseline,
        first.trajectory_seed,
        first.concrete_seed_id,
        first.analysis_window_id,
        first.trial.execution_key,
        first.trial.checkpoint_id,
    )
    trial_ids = {row.trial_id for row in rows}
    if len(trial_ids) != len(rows):
        raise Phase13ObservabilityError("ORDINARY_SEQUENCE_CONTINUITY_MISMATCH")
    for previous, current in zip(rows, rows[1:]):
        if (
            (
                current.task,
                current.baseline,
                current.trajectory_seed,
                current.concrete_seed_id,
                current.analysis_window_id,
                current.trial.execution_key,
                current.trial.checkpoint_id,
            )
            != identity
            or current.order_key != previous.order_key + 1
            or not isinstance(previous.trial.event_time, int)
            or not isinstance(current.trial.event_time, int)
            or current.trial.event_time != previous.trial.event_time + 1
            or current.trial.prefix_run_id != previous.trial.prefix_run_id
            or current.trial.branch_id != previous.trial.branch_id
            or current.memory_before_ids != previous.memory_after_ids
        ):
            raise Phase13ObservabilityError("ORDINARY_SEQUENCE_CONTINUITY_MISMATCH")


def _root_ids(evidence: Phase13TrialEvidence) -> set[str]:
    return {
        node.entry_id
        for node in evidence.lineage
        if node.entry_id in evidence.target_set.target_entry_ids
        and node.entry_id in node.injected_root_ids
        and not node.direct_parent_ids
        and node.version_predecessor_id is None
    }


def _exposed_roots(evidence: Phase13TrialEvidence) -> set[str]:
    return {
        root_id
        for span in evidence.target_set.answer_call_spans
        for root_id in span.injected_root_ids
        if span.lineage_status == "exact"
    }


def _indicator(value: bool, failure_class: str | None) -> MetricValue:
    if failure_class is None:
        return MetricValue(status="not_applicable", reason="NO_REGISTERED_FAILURE")
    return MetricValue(status="supported", value=value, reason="RECURRENCE_LOOKBACK_H10")


def _exposure_indicator(
    row: Phase13TrialAnalysis, recurrence: bool, failure_class: str | None
) -> MetricValue:
    if failure_class is None or row.theory_exposure.value is not True:
        return MetricValue(status="not_applicable", reason="CURRENT_ROW_NOT_EXPOSED_REGISTERED_FAILURE")
    return MetricValue(status="supported", value=recurrence, reason="CURRENT_Z_T_EQUALS_1")


def _post_eviction(
    index: int,
    classes: tuple[str | None, ...],
    exposed_roots: tuple[set[str], ...],
    evictions: dict[str, int],
    lookback: int,
) -> MetricValue:
    eligible = tuple((root, eviction) for root, eviction in evictions.items() if index > eviction)
    if not eligible:
        return MetricValue(status="not_estimable", reason="NO_ELIGIBLE_POST_EVICTION_RISK_ROW")
    if classes[index] is None:
        return MetricValue(status="supported", value=False, reason="POST_EVICTION_RISK_SET")
    value = any(
        classes[prior] == classes[index] and root in exposed_roots[prior]
        for root, eviction in eligible
        for prior in range(max(0, index - lookback), min(eviction, index - 1) + 1)
    )
    return MetricValue(status="supported", value=value, reason="EXACT_ROOT_POST_EVICTION_RISK_SET")


__all__ = [
    "first_continuous_episode",
    "has_registered_recurrence",
    "is_exact_root_eviction",
    "reconstruct_registered_sequence",
]
