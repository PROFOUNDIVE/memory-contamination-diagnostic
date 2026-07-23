from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from statistics import mean
from typing import Any, Literal, cast

from memcontam.evaluation.compatibility import CompatibilityError, build_compatibility_key
from memcontam.evaluation.estimability import (
    EstimabilityDecision,
    EstimabilityRule,
    evaluate_estimability,
)
from memcontam.evaluation.phase12_observables import ObservableRecord
from memcontam.evaluation.sequential import SequentialTrialOutcome
from memcontam.experiment.phase12.contracts import (
    ValidatedExploratoryActivation,
    ValidatedRouteSelection,
)
from memcontam.logging.schema_v3 import (
    MemoryArmExecutionKey,
    RunMetadataV3,
    ScientificExploratoryCodeRunMetadata,
    SelectedRouteRunMetadata,
)


NOT_ESTIMABLE = "not_estimable"
_ARMS = ("clean", "correct", "irrelevant", "contam", "filter")


class AggregateError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class AggregateTrial:
    trial_id: str
    verified_score: Literal[0, 1] | None
    analysis_inclusion: str
    execution_status: str
    failure_class: str | None = None
    rerun_trial_id_or_none: str | None = None
    observables: ObservableRecord | None = None
    sequential: SequentialTrialOutcome | None = None
    eligible: bool | None = None
    operational: Mapping[str, int | float | bool] | None = None


@dataclass(frozen=True)
class ValidatedRun:
    metadata: RunMetadataV3
    trials: tuple[AggregateTrial, ...]
    route_selection: ValidatedRouteSelection | None = None
    exploratory_activation: ValidatedExploratoryActivation | None = None


@dataclass(frozen=True)
class AggregateSpec:
    expected_arms: tuple[str, ...] = _ARMS
    estimability_rule: EstimabilityRule = EstimabilityRule()
    resampling_unit: Literal["seed", "trial"] = "seed"
    complete_case_policy: Literal["reject", "drop"] = "reject"
    weights: Mapping[int, float] | None = None
    bootstrap_config: Any | None = None


@dataclass(frozen=True)
class AggregateCell:
    population: Mapping[str, str | None]
    seed_scores: Mapping[int, Mapping[str, float]]
    contrasts: Mapping[str, float | str]
    metrics: Mapping[str, float | int | str]
    estimability: EstimabilityDecision
    intervals: Any | None

    @property
    def seed_count(self) -> int:
        return len(self.seed_scores)


@dataclass(frozen=True)
class Phase12Aggregate:
    cells: tuple[AggregateCell, ...]


def aggregate_phase12(runs: Sequence[ValidatedRun], spec: AggregateSpec) -> Phase12Aggregate:
    _validate_spec(spec)
    if not runs:
        raise AggregateError("EMPTY_RUNS")
    trial_ids = {trial.trial_id for run in runs for trial in run.trials}
    groups: dict[tuple[str, ...], list[ValidatedRun]] = defaultdict(list)
    for run in runs:
        _validate_run(run, trial_ids)
        groups[_population_key(run)].append(run)
    return Phase12Aggregate(
        tuple(_aggregate_cell(key, cell_runs, spec) for key, cell_runs in groups.items())
    )


def _validate_spec(spec: AggregateSpec) -> None:
    if spec.resampling_unit != "seed":
        raise AggregateError("TRIAL_RESAMPLING_FORBIDDEN")
    if spec.complete_case_policy != "reject":
        raise AggregateError("COMPLETE_CASE_SELECTION_FORBIDDEN")
    if spec.weights is not None:
        raise AggregateError("WEIGHT_RENORMALIZATION_FORBIDDEN")
    if tuple(spec.expected_arms) != _ARMS:
        raise AggregateError("INCOMPLETE_FIVE_ARM_PAIR")


def _validate_run(run: ValidatedRun, trial_ids: set[str]) -> None:
    _validate_governance(run)
    for trial in run.trials:
        if not trial.trial_id:
            raise AggregateError("INVALID_TRIAL")
        if trial.analysis_inclusion == "included":
            if trial.execution_status != "completed" or trial.verified_score not in {0, 1}:
                raise AggregateError("INVALID_OUTCOME")
            continue
        if (
            trial.execution_status != "invalidated"
            or not trial.rerun_trial_id_or_none
            or trial.rerun_trial_id_or_none not in trial_ids
        ):
            raise AggregateError("INVALIDATED_RERUN_LINK_REQUIRED")


def _validate_governance(run: ValidatedRun) -> None:
    metadata = run.metadata
    if isinstance(metadata, SelectedRouteRunMetadata):
        selection = run.route_selection
        if (
            selection is None
            or metadata.route_selection_manifest_id != selection.route_selection_manifest_id
            or metadata.seed_allocation_manifest_id != selection.seed_allocation_manifest_id
            or not metadata.abstract_seed_slot_or_none
            or selection.slot_to_seed.get(metadata.abstract_seed_slot_or_none)
            != metadata.trajectory_seed
        ):
            raise AggregateError("SEED_ASSIGNMENT_MISMATCH")
    elif isinstance(metadata, ScientificExploratoryCodeRunMetadata):
        activation = run.exploratory_activation
        if (
            activation is None
            or metadata.exploratory_activation_manifest_id
            != activation.exploratory_activation_manifest_id
            or metadata.source_route_selection_manifest_id != activation.route_selection_manifest_id
            or metadata.source_seed_allocation_manifest_id != activation.seed_allocation_manifest_id
            or not metadata.abstract_seed_slot_or_none
            or activation.exploratory_slot_to_seed.get(metadata.abstract_seed_slot_or_none)
            != metadata.trajectory_seed
        ):
            raise AggregateError("ACTIVATION_MISMATCH")


def _population_key(run: ValidatedRun) -> tuple[str, ...]:
    metadata = run.metadata
    return (
        metadata.metadata_kind,
        metadata.protocol_version,
        metadata.evidence_layer,
        metadata.run_family,
        metadata.task_family,
        metadata.baseline_condition_id,
        _canonical(metadata.sensitivity_cell_ref),
        _governance_id(
            metadata, "route_selection_manifest_id", "source_route_selection_manifest_id"
        ),
        _governance_id(
            metadata, "seed_allocation_manifest_id", "source_seed_allocation_manifest_id"
        ),
        _governance_id(metadata, "exploratory_activation_manifest_id"),
    )


def _governance_id(metadata: Any, *names: str) -> str:
    for name in names:
        value = getattr(metadata, name, None)
        if value is not None:
            return str(value)
    return ""


def _aggregate_cell(
    population_key: tuple[str, ...], runs: Sequence[ValidatedRun], spec: AggregateSpec
) -> AggregateCell:
    signatures = {_compatibility_signature(run.metadata) for run in runs}
    if len(signatures) != 1:
        raise AggregateError("COMPATIBILITY_MISMATCH")
    scores: dict[int, dict[str, float]] = defaultdict(dict)
    records_by_seed: dict[int, list[AggregateTrial]] = defaultdict(list)
    for run in runs:
        arm = _arm(run)
        seed = run.metadata.trajectory_seed
        if arm in scores[seed]:
            raise AggregateError("DUPLICATE_SEED_ARM")
        scores[seed][arm] = _seed_arm_score(run.trials)
        records_by_seed[seed].extend(run.trials)
    for seed, arm_scores in scores.items():
        if set(arm_scores) != set(spec.expected_arms):
            raise AggregateError("INCOMPLETE_FIVE_ARM_PAIR")
    contrast_values = {
        name: {seed: _contrast(arm_scores, left, right) for seed, arm_scores in scores.items()}
        for name, left, right in (
            ("clean_minus_contam", "clean", "contam"),
            ("clean_minus_filter", "clean", "filter"),
            ("correct_minus_contam", "correct", "contam"),
            ("filter_minus_contam", "filter", "contam"),
            ("irrelevant_minus_contam", "irrelevant", "contam"),
        )
    }
    decisions = [
        evaluate_estimability(values, spec.estimability_rule) for values in contrast_values.values()
    ]
    estimability = next(
        (decision for decision in decisions if not decision.estimable), decisions[0]
    )
    contrasts: dict[str, float | str] = {
        name: mean(values.values()) if decision.estimable else NOT_ESTIMABLE
        for (name, values), decision in zip(contrast_values.items(), decisions, strict=True)
    }
    intervals = None
    if spec.bootstrap_config is not None:
        from memcontam.evaluation.bootstrap import bootstrap_seeds

        intervals = bootstrap_seeds(contrast_values, spec.bootstrap_config)
    return AggregateCell(
        population=_population(population_key),
        seed_scores={seed: dict(arms) for seed, arms in sorted(scores.items())},
        contrasts=contrasts,
        metrics=_metrics(records_by_seed),
        estimability=estimability,
        intervals=intervals,
    )


def _compatibility_signature(metadata: RunMetadataV3) -> tuple[Any, ...]:
    try:
        key = build_compatibility_key(metadata)
    except CompatibilityError as error:
        raise AggregateError(error.code) from error
    return tuple(
        getattr(key, field.name)
        for field in fields(key)
        if field.name not in {"execution_key", "run_template_id"}
    )


def _arm(run: ValidatedRun) -> str:
    execution_key = run.metadata.execution_key
    if not isinstance(execution_key, MemoryArmExecutionKey):
        raise AggregateError("INCOMPLETE_FIVE_ARM_PAIR")
    return execution_key.arm


def _seed_arm_score(trials: Sequence[AggregateTrial]) -> float:
    scores = [
        cast(Literal[0, 1], trial.verified_score)
        for trial in trials
        if trial.analysis_inclusion == "included"
    ]
    if not scores:
        raise AggregateError("INCOMPLETE_FIVE_ARM_PAIR")
    return mean(scores)


def _contrast(scores: Mapping[str, float], left: str, right: str) -> float:
    return scores[left] - scores[right]


def _metrics(
    records_by_seed: Mapping[int, Sequence[AggregateTrial]],
) -> dict[str, float | int | str]:
    records = [record for seed_records in records_by_seed.values() for record in seed_records]
    exposures = [
        record
        for record in records
        if record.analysis_inclusion == "included"
        and record.observables is not None
        and record.observables.exposure.status == "supported"
        and record.observables.exposure.is_exposed is not None
    ]
    exposure_values = [
        bool(record.observables.exposure.is_exposed) for record in exposures if record.observables
    ]
    supported_contrast = [
        record
        for record in exposures
        if record.observables and record.observables.exposure.exposed_non_exposed_contrast_supported
    ]
    exposed_scores = [
        record.verified_score
        for record in supported_contrast
        if record.observables
        and record.observables.exposure.is_exposed
        and record.verified_score is not None
    ]
    unexposed_scores = [
        record.verified_score
        for record in supported_contrast
        if record.observables
        and not record.observables.exposure.is_exposed
        and record.verified_score is not None
    ]
    sequentials = [record.sequential for record in records if record.sequential is not None]
    uses = [
        record.observables.use.is_used
        for record in records
        if record.observables is not None and record.observables.use.status == "supported"
    ]
    eligible = [
        all(record.eligible is True for record in seed_records)
        for seed_records in records_by_seed.values()
    ]
    return {
        "model_behavior_row_count": sum(
            record.failure_class == "model_behavior" for record in records
        ),
        "invalidated_row_count": sum(record.analysis_inclusion != "included" for record in records),
        "exposure_rate": mean(exposure_values) if exposure_values else NOT_ESTIMABLE,
        "exposure_score_contrast": (
            mean(exposed_scores) - mean(unexposed_scores)
            if exposed_scores and unexposed_scores
            else NOT_ESTIMABLE
        ),
        "generic_recurrence_rate": (
            mean(outcome.generic_recurrence for outcome in sequentials)
            if sequentials
            else NOT_ESTIMABLE
        ),
        "exact_lineage_recurrence_rate": (
            mean(outcome.same_root_exact_lineage_recurrence for outcome in sequentials)
            if sequentials
            else NOT_ESTIMABLE
        ),
        "propagation_rate": (
            mean(outcome.propagation.value is True for outcome in sequentials)
            if sequentials
            else NOT_ESTIMABLE
        ),
        "eligible_seed_rate": mean(eligible) if eligible else NOT_ESTIMABLE,
        "operational_use_rate": mean(uses) if uses else NOT_ESTIMABLE,
    }


def _population(values: tuple[str, ...]) -> dict[str, str | None]:
    names = (
        "metadata_kind",
        "protocol_version",
        "evidence_layer",
        "run_family",
        "task_family",
        "baseline_condition_id",
        "sensitivity_cell_ref",
        "route_selection_manifest_id",
        "seed_allocation_manifest_id",
        "exploratory_activation_manifest_id",
    )
    return {name: value or None for name, value in zip(names, values, strict=True)}


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "AggregateCell",
    "AggregateError",
    "AggregateSpec",
    "AggregateTrial",
    "NOT_ESTIMABLE",
    "Phase12Aggregate",
    "ValidatedRun",
    "aggregate_phase12",
]
