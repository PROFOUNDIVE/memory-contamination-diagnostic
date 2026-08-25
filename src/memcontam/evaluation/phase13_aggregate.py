from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import mean

from memcontam.evaluation.phase13_observability_models import (
    AggregateBaseline,
    MetricValue,
    Phase13Aggregate,
    Phase13AggregateCell,
    Phase13AggregateTrial,
    Phase13ObservabilityError,
    Task,
)
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY


_ARMS = frozenset(CORE_MAIN_REGISTRY.arms)


def aggregate_phase13(rows: Sequence[Phase13AggregateTrial]) -> Phase13Aggregate:
    grouped: dict[tuple[Task, AggregateBaseline], list[Phase13AggregateTrial]] = defaultdict(list)
    task_seed_ids: dict[Task, dict[int, str]] = defaultdict(dict)
    for row in rows:
        if (row.task, row.baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells:
            raise Phase13ObservabilityError("EXCLUDED_CURRENT_MAIN_CELL")
        grouped[(row.task, row.baseline)].append(row)
        prior = task_seed_ids[row.task].setdefault(row.trajectory_seed, row.concrete_seed_id)
        if prior != row.concrete_seed_id:
            raise Phase13ObservabilityError("TASK_LOCAL_SEED_IDENTITY_MISMATCH")
    if any(len(set(seed_map.values())) != len(seed_map) for seed_map in task_seed_ids.values()):
        raise Phase13ObservabilityError("DUPLICATE_CONCRETE_SEED_ID")
    if any(set(seed_map) != set(range(10)) for seed_map in task_seed_ids.values()):
        raise Phase13ObservabilityError("TASK_LOCAL_SEED_RANK_MISMATCH")
    if not grouped:
        raise Phase13ObservabilityError("EMPTY_AGGREGATE")
    return Phase13Aggregate(
        evidence_scope="synthetic_contract_fixture",
        cells=tuple(
            _aggregate_cell(task, baseline, cell_rows)
            for (task, baseline), cell_rows in sorted(grouped.items())
        )
    )


def _aggregate_cell(
    task: Task, baseline: AggregateBaseline, rows: Sequence[Phase13AggregateTrial]
) -> Phase13AggregateCell:
    by_seed: dict[int, dict[str, Phase13AggregateTrial]] = defaultdict(dict)
    for row in rows:
        if row.arm in by_seed[row.trajectory_seed]:
            raise Phase13ObservabilityError("DUPLICATE_SEED_ARM")
        by_seed[row.trajectory_seed][row.arm] = row
    if len(by_seed) != CORE_MAIN_REGISTRY.attempted_seed_count:
        raise Phase13ObservabilityError("EXACTLY_TEN_ATTEMPTED_SEEDS_REQUIRED")
    expected = {"nomem"} if baseline == "nomem" else _ARMS
    if any(set(seed_rows) != expected for seed_rows in by_seed.values()):
        raise Phase13ObservabilityError("INCOMPLETE_PHASE13_ARM_BLOCK")
    if baseline != "nomem" and any(
        len({row.structural_support for row in seed_rows.values()}) != 1
        for seed_rows in by_seed.values()
    ):
        raise Phase13ObservabilityError("INCONSISTENT_BASELINE_STRUCTURAL_SUPPORT")
    supported = {
        arm: tuple(
            row
            for seed_rows in by_seed.values()
            if (row := seed_rows[arm]).structural_support or baseline == "nomem"
        )
        for arm in sorted(expected)
    }
    accuracy = {arm: _score(rows) for arm, rows in supported.items()}
    contrasts = {} if baseline == "nomem" else {
        f"{arm}_minus_contam": _contrast(by_seed, arm, "contam")
        for arm in ("clean", "correct", "irrelevant")
    }
    flat = tuple(row for seed_rows in by_seed.values() for row in seed_rows.values())
    return Phase13AggregateCell(
        task=task,
        baseline=baseline,
        attempted_seed_count=10,
        supported_seed_count_by_arm={arm: len(values) for arm, values in supported.items()},
        verified_accuracy_by_arm=accuracy,
        contrasts=contrasts,
        observability_rates={
            arm: {
                    name: _optional_rate(
                        tuple(row for row in flat if row.arm == arm), name, baseline, arm
                    )
                for name in (
                    "target_present_in_store_before_answer", "target_retrieved",
                    "target_final_context_included", "theory_exposure",
                )
            }
            for arm in sorted(expected)
        },
        exposure_conditional_diagnostic=_exposure_diagnostic(flat),
    )


def _score(rows: Sequence[Phase13AggregateTrial]) -> MetricValue:
    if not rows:
        return MetricValue(status="not_estimable", reason="NO_STRUCTURALLY_SUPPORTED_SEEDS")
    return MetricValue(status="supported", value=mean(row.verified_outcome for row in rows), reason="REALIZED_STRUCTURAL_SUPPORT")


def _contrast(
    by_seed: dict[int, dict[str, Phase13AggregateTrial]], left: str, right: str
) -> MetricValue:
    values = tuple(
        seed_rows[left].verified_outcome - seed_rows[right].verified_outcome
        for seed_rows in by_seed.values()
        if seed_rows[left].structural_support and seed_rows[right].structural_support
    )
    if not values:
        return MetricValue(status="not_estimable", reason="NO_PAIRWISE_COMMON_STRUCTURAL_SUPPORT")
    return MetricValue(
        status="supported",
        value=mean(values),
        reason="BASELINE_SPECIFIC_STRUCTURAL_SUPPORT",
    )


def _optional_rate(
    rows: Sequence[Phase13AggregateTrial],
    field: str,
    baseline: AggregateBaseline,
    arm: str,
) -> MetricValue:
    if baseline == "nomem" or arm != "contam":
        return MetricValue(status="not_applicable", reason="ARM_HAS_NO_TARGET_CONTAMINATION")
    values = tuple(getattr(row, field) for row in rows)
    if not values or any(value.status != "supported" for value in values):
        return MetricValue(status="unavailable", reason="TRIAL_OBSERVABILITY_UNAVAILABLE")
    if any(not isinstance(value.value, bool) for value in values):
        raise Phase13ObservabilityError("NON_BOOLEAN_OBSERVABILITY_VALUE")
    return MetricValue(
        status="supported",
        value=mean(bool(value.value) for value in values),
        reason="SYNTHETIC_CONTRACT_FIXTURE_ONLY",
    )


def _exposure_diagnostic(rows: Sequence[Phase13AggregateTrial]) -> MetricValue:
    exposed = tuple(
        row.verified_outcome
        for row in rows
        if row.arm == "contam"
        and row.structural_support
        and row.theory_exposure.status == "supported"
        and row.theory_exposure.value is True
    )
    unexposed = tuple(
        row.verified_outcome
        for row in rows
        if row.arm == "contam"
        and row.structural_support
        and row.theory_exposure.status == "supported"
        and row.theory_exposure.value is False
    )
    if not exposed or not unexposed:
        return MetricValue(
            status="not_estimable",
            reason="EXPOSURE_CONDITIONAL_SUPPORT_MISSING",
        )
    return MetricValue(
        status="supported",
        value=mean(exposed) - mean(unexposed),
        reason="CONDITIONAL_DIAGNOSTIC_NOT_CAUSAL_EFFECT",
    )


__all__ = ["aggregate_phase13"]
