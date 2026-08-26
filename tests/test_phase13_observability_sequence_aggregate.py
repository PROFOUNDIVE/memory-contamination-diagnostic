from __future__ import annotations

from typing import Literal

from memcontam.evaluation.phase13_aggregate import aggregate_phase13
from memcontam.evaluation.phase13_observability_models import (
    AggregateArm,
    MetricStatus,
    MetricValue,
    Phase13AggregateTrial,
)


SEQUENTIAL_FIELDS = (
    "generic_recurrence",
    "exact_lineage_recurrence",
    "exposure_conditioned_recurrence",
    "post_eviction_recurrence",
    "root_storage_persistence",
    "descendant_storage_persistence",
    "root_prompt_visibility",
    "descendant_prompt_visibility",
    "root_retention_duration",
    "prompt_retention_duration",
    "descendant_retention_duration",
    "propagation",
)


def _metric(
    status: MetricStatus, value: float | None = None, reason: str = "fixture"
) -> MetricValue:
    return MetricValue(status=status, value=value, reason=reason)


def _trial(seed: int, arm: AggregateArm, value: float) -> Phase13AggregateTrial:
    verified_outcome: Literal[0, 1] = 0 if seed % 2 == 0 else 1
    target_metric = (
        _metric("supported", value)
        if arm == "contam"
        else _metric("not_applicable", reason="TARGET_CONTAMINATION_SCOPE_NOT_APPLICABLE")
    )
    return Phase13AggregateTrial(
        evidence_scope="synthetic_contract_fixture",
        task="game24",
        baseline="bot_style" if arm != "nomem" else "nomem",
        arm=arm,
        trajectory_seed=seed,
        concrete_seed_id=f"fixture-game24-seed-{seed}",
        analysis_window_id="core_prefix_50",
        source_trial_count=50,
        structural_support=True,
        verified_outcome=verified_outcome,
        generic_recurrence=_metric("supported", value),
        **{field: target_metric for field in SEQUENTIAL_FIELDS[1:]},
    )


def test_aggregate_computes_supported_sequential_means_by_arm() -> None:
    trials = tuple(
        _trial(seed, arm, float(seed % 2))
        for seed in range(10)
        for arm in ("contam", "clean", "correct", "irrelevant", "nomem")
    )

    aggregate = aggregate_phase13(trials)
    cell = next(item for item in aggregate.cells if item.baseline == "bot_style")

    assert cell.observability_rates["contam"]["generic_recurrence"].value == 0.5
    assert cell.observability_rates["clean"]["generic_recurrence"].value == 0.5
    for field in SEQUENTIAL_FIELDS[1:]:
        assert cell.observability_rates["contam"][field].value == 0.5
        assert cell.observability_rates["clean"][field].status == "not_applicable"


def test_aggregate_preserves_uniform_explicit_status() -> None:
    unavailable = _metric(
        "unavailable", reason="PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
    trials = tuple(
        _trial(seed, arm, 0.0).model_copy(update={"generic_recurrence": unavailable})
        for seed in range(10)
        for arm in ("contam", "clean", "correct", "irrelevant")
    )

    cell = aggregate_phase13(trials).cells[0]

    assert cell.observability_rates["contam"]["generic_recurrence"] == unavailable
