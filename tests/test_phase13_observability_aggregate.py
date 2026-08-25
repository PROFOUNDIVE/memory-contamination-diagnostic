from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def _module() -> ModuleType:
    return importlib.import_module("memcontam.evaluation.phase13_observability")


def test_aggregates_four_arms_and_nomem_over_exactly_ten_attempted_seeds() -> None:
    module = _module()
    rows = tuple(
        module.Phase13AggregateTrial(
            evidence_scope="synthetic_contract_fixture",
            task="game24",
            baseline="rag_frozen",
            arm=arm,
            trajectory_seed=seed,
            concrete_seed_id=f"game24-seed-{seed}",
            analysis_window_id="core_prefix_50",
            source_trial_count=50,
            structural_support=True,
            verified_outcome=1 if arm != "contam" else 0,
        )
        for seed in range(10)
        for arm in ("clean", "correct", "irrelevant", "contam")
    ) + tuple(
        module.Phase13AggregateTrial(
            evidence_scope="synthetic_contract_fixture",
            task="game24",
            baseline="nomem",
            arm="nomem",
            trajectory_seed=seed,
            concrete_seed_id=f"game24-seed-{seed}",
            analysis_window_id="core_prefix_50",
            source_trial_count=50,
            structural_support=True,
            verified_outcome=1,
        )
        for seed in range(10)
    )

    aggregate = module.aggregate_phase13(rows)

    rag = next(cell for cell in aggregate.cells if cell.baseline == "rag_frozen")
    nomem = next(cell for cell in aggregate.cells if cell.baseline == "nomem")
    assert rag.attempted_seed_count == 10
    assert aggregate.evidence_scope == "synthetic_contract_fixture"
    assert {arm: metric.value for arm, metric in rag.verified_accuracy_by_arm.items()} == {
        "clean": 1.0,
        "correct": 1.0,
        "irrelevant": 1.0,
        "contam": 0.0,
    }
    assert {name: metric.value for name, metric in rag.contrasts.items()} == {
        "clean_minus_contam": 1.0,
        "correct_minus_contam": 1.0,
        "irrelevant_minus_contam": 1.0,
    }
    assert nomem.observability_rates["nomem"]["theory_exposure"].status == "not_applicable"

    with pytest.raises(module.Phase13ObservabilityError, match="EXACTLY_TEN_ATTEMPTED_SEEDS_REQUIRED"):
        module.aggregate_phase13(rows[:-1])


def test_aggregate_rejects_unknown_tasks_and_task_local_seed_drift() -> None:
    module = _module()
    with pytest.raises(Exception):
        module.Phase13AggregateTrial(
            evidence_scope="synthetic_contract_fixture",
            task="not_registered",
            baseline="rag_frozen",
            arm="clean",
            trajectory_seed=0,
            concrete_seed_id="bad",
            analysis_window_id="core_prefix_50",
            source_trial_count=50,
            structural_support=True,
            verified_outcome=1,
        )

    rows = tuple(
        module.Phase13AggregateTrial(
            evidence_scope="synthetic_contract_fixture",
            task="game24",
            baseline=baseline,
            arm=arm,
            trajectory_seed=seed,
            concrete_seed_id=f"{baseline}-seed-{seed}",
            analysis_window_id="core_prefix_50",
            source_trial_count=50,
            structural_support=True,
            verified_outcome=1,
        )
        for baseline in ("rag_frozen", "bot_style")
        for seed in range(10)
        for arm in ("clean", "correct", "irrelevant", "contam")
    )
    with pytest.raises(module.Phase13ObservabilityError, match="TASK_LOCAL_SEED_IDENTITY_MISMATCH"):
        module.aggregate_phase13(rows)

    disjoint_ranks = tuple(
        row.model_copy(
            update={
                "trajectory_seed": row.trajectory_seed + 10,
                "concrete_seed_id": f"game24-seed-{row.trajectory_seed + 10}",
            }
        )
        for row in rows
        if row.baseline == "bot_style"
    ) + tuple(row for row in rows if row.baseline == "rag_frozen")
    with pytest.raises(module.Phase13ObservabilityError, match="TASK_LOCAL_SEED_RANK_MISMATCH"):
        module.aggregate_phase13(disjoint_ranks)


def test_aggregate_reports_not_estimable_without_structural_support() -> None:
    module = _module()
    rows = tuple(
        module.Phase13AggregateTrial(
            evidence_scope="synthetic_contract_fixture",
            task="game24",
            baseline="rag_frozen",
            arm=arm,
            trajectory_seed=seed,
            concrete_seed_id=f"game24-seed-{seed}",
            analysis_window_id="core_prefix_50",
            source_trial_count=50,
            structural_support=False,
            verified_outcome=1,
        )
        for seed in range(10)
        for arm in ("clean", "correct", "irrelevant", "contam")
    )

    cell = module.aggregate_phase13(rows).cells[0]

    assert cell.supported_seed_count_by_arm["contam"] == 0
    assert cell.verified_accuracy_by_arm["contam"].status == "not_estimable"
    assert cell.contrasts["clean_minus_contam"].status == "not_estimable"

    inconsistent = tuple(
        row.model_copy(update={"structural_support": row.arm != "contam"}) for row in rows
    )
    with pytest.raises(
        module.Phase13ObservabilityError, match="INCONSISTENT_BASELINE_STRUCTURAL_SUPPORT"
    ):
        module.aggregate_phase13(inconsistent)


def test_aggregate_rejects_duplicate_concrete_seed_identities() -> None:
    module = _module()
    rows = tuple(
        module.Phase13AggregateTrial(
            evidence_scope="synthetic_contract_fixture",
            task="game24",
            baseline="rag_frozen",
            arm=arm,
            trajectory_seed=seed,
            concrete_seed_id="same-seed",
            analysis_window_id="core_prefix_50",
            source_trial_count=50,
            structural_support=True,
            verified_outcome=1,
        )
        for seed in range(10)
        for arm in ("clean", "correct", "irrelevant", "contam")
    )

    with pytest.raises(module.Phase13ObservabilityError, match="DUPLICATE_CONCRETE_SEED_ID"):
        module.aggregate_phase13(rows)


def test_aggregate_excludes_noncontam_exposure_and_rejects_nonboolean_values() -> None:
    module = _module()
    supported = module.MetricValue(
        status="supported", value=True, reason="SYNTHETIC_CONTRACT_FIXTURE_ONLY"
    )
    rows = tuple(
        module.Phase13AggregateTrial(
            evidence_scope="synthetic_contract_fixture",
            task="game24",
            baseline="rag_frozen",
            arm=arm,
            trajectory_seed=seed,
            concrete_seed_id=f"game24-seed-{seed}",
            analysis_window_id="core_prefix_50",
            source_trial_count=50,
            structural_support=True,
            verified_outcome=1,
            theory_exposure=supported
            if arm == "clean"
            else module.MetricValue(status="unavailable", reason="NO_CONTAM_EVIDENCE"),
        )
        for seed in range(10)
        for arm in ("clean", "correct", "irrelevant", "contam")
    )

    cell = module.aggregate_phase13(rows).cells[0]

    assert cell.exposure_conditional_diagnostic.status == "not_estimable"

    invalid = tuple(
        row.model_copy(
            update={
                "theory_exposure": module.MetricValue(
                    status="supported", value="no", reason="INVALID_BOOLEAN"
                )
            }
        )
        if row.arm == "contam"
        else row
        for row in rows
    )
    with pytest.raises(module.Phase13ObservabilityError, match="NON_BOOLEAN_OBSERVABILITY_VALUE"):
        module.aggregate_phase13(invalid)
