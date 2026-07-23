from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest

from memcontam.evaluation.phase12_aggregate import AggregateTrial, ValidatedRun
from memcontam.experiment.phase12.contracts import ValidatedExploratoryActivation
from memcontam.logging.schema_v3 import (
    MemoryArmExecutionKey,
    NoMemExecutionKey,
    BaseSensitivityCellRef,
    ScientificExploratoryCodeRunMetadata,
    ScientificAdmissionReference,
    ToolEvent,
)


ROOT = Path(__file__).resolve().parents[1]
SLOT = "game24|exploratory|slot-001"


def _config(
    contract_path: Path = ROOT / "containers" / "python-sandbox" / "image.lock.json",
    abstract_slots: tuple[str, ...] = (SLOT,),
) -> dict[str, object]:
    return {
        "protocol_version": "phase12_code_exploratory_v1",
        "activation_status": "inactive",
        "task_family": "game24",
        "baseline_condition_ids": ("nomem", "bot_style", "dc_rs"),
        "exploratory_run_template_registry_id": "exploratory-registry-001",
        "exploratory_run_template_registry_hash": "exploratory-registry-hash",
        "abstract_slots": abstract_slots,
        "estimated_exploratory_calls": 12,
        "oci_contract_path": contract_path,
    }


def _activation(plan) -> ValidatedExploratoryActivation:
    return ValidatedExploratoryActivation(
        validation_hash="validated-activation-hash",
        exploratory_activation_manifest_id="activation-001",
        exploratory_activation_manifest_hash="activation-hash",
        resource_manifest_id="resource-001",
        resource_manifest_hash="resource-hash",
        exploratory_plan_id=plan.plan_id,
        exploratory_plan_hash=plan.artifact_hash,
        exploratory_run_template_registry_id=plan.exploratory_run_template_registry_id,
        exploratory_run_template_registry_hash=plan.exploratory_run_template_registry_hash,
        exploratory_slot_to_seed={
            slot: 7 + index for index, slot in enumerate(plan.abstract_slots)
        },
        route_selection_manifest_id="route-001",
        route_selection_manifest_hash="route-hash",
        seed_allocation_manifest_id="allocation-001",
        seed_allocation_manifest_hash="allocation-hash",
        estimated_exploratory_calls=plan.estimated_exploratory_calls,
        exploratory_call_budget=12,
        reproducibility_reserve=2,
        remaining_call_capacity=14,
    )


def _metadata(baseline: str, mode: str, plan, activation: ValidatedExploratoryActivation):
    execution_key = (
        NoMemExecutionKey(kind="nomem_singleton", key="*")
        if baseline == "nomem"
        else MemoryArmExecutionKey(kind="memory_arm", arm="clean")
    )
    return ScientificExploratoryCodeRunMetadata(
        metadata_kind="exploratory_code_scientific",
        protocol_version="phase12_code_exploratory_v1",
        scientific_result=True,
        evidence_layer="main",
        run_family="exploratory_code",
        run_template_id=f"{baseline}-{mode}-template",
        prefix_template_key_or_none=None if baseline == "nomem" else f"{baseline}-prefix",
        task_family="game24",
        baseline_condition_id=baseline,
        execution_key=execution_key,
        protocol_index_or_none="clean" if baseline != "nomem" else None,
        trajectory_seed=7,
        abstract_seed_slot_or_none=SLOT,
        sensitivity_cell_ref=BaseSensitivityCellRef(kind="base", cell_id="base"),
        metric_registry_version="metrics-v1",
        embedding_contract_hash="embedding-hash",
        tool_contract_hash=f"{mode}-tool-contract",
        candidate_registry_version="candidates-v1",
        split_manifest_version="split-v1",
        behavior_registry_version="behavior-v1",
        run_template_registry_version=plan.exploratory_run_template_registry_hash,
        rerun_policy_version="rerun-v1",
        scientific_admission_ref=ScientificAdmissionReference(p12i_certificate_id="p12i-001"),
        source_route_selection_manifest_id=activation.route_selection_manifest_id,
        source_seed_allocation_manifest_id=activation.seed_allocation_manifest_id,
        exploratory_activation_manifest_id=activation.exploratory_activation_manifest_id,
    )


def _tool_event(baseline: str) -> ToolEvent:
    return ToolEvent(
        record_type="tool_event",
        event_id=f"{baseline}:tool",
        run_id=f"{baseline}:code",
        trial_id=f"{baseline}:trial",
        event_seq=0,
        tool_mode="python_sandbox",
        action="execute_python",
        code_hash=f"{baseline}-code-hash",
        output="24\n",
        stderr="",
        exit_code=0,
        status="completed",
        duration_ms=1,
        executor_identity="oci-python-sandbox",
        parent_call_id=f"{baseline}:parent",
        continuation_call_id=f"{baseline}:continuation",
    )


def _run(baseline: str, mode: str, score: Literal[0, 1], plan, activation):
    CodeMatrixRun = importlib.import_module(
        "memcontam.experiment.phase12.code_matrix"
    ).CodeMatrixRun

    return CodeMatrixRun(
        run=ValidatedRun(
            _metadata(baseline, mode, plan, activation),
            (
                AggregateTrial(
                    trial_id=f"{baseline}:{mode}:trial",
                    verified_score=score,
                    analysis_inclusion="included",
                    execution_status="completed",
                ),
            ),
        ),
        tool_mode=mode,
        suffix_id=f"{baseline}:{SLOT}:suffix",
        artifact_id=f"{baseline}:{mode}:artifact",
        tool_events=(_tool_event(baseline),) if mode == "python_sandbox" else (),
    )


def test_plans_inactive_matrix_and_aggregates_activated_paired_seed() -> None:
    module = importlib.import_module("memcontam.experiment.phase12.code_matrix")

    plan = module.build_code_matrix(_config())
    activation = _activation(plan)
    runs = tuple(
        _run(baseline, mode, cast(Literal[0, 1], score), plan, activation)
        for baseline, scores in {
            "nomem": (0, 1),
            "bot_style": (0, 1),
            "dc_rs": (1, 0),
        }.items()
        for mode, score in zip(("text_only", "python_sandbox"), scores, strict=True)
    )

    aggregate = module.aggregate_code_matrix(runs, plan, activation)

    assert plan.plan_id.startswith("code-matrix-")
    assert plan == module.build_code_matrix(_config())
    assert plan.abstract_slots == (SLOT,)
    assert aggregate.activation_manifest_id == activation.exploratory_activation_manifest_id
    assert [
        (item.baseline_condition_id, item.mean_score_delta) for item in aggregate.diagnostics
    ] == [
        ("bot_style", 1.0),
        ("dc_rs", -1.0),
        ("nomem", 1.0),
    ]
    assert {
        item.baseline_condition_id: item.nomem_adjusted_mean_score_delta
        for item in aggregate.diagnostics
    } == {
        "bot_style": 0.0,
        "dc_rs": -2.0,
        "nomem": None,
    }
    assert {
        item.baseline_condition_id: item.tool_event_count for item in aggregate.diagnostics
    } == {
        "bot_style": 1,
        "dc_rs": 1,
        "nomem": 1,
    }

    incomplete_plan = module.build_code_matrix(
        _config(abstract_slots=(SLOT, "game24|exploratory|slot-002"))
    )
    incomplete_activation = _activation(incomplete_plan)
    with pytest.raises(module.CodeMatrixError, match="UNPAIRED_SUFFIX"):
        module.aggregate_code_matrix(runs, incomplete_plan, incomplete_activation)

    stale_run = replace(
        runs[0],
        run=replace(
            runs[0].run,
            metadata=runs[0].run.metadata.model_copy(
                update={"run_template_registry_version": "stale-registry"}
            ),
        ),
    )
    with pytest.raises(module.CodeMatrixError, match="STALE_EXPLORATORY_REGISTRY"):
        module.aggregate_code_matrix((stale_run, *runs[1:]), plan, activation)
