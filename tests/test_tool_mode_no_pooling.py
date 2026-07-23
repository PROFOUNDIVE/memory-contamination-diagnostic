from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from memcontam.evaluation.phase12_aggregate import AggregateTrial, ValidatedRun
from memcontam.experiment.phase12.contracts import ValidatedExploratoryActivation
from memcontam.logging.schema_v3 import (
    BaseSensitivityCellRef,
    NoMemExecutionKey,
    NonScientificExploratoryCodeRunMetadata,
    ScientificExploratoryCodeRunMetadata,
)


ROOT = Path(__file__).resolve().parents[1]
SLOT = "game24|exploratory|slot-001"


def _config(contract_path: Path = ROOT / "containers" / "python-sandbox" / "image.lock.json") -> dict[str, object]:
    return {
        "protocol_version": "phase12_code_exploratory_v1",
        "activation_status": "inactive",
        "task_family": "game24",
        "baseline_condition_ids": ("nomem", "bot_style", "dc_rs"),
        "exploratory_run_template_registry_id": "exploratory-registry-001",
        "exploratory_run_template_registry_hash": "exploratory-registry-hash",
        "abstract_slots": (SLOT,),
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
        exploratory_slot_to_seed={SLOT: 7},
        route_selection_manifest_id="route-001",
        route_selection_manifest_hash="route-hash",
        seed_allocation_manifest_id="allocation-001",
        seed_allocation_manifest_hash="allocation-hash",
        estimated_exploratory_calls=plan.estimated_exploratory_calls,
        exploratory_call_budget=12,
        reproducibility_reserve=2,
        remaining_call_capacity=14,
    )


def _run(mode: str, plan, activation, suffix_id: str = "suffix-001"):
    CodeMatrixRun = importlib.import_module("memcontam.experiment.phase12.code_matrix").CodeMatrixRun

    metadata = ScientificExploratoryCodeRunMetadata(
        metadata_kind="exploratory_code_scientific",
        protocol_version="phase12_code_exploratory_v1",
        scientific_result=True,
        evidence_layer="main",
        run_family="exploratory_code",
        run_template_id=f"nomem-{mode}-template",
        prefix_template_key_or_none=None,
        task_family="game24",
        baseline_condition_id="nomem",
        execution_key=NoMemExecutionKey(kind="nomem_singleton", key="*"),
        protocol_index_or_none=None,
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
        scientific_admission_ref={"p12i_certificate_id": "p12i-001"},
        source_route_selection_manifest_id=activation.route_selection_manifest_id,
        source_seed_allocation_manifest_id=activation.seed_allocation_manifest_id,
        exploratory_activation_manifest_id=activation.exploratory_activation_manifest_id,
    )
    return CodeMatrixRun(
        run=ValidatedRun(
            metadata,
            (
                AggregateTrial(
                    trial_id=f"nomem:{mode}:trial",
                    verified_score=1,
                    analysis_inclusion="included",
                    execution_status="completed",
                ),
            ),
        ),
        tool_mode=mode,
        suffix_id=suffix_id,
        artifact_id=f"nomem:{mode}:artifact",
    )


def test_rejects_cross_tool_superiority_claim_unpaired_suffix_and_unactivated_scientific_run(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("memcontam.experiment.phase12.code_matrix")

    plan = module.build_code_matrix(_config())
    activation = _activation(plan)
    paired = (_run("text_only", plan, activation), _run("python_sandbox", plan, activation))

    with pytest.raises(module.CodeMatrixError, match="CROSS_TOOL_SUPERIORITY_CLAIM_FORBIDDEN"):
        module.aggregate_code_matrix(paired, plan, activation, claim_kind="superiority")
    with pytest.raises(module.CodeMatrixError, match="UNPAIRED_SUFFIX"):
        module.aggregate_code_matrix(
            (paired[0], _run("python_sandbox", plan, activation, suffix_id="other-suffix")),
            plan,
            activation,
        )
    with pytest.raises(module.CodeMatrixError, match="EXPLORATORY_ACTIVATION_REQUIRED"):
        module.aggregate_code_matrix(paired, plan, None)
    with pytest.raises(module.CodeMatrixError, match="OCI_CONTRACT_UNAVAILABLE"):
        module.build_code_matrix(_config(tmp_path / "missing-image.lock.json"))

    cases = (
        (
            activation.model_copy(update={"resource_manifest_id": ""}),
            "EXPLORATORY_RESOURCE_RESERVATION_REQUIRED",
        ),
        (activation.model_copy(update={"exploratory_plan_hash": "stale"}), "STALE_EXPLORATORY_PLAN"),
        (
            activation.model_copy(update={"exploratory_run_template_registry_hash": "stale"}),
            "STALE_EXPLORATORY_REGISTRY",
        ),
        (
            activation.model_copy(update={"exploratory_call_budget": 11}),
            "EXPLORATORY_BUDGET_INSUFFICIENT",
        ),
        (
            activation.model_copy(update={"remaining_call_capacity": 13}),
            "REPRODUCIBILITY_RESERVE_INSUFFICIENT",
        ),
        (
            activation.model_copy(update={"exploratory_slot_to_seed": {SLOT: 8}}),
            "EXPLORATORY_SEED_ASSIGNMENT_MISMATCH",
        ),
    )
    for invalid_activation, code in cases:
        with pytest.raises(module.CodeMatrixError, match=code):
            module.aggregate_code_matrix(paired, plan, invalid_activation)

    payload = paired[0].run.metadata.model_dump(mode="python")
    payload.update(
        {
            "metadata_kind": "exploratory_code_non_scientific",
            "scientific_result": False,
            "scientific_admission_ref_or_none": None,
            "source_route_selection_manifest_id": None,
            "source_seed_allocation_manifest_id": None,
            "exploratory_activation_manifest_id": None,
        }
    )
    payload.pop("scientific_admission_ref")
    unactivated = paired[0].__class__(
        run=ValidatedRun(
            NonScientificExploratoryCodeRunMetadata.model_validate(payload), paired[0].run.trials
        ),
        tool_mode=paired[0].tool_mode,
        suffix_id=paired[0].suffix_id,
        artifact_id=paired[0].artifact_id,
    )
    with pytest.raises(module.CodeMatrixError, match="EXPLORATORY_ACTIVATION_REQUIRED"):
        module.aggregate_code_matrix((unactivated,), plan, None)
