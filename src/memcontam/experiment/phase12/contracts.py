from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


RouteCandidateId = Literal["3w", "5w"]
ExperimentalArm = Literal["clean", "correct", "irrelevant", "contam", "filter"]
ProtocolIndex = Literal["clean", "contam", "filter"]
RagMode = Literal["frozen", "online_ext", "online_self", "not_applicable"]
FidelityLabel = Literal["negative_control", "source_aligned", "adapted", "style_proxy", "bounded"]
ToolMode = Literal["text_only", "python_sandbox"]
EvidenceLayer = Literal["build", "calibration", "main", "extension"]
RunFamily = Literal[
    "readiness",
    "pilot_a",
    "pilot_b",
    "behavioral",
    "main_a",
    "main_b",
    "main_c",
    "sequential",
    "extension",
    "exploratory_code",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class PrefixExecutionKey(_StrictModel):
    kind: Literal["branch_free_prefix"]


class MemoryArmExecutionKey(_StrictModel):
    kind: Literal["memory_arm"]
    arm: ExperimentalArm


class NoMemExecutionKey(_StrictModel):
    kind: Literal["nomem_singleton"]
    key: Literal["*"]


ExecutionKey = Annotated[
    PrefixExecutionKey | MemoryArmExecutionKey | NoMemExecutionKey, Field(discriminator="kind")
]


class BaseSensitivityCellSpec(_StrictModel):
    kind: Literal["base"]
    cell_id: str


class TimingSensitivityCellSpec(_StrictModel):
    kind: Literal["timing"]
    cell_id: str
    base_cell_id: str
    timing_quantile: Literal["early", "base", "late"]


class HorizonSensitivityCellSpec(_StrictModel):
    kind: Literal["horizon"]
    cell_id: str
    base_cell_id: str
    horizon: int = Field(gt=0)


class AffinitySensitivityCellSpec(_StrictModel):
    kind: Literal["affinity"]
    cell_id: str
    base_cell_id: str
    affinity_band: str


class FhBudgetSensitivityCellSpec(_StrictModel):
    kind: Literal["fh_budget"]
    cell_id: str
    base_cell_id: str
    fh_context_budget_id: str


class EmbeddingSensitivityCellSpec(_StrictModel):
    kind: Literal["embedding"]
    cell_id: str
    base_cell_id: str
    embedding_contract_id: str


class BehavioralSensitivityCellSpec(_StrictModel):
    kind: Literal["behavior"]
    cell_id: str
    base_cell_id: str
    behavior_test_id: str


SensitivityCellSpec = Annotated[
    BaseSensitivityCellSpec
    | TimingSensitivityCellSpec
    | HorizonSensitivityCellSpec
    | AffinitySensitivityCellSpec
    | FhBudgetSensitivityCellSpec
    | EmbeddingSensitivityCellSpec
    | BehavioralSensitivityCellSpec,
    Field(discriminator="kind"),
]


class BaselineConditionSpec(_StrictModel):
    condition_id: str
    baseline_family: Literal["nomem", "full_history", "rag", "bot", "reflexion"]
    fidelity_label: FidelityLabel
    rag_mode: RagMode
    fh_mode: str
    execution_key: NoMemExecutionKey | None = None
    execution_key_example: MemoryArmExecutionKey | None = None

    @model_validator(mode="after")
    def _validate_primary_condition(self) -> BaselineConditionSpec:
        if self.baseline_family == "nomem":
            if (
                not isinstance(self.execution_key, NoMemExecutionKey)
                or self.execution_key_example is not None
            ):
                raise ValueError("NOMEM_ARM_FORBIDDEN")
        elif self.execution_key is not None or not isinstance(
            self.execution_key_example, MemoryArmExecutionKey
        ):
            raise ValueError("INVALID_RUN_EXECUTION_KEY")
        if self.baseline_family == "rag" and self.rag_mode != "frozen":
            raise ValueError("UNSUPPORTED_PRIMARY_CONDITION")
        if self.baseline_family != "rag" and self.rag_mode != "not_applicable":
            raise ValueError("UNSUPPORTED_PRIMARY_CONDITION")
        return self


class PrefixTemplateSpec(_StrictModel):
    prefix_template_key: str
    execution_key: PrefixExecutionKey
    model_snapshot: str
    evidence_layer: EvidenceLayer
    task_family: str
    baseline_condition_id: str
    sensitivity_cell_ref: Mapping[str, str | int]
    prompt_version: str
    tool_contract_hash: str
    corpus_version: str
    capacity_contract_id: str
    artifact_hash: str


class RunTemplateSpec(_StrictModel):
    run_template_id: str
    layer: Literal["core", "sensitivity", "replication", "extension"]
    population_layer: Literal["core", "extension"]
    run_family: RunFamily
    analysis_status: Literal[
        "primary", "robustness", "exploratory_model_specificity", "confirmatory_extension"
    ]
    model_snapshot: str
    evidence_layer: EvidenceLayer
    task_family: str
    baseline_condition_id: str
    execution_key: MemoryArmExecutionKey | NoMemExecutionKey
    sensitivity_cell_ref: Mapping[str, str | int]
    contamination_type: Literal["core", "chi_rule", "chi_trace", "not_applicable"]
    horizon: int = Field(gt=0)
    prefix_template_key_or_none: str | None
    candidate_and_control_ids: tuple[str, ...]
    corpus_index_filter_versions: Mapping[str, str]
    prompt_version: str
    tool_contract_hash: str
    artifact_hash: str


class CandidateTemplateSet(_StrictModel):
    template_set_id: str
    source_config_id: str
    template_package_hash: str
    candidate_route: RouteCandidateId
    repository_commit: str
    authoritative_experiment_design_sha256: str
    behavior_test_registry_id: str
    behavior_test_registry_hash: str
    inv03_equivalence_registry_id: str
    inv03_equivalence_registry_hash: str
    run_templates: tuple[RunTemplateSpec, ...]
    prefix_templates: tuple[PrefixTemplateSpec, ...]
    abstract_slots: tuple[str, ...]
    artifact_hash: str


class BehaviorTestRow(_StrictModel):
    test_id: str
    test_class: Literal["MFT", "INV", "DIR"]
    capability_id: str
    task_families: tuple[str, ...]
    applicable_baseline_mode_conditions: tuple[str, ...]
    source_run_family: str
    transformation_generator_version: str
    base_template_count: int
    generated_variant_count: int
    fixed_size_rule: str
    candidate_and_control_ids: tuple[str, ...]
    expected_relation: str
    metric_and_aggregation_unit: str
    equivalence_tolerance_or_interval_rule: str
    case_level_and_aggregate_failure_rule: str
    duplicate_exclusion_rule: str
    unnatural_or_invalid_case_exclusion_rule: str
    human_or_mechanical_validation_rule: str
    evidence_status: Literal["pilot_behavioral", "implementation_gating", "secondary"]
    representative_failure_selection_rule: str
    source_path: str
    auxiliary_contract_refs: Mapping[str, str]
    row_hash: str


class BehaviorTestRegistry(_StrictModel):
    registry_id: str
    schema_version: str
    required_test_ids: tuple[str, ...]
    rows: tuple[BehaviorTestRow, ...]
    source_experiment_design_sha256: str
    frozen_after_pilot_b: Literal[True]
    artifact_hash: str


class Inv03EquivalenceContract(_StrictModel):
    contract_id: str
    task_family: str
    baseline_condition_id: Literal["rag_frozen"]
    arm: Literal["clean", "contam", "filter"]
    reference_cell_id: Literal["base"]
    variant_cell_id: Literal["behavior-inv03"]
    reference_artifact_id: str
    variant_artifact_id: str
    canonical_content_id: str
    id_correspondence: Mapping[str, str]
    renderer_content_hash: str
    embedding_input_hash: str
    embedding_vector_hash: str
    query_transform_hash: str
    filter_input_hash: str
    verifier_input_hash: str
    reference_index_id: str
    reference_index_hash: str
    variant_index_id: str
    variant_index_hash: str
    deterministic_ranking_contract_hash: str
    filter_admission_input_hash_or_none: str | None
    filter_decision_hash_or_none: str | None
    mechanical_gate_status: Literal["pass"]
    artifact_hash: str


class Inv03EquivalenceRegistry(_StrictModel):
    registry_id: str
    required_contract_keys: tuple[str, ...]
    contracts: tuple[Inv03EquivalenceContract, ...]
    frozen_before_route_selection: Literal[True]
    artifact_hash: str


class RegistryRefs(_StrictModel):
    behavior_test_registry_id: str
    behavior_test_registry_hash: str
    inv03_equivalence_registry_id: str
    inv03_equivalence_registry_hash: str
    call_cost_registry_id: str
    call_cost_registry_hash: str
    conditional_call_scope_registry_id: str
    conditional_call_scope_registry_hash: str
    pilot_call_statistics_manifest_id: str
    pilot_call_statistics_manifest_hash: str
    conservative_rate_upper_bound_registry_id: str
    conservative_rate_upper_bound_registry_hash: str
    conditional_call_rate_registry_id: str
    conditional_call_rate_registry_hash: str


class PilotBManifest(_StrictModel):
    manifest_id: str
    artifact_hash: str
    completed_before_main_unblinding: Literal[True]
    attempted_seed_counts: Mapping[str, int]
    cost_registry_hash: str
    conditional_call_scope_registry_id: str
    conditional_call_scope_registry_hash: str
    pilot_call_statistics_manifest_id: str
    pilot_call_statistics_manifest_hash: str
    conservative_rate_upper_bound_registry_id: str
    conservative_rate_upper_bound_registry_hash: str
    conditional_call_rate_registry_id: str
    conditional_call_rate_registry_hash: str
    joint_eligibility_summary_hash: str
    variance_summary_hash: str
    frozen_at: str


class MftManifest(_StrictModel):
    manifest_id: str
    artifact_hash: str
    all_registered_cases_attempted: Literal[True]
    mft04_status: Literal["pass", "fail"]
    route_gate_status: Literal["pass", "blocked"]
    case_status_ledger_hash: str
    pilot_allowance_ledger_hash: str
    frozen_at: str


class CodeMatrixPlan(_StrictModel):
    plan_id: str
    artifact_hash: str
    exploratory_run_template_registry_id: str
    exploratory_run_template_registry_hash: str
    abstract_slots: tuple[str, ...]
    estimated_exploratory_calls: int = Field(ge=0)


class RouteFeasibilityReport(_StrictModel):
    report_id: str
    candidate_route: RouteCandidateId
    run_template_registry_id: str
    run_template_registry_hash: str
    requested_core_counts: Mapping[str, int | Literal["not_feasible"]]
    requested_extension_counts: Mapping[str, int | Literal["not_feasible"]]
    estimated_calls: int = Field(ge=0)
    call_budget_breakdown_id: str
    call_budget_breakdown_hash: str
    call_capacity: int = Field(ge=0)
    feasible: bool
    reasons: tuple[str, ...]
    pilot_b_manifest_id: str
    pilot_b_manifest_hash: str
    mft_manifest_id: str
    mft_manifest_hash: str
    artifact_hash: str


class SeedAllocationManifest(_StrictModel):
    manifest_id: str
    selected_route: RouteCandidateId
    selected_feasibility_report_id: str
    selected_feasibility_report_hash: str
    run_template_registry_id: str
    run_template_registry_hash: str
    requested_core_counts: Mapping[str, int]
    requested_extension_counts: Mapping[str, int]
    slot_to_seed: Mapping[str, int]
    approved_by: str
    frozen_at: str
    artifact_hash: str


class RouteSelectionManifest(_StrictModel):
    manifest_id: str
    selected_route: RouteCandidateId
    feasibility_report_ids: tuple[str, str]
    selected_feasibility_report_id: str
    selected_feasibility_report_hash: str
    seed_allocation_manifest_id: str
    seed_allocation_manifest_hash: str
    selected_after_pilot_b: Literal[True]
    mft_gate_status: Literal["pass"]
    approved_by: str
    frozen_at: str
    artifact_hash: str


class SelectedPackageResourceManifest(_StrictModel):
    manifest_id: str
    route_selection_manifest_id: str
    route_selection_manifest_hash: str
    seed_allocation_manifest_id: str
    seed_allocation_manifest_hash: str
    exploratory_plan_id: str
    exploratory_plan_hash: str
    estimated_exploratory_calls: int = Field(ge=0)
    mandatory_package_status: Literal["fully_resourced"]
    remaining_call_capacity: int = Field(ge=0)
    exploratory_call_budget: int = Field(ge=0)
    reproducibility_reserve: int = Field(ge=0)
    approved_by: str
    frozen_at: str
    artifact_hash: str


class ExploratoryActivationManifest(_StrictModel):
    manifest_id: str
    route_selection_manifest_id: str
    route_selection_manifest_hash: str
    seed_allocation_manifest_id: str
    seed_allocation_manifest_hash: str
    resource_manifest_id: str
    resource_manifest_hash: str
    exploratory_plan_id: str
    exploratory_plan_hash: str
    exploratory_run_template_registry_id: str
    exploratory_run_template_registry_hash: str
    exploratory_slot_to_seed: Mapping[str, int]
    approved_by: str
    frozen_at: str
    artifact_hash: str


class ValidatedRouteSelection(_StrictModel):
    validation_hash: str
    selected_route: RouteCandidateId
    route_selection_manifest_id: str
    route_selection_manifest_hash: str
    seed_allocation_manifest_id: str
    seed_allocation_manifest_hash: str
    selected_feasibility_report_id: str
    selected_feasibility_report_hash: str
    run_template_registry_id: str
    run_template_registry_hash: str
    pilot_b_manifest_id: str
    pilot_b_manifest_hash: str
    mft_manifest_id: str
    mft_manifest_hash: str
    slot_to_seed: Mapping[str, int]


class ValidatedExploratoryActivation(_StrictModel):
    validation_hash: str
    exploratory_activation_manifest_id: str
    exploratory_activation_manifest_hash: str
    resource_manifest_id: str
    resource_manifest_hash: str
    exploratory_plan_id: str
    exploratory_plan_hash: str
    exploratory_run_template_registry_id: str
    exploratory_run_template_registry_hash: str
    exploratory_slot_to_seed: Mapping[str, int]
    route_selection_manifest_id: str
    route_selection_manifest_hash: str
    seed_allocation_manifest_id: str
    seed_allocation_manifest_hash: str
    estimated_exploratory_calls: int
    exploratory_call_budget: int
    reproducibility_reserve: int
    remaining_call_capacity: int


RouteGovernanceArtifact = (
    PilotBManifest
    | MftManifest
    | RouteFeasibilityReport
    | SeedAllocationManifest
    | RouteSelectionManifest
)
ExploratoryGovernanceArtifact = (
    CodeMatrixPlan | SelectedPackageResourceManifest | ExploratoryActivationManifest
)


class RerunPolicySpec(_StrictModel):
    policy_id: str
    version: str
    rerunnable_failure_classes: tuple[str, ...]
    artifact_hash: str


class ToolPolicyRef(_StrictModel):
    tool_mode: ToolMode
    tool_contract_hash: str
    policy_version: str


class FidelityCertificate(_StrictModel):
    certificate_id: str
    protocol_version: str
    git_commit: str
    overall_status: Literal["pass", "blocked", "fail"]
    issued_at: str


class Phase12IntegrationCertificate(_StrictModel):
    certificate_id: str
    protocol_version: str
    git_commit: str
    resolved_config_hash: str
    public_artifact_manifest_hash: str
    bfv2_certificate_id: str
    bfv2_certificate_hash: str
    prefix_checkpoint_gate: str
    five_arm_branch_gate: str
    nomem_alias_gate: str
    filter_information_boundary_gate: str
    logging_v3_join_gate: str
    model_behavior_denominator_gate: str
    eligibility_recomputation_gate: str
    run_archive_reconstruction_gate: str
    overall_status: Literal["pass", "blocked", "fail"]
    issued_at: str


class MetricRegistry(_StrictModel):
    registry_id: str
    version: str
    metric_ids: tuple[str, ...]
    artifact_hash: str


class EmbeddingRuntimeContract(_StrictModel):
    contract_id: str
    model_id: str
    model_revision: str
    corpus_version: str
    artifact_hash: str
