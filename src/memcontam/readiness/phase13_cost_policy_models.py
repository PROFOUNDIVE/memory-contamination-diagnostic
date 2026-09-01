from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ActivationStatus = Literal["PENDING_CONTROLLED_EXTERNAL_AUTHORITY_WRITE"]
PolicyClassification = Literal["PROSPECTIVE_EXECUTION_POLICY_CHANGE_EXPLICITLY_APPROVED"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactIdentity(_FrozenModel):
    path: str = Field(min_length=1)
    sha256: Sha256


class Capacity(_FrozenModel):
    writer_max_output_tokens: Literal[8192]
    B_DC_feasible: Literal[8192]
    B_mem_tokens: Literal[8192]
    L_DC_tokens: Literal[8192]


class StageEnvelope(_FrozenModel):
    semantic_stage_id: str = Field(min_length=1)
    authority_stage_id: str = Field(min_length=1)
    suffix_calls: int = Field(ge=0)
    prefix_calls: int = Field(ge=0)
    calls: int = Field(gt=0)
    maximum_input_tokens: int = Field(gt=0)
    maximum_output_tokens: int = Field(gt=0)


class StageEnvelopeRegistry(_FrozenModel):
    schema_version: Literal["phase13_stage_envelope_registry_v1"]
    registry_id: Literal["CORE_EXECUTION_ENVELOPE_REGISTRY_V2"]
    authority_sha256: Literal[
        "4dec48f105c8d4730706d1d99d05bb14bab96a8e643811db1ebdd26e612590d5"
    ]
    activation_status: ActivationStatus
    policy_classification: PolicyClassification
    request_compiler_id: Literal["phase13_runtime_registry_method_stage_v1"]
    serializer_id: Literal["role_content_blank_line_v1"]
    tokenizer_id: Literal["tiktoken_o200k_base"]
    counting_implementation: Literal[
        "src/memcontam/baselines/prompt_budget.py:count_prompt_tokens"
    ]
    capacity: Capacity
    stages: tuple[StageEnvelope, ...]
    registry_hash: Sha256


class TerminalFailure(_FrozenModel):
    error_type: Literal["ProviderCallFailure"]
    failure_disposition: Literal["provider_call_failed"]
    scientific_ineligibility_reason: Literal["provider_call_failed"]
    failed_trial_retained: Literal[True]
    subsequent_trial_dispatch: Literal["prohibited"]
    missingness_propagation: Literal["failed_trial_and_unexecuted_suffix_remain_missing"]
    seed_or_cell_replacement: Literal["prohibited"]
    unregistered_rerun: Literal["prohibited"]
    imputation: Literal["prohibited"]


class RetryFailureContract(_FrozenModel):
    schema_version: Literal["phase13_retry_failure_contract_v1"]
    contract_id: Literal["CORE_TRANSPORT_ATTEMPT_CONTRACT_V2"]
    terminal_failure_contract_id: Literal["CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"]
    terminal_failure_contract_sha256: Literal[
        "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
    ]
    activation_status: ActivationStatus
    policy_classification: PolicyClassification
    maximum_transport_attempts_per_semantic_call: Literal[1]
    retries_after_initial_attempt: Literal[0]
    retry_trigger_classes: tuple[()]
    same_semantic_call_retry: Literal[False]
    semantic_invalid_generic_retry: Literal[False]
    terminal_failure: TerminalFailure
    input_envelope_violation: Literal["terminal_technical_failure_before_provider_dispatch"]
    contract_hash: Sha256


class RateCard(_FrozenModel):
    input_usd_per_million: Literal["0.20"]
    cached_input_usd_per_million: Literal["0.02"]
    output_usd_per_million: Literal["1.20"]
    cache_write_planning_premium: Literal["1.25"]
    cache_read_credit: Literal["none"]
    long_context_threshold_tokens: Literal[272000]
    long_context_input_multiplier: Literal["2.0"]
    long_context_output_multiplier: Literal["1.5"]
    fx_planning_ceiling_krw_per_usd: Literal[1600]


class Budget(_FrozenModel):
    total_budget_ceiling_krw: Literal[500000]
    reserve_fraction: Literal["0.10"]
    core_authorization_gate_krw: Literal[450000]


class StageCost(_FrozenModel):
    semantic_stage_id: str = Field(min_length=1)
    input_krw_ceiling: int = Field(ge=0)
    output_krw_ceiling: int = Field(ge=0)
    stage_krw: int = Field(ge=0)


class CleanPrefixReconciliation(_FrozenModel):
    status: Literal["CLOSED"]
    case: Literal["B"]
    authorization: Literal["EXPLICIT_USER_AUTHORIZATION_2026-08-31"]
    main_outcome_blind: Literal[True]
    prefix_ownership_instances: Literal[230]
    prefix_semantic_calls: Literal[430]
    suffix_semantic_calls: Literal[108500]


class CostProof(_FrozenModel):
    schema_version: Literal["cost_envelope_v2_cost_proof_v1"]
    proof_id: Literal["cost_envelope_v2_main_a_450k_v1"]
    activation_status: ActivationStatus
    package_id: Literal["phase13_main_a_post_cutoff_partial_crossed_v1"]
    package_selection_path: str = Field(min_length=1)
    package_selection_sha256: Sha256
    common_capacity_path: str = Field(min_length=1)
    common_capacity_sha256: Sha256
    cost_envelope_id: Literal["COST_ENVELOPE_V2"]
    cost_envelope_sha256: Literal[
        "6de377752cd80e45147a8b47aa83828f2921363b564c44004ac90650dac65cf2"
    ]
    cost_envelope_path: Literal[
        "data/phase13/main/cost_envelope_v2/corrected_cost_envelope_v2.txt"
    ]
    stage_envelope_registry_id: Literal["CORE_EXECUTION_ENVELOPE_REGISTRY_V2"]
    stage_envelope_registry_authority_sha256: Literal[
        "4dec48f105c8d4730706d1d99d05bb14bab96a8e643811db1ebdd26e612590d5"
    ]
    stage_envelope_registry_hash: Sha256
    retry_failure_contract_id: Literal["CORE_TRANSPORT_ATTEMPT_CONTRACT_V2"]
    retry_failure_contract_hash: Sha256
    terminal_failure_contract_id: Literal["CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"]
    terminal_failure_contract_sha256: Literal[
        "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
    ]
    reconciliation: CleanPrefixReconciliation
    semantic_calls: Literal[108930]
    rate_card: RateCard
    budget: Budget
    rounding: Literal["ceil_each_stage_input_and_output_krw_component_then_sum"]
    stage_costs: tuple[StageCost, ...]
    cmax_main_krw: Literal[444126]
    margin_to_core_gate_krw: Literal[5874]
    proof_hash: Sha256


class Approval(_FrozenModel):
    decision_classification: PolicyClassification
    session_id: Literal["ses_fc2c0fee1ffeCtvJ45A7qgIbis"]
    main_outcome_blind: Literal[True]


class AuthoritySource(_FrozenModel):
    filename: str = Field(min_length=1)
    source_sha256: Sha256


class CandidateState(_FrozenModel):
    canonical_authority_updated: Literal[False]
    reference_reviewer_completed: Literal[False]
    router_synchronized: Literal[False]
    repository_authority_projections_activated: Literal[False]
    mr_p4_resumed_or_reclosed: Literal[False]
    mr_p5_started: Literal[False]
    mr_p6_started: Literal[False]
    main_execution_authorized: Literal[False]
    main_a_measured_scientific_execution_count: Literal[0]


class CandidateManifest(_FrozenModel):
    schema_version: Literal["phase13_cost_policy_candidate_manifest_v1"]
    manifest_id: Literal["phase13_main_a_cost_envelope_v2_450k_candidate_v1"]
    policy_id: Literal["phase13_main_a_cost_envelope_v2_450k_w8192_a512_s384_t1"]
    activation_status: ActivationStatus
    approval: Approval
    artifacts: dict[str, ArtifactIdentity]
    controlled_external_write_sources: dict[str, AuthoritySource]
    state: CandidateState
    manifest_hash: Sha256


class CostPolicyValidationReport(_FrozenModel):
    policy_id: str
    activation_status: str
    total_budget_ceiling_krw: int
    reserve_fraction: str
    core_authorization_gate_krw: int
    cmax_main_krw: int
    margin_krw: int
    writer_cap: int
    common_capacity_tokens: int
    maximum_transport_attempts: int
    execution_envelope_registry_id: str
    execution_envelope_registry_sha256: Sha256
    terminal_failure_contract_sha256: Sha256
    cost_envelope_sha256: Sha256
    manifest_sha256: Sha256
