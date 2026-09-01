from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Task = Literal[
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
]
Baseline = Literal["fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"]
Arm = Literal["clean", "correct", "irrelevant", "contam"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactBinding(_FrozenModel):
    role: Identifier
    path: str
    sha256: Sha256


class AuthorityBinding(_FrozenModel):
    role: Identifier
    sha256: Sha256


class DispatchRealization(_FrozenModel):
    realization_id: Identifier
    task_order: tuple[Task, ...]
    concrete_seed_ids: tuple[int, ...]
    expanded_dispatch_sha256: Sha256

    @field_validator("task_order", mode="before")
    @classmethod
    def _task_tuple(cls, value: list[str]) -> tuple[str, ...]:
        return tuple(value)

    @field_validator("concrete_seed_ids", mode="before")
    @classmethod
    def _seed_tuple(cls, value: list[int]) -> tuple[int, ...]:
        return tuple(value)


class ArmSequence(_FrozenModel):
    sequence_index: Annotated[int, Field(ge=0, le=3)]
    arms: tuple[Arm, ...]

    @field_validator("arms", mode="before")
    @classmethod
    def _arms(cls, value: list[str]) -> tuple[str, ...]:
        return tuple(value)


class ArmOrderRealization(_FrozenModel):
    realization_id: Identifier
    sequences: tuple[ArmSequence, ...]
    seed_sequence_indices: tuple[int, ...]
    realization_sha256: Sha256

    @field_validator("sequences", "seed_sequence_indices", mode="before")
    @classmethod
    def _tuples(cls, value: list[dict[str, str | int | list[str]]] | list[int]) -> tuple:
        return tuple(value)


class ActiveCells(_FrozenModel):
    tasks: tuple[Task, ...]
    memory_baselines: tuple[Baseline, ...]
    arms: tuple[Arm, ...]
    included_task_baseline_pairs: tuple[tuple[Task, Baseline], ...]
    nomem_tasks: tuple[Task, ...]
    nomem_policy: Literal["singleton_per_task_seed"]
    memory_cell_count: Annotated[int, Field(gt=0)]
    nomem_cell_count: Annotated[int, Field(gt=0)]
    attempted_trajectory_count: Literal[970]

    @field_validator(
        "tasks",
        "memory_baselines",
        "arms",
        "included_task_baseline_pairs",
        "nomem_tasks",
        mode="before",
    )
    @classmethod
    def _tuples(cls, value: list) -> tuple:
        return tuple(tuple(item) if isinstance(item, list) else item for item in value)


class RuntimeBinding(_FrozenModel):
    provider_id: Literal["openai"]
    api: Literal["OpenAI Responses API"]
    model: Literal["gpt-5.6-luna"]
    service_tier: Literal["default"]
    reasoning_effort: Literal["none"]
    reasoning_context: Literal["current_turn"]
    previous_response_id: None
    store: Literal[False]
    tools: tuple[()]
    timeout_seconds: Literal[180]
    request_contract_id: Literal["phase13_openai_luna_provider_contract_v1"]
    session_contract_id: Literal["independent_per_trial_and_arm"]
    execution_envelope_registry_id: Literal["CORE_EXECUTION_ENVELOPE_REGISTRY_V2"]
    transport_contract_id: Literal["CORE_TRANSPORT_ATTEMPT_CONTRACT_V2"]
    terminal_failure_contract_id: Literal["CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"]
    maximum_transport_attempts: Literal[1]
    retries_after_initial_attempt: Literal[0]


class ObservabilityBinding(_FrozenModel):
    packet_id: Literal["OBSERVABILITY_REGISTRATION_PACKET_V1"]
    packet_sha256: Sha256
    failure_classifier_registry_id: Literal[
        "OBSERVABILITY_REGISTRATION_PACKET_V1.failure_classifier"
    ]
    failure_classifier_status: Literal["PACKET_BOUND_REGISTERED"]
    recurrence_lookback_h: Literal[10]
    exposure_conditioning: Literal["CURRENT_Z_T_EQUALS_1_PRIOR_MATCH_NEED_NOT_BE_EXPOSED"]
    post_eviction_rule: Literal[
        "EXACT_ROOT_PRESENT_EXPLICITLY_REMOVED_ABSENT_AFTER_NEXT_ORDINARY_ROW_FIRST_RISK"
    ]
    retention_rule: Literal["FIRST_CONTINUOUS_FINITE_WINDOW_EPISODE_NO_GAP_BRIDGING"]
    censoring_rule: Literal["RIGHT_CENSORED_AT_REGISTERED_OR_FIXTURE_ENDPOINT"]
    u_t_status: Literal["NOT_REGISTERED_FOR_CURRENT_MAIN"]
    production_reconstruction_binding_id: Literal[
        "phase13-production-observability-reconstruction-binding-v1"
    ]
    production_reconstruction_binding_sha256: Sha256


class CostGuardBinding(_FrozenModel):
    cost_envelope_id: Literal["COST_ENVELOPE_V2"]
    cost_envelope_sha256: Sha256
    semantic_calls: Literal[108930]
    total_budget_ceiling_krw: Literal[500000]
    reserve_krw: Literal[50000]
    core_authorization_gate_krw: Literal[450000]
    cmax_main_krw: Literal[444126]
    margin_krw: Literal[5874]


class ExecutionControlBinding(_FrozenModel):
    runner_id: Literal["phase13-main-a-runner-v1"]
    runner_code_sha256: Sha256
    ledger_schema_id: Literal["phase13-main-a-ledger-v1"]
    unit_identity_law_id: Literal["phase13-main-a-disjoint-unit-id-v1"]
    nomem_realization_law_id: Literal["singleton_per_task_seed"]
    dispatch_contract_id: Literal["phase13-main-a-at-most-once-no-replay-v1"]
    planned_pause_contract_id: Literal["phase13-main-a-pre-intent-tranche-pause-v1"]
    resume_contract_id: Literal["phase13-main-a-validated-resume-v1"]
    inflight_reconciliation_contract_id: Literal["phase13-main-a-evidence-only-reconcile-v1"]
    quota_terminal_contract_id: Literal["CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"]


class MainExecutionFreeze(_FrozenModel):
    schema_version: Literal["phase13_main_execution_freeze_v1"]
    package_id: Literal["phase13-main-a-execution-freeze-v1"]
    status: Literal["FROZEN"]
    selected_package_id: Literal["phase13_main_a_post_cutoff_partial_crossed_v1"]
    mr_p4_status: Literal["CLOSED"]
    mr_p4_closure_hash: Sha256
    authority: tuple[AuthorityBinding, ...]
    artifacts: tuple[ArtifactBinding, ...]
    dispatch: DispatchRealization
    arm_order: ArmOrderRealization
    active_cells: ActiveCells
    H_run: Literal[50]
    H_primary: Literal[50]
    primary_analysis_window_id: Literal["core_prefix_50"]
    capacity_law_id: Literal["luna_common_visible_memory_capacity_v1"]
    capacity_unit: Literal["registered_serialized_tokens"]
    B_mem_tokens: Literal[8192]
    L_DC_tokens: Literal[8192]
    level2_registry_status: Literal["MATERIALIZED"]
    level2_registry_sha256: Sha256
    runtime: RuntimeBinding
    observability: ObservabilityBinding
    cost_guard: CostGuardBinding
    execution_control: ExecutionControlBinding
    measured_main_a_trajectory_count: Literal[0]
    package_hash: Sha256

    @field_validator("authority", "artifacts", mode="before")
    @classmethod
    def _bindings(cls, value: list[dict[str, str]]) -> tuple[dict[str, str], ...]:
        return tuple(value)


class AuthorizedExecution(_FrozenModel):
    schema_version: Literal["phase13_main_authorization_v1"]
    authorization_id: Literal["phase13-main-a-authorized-execution-v1"]
    status: Literal["AUTHORIZED_EXECUTION"]
    execution_package_id: Literal["phase13-main-a-execution-freeze-v1"]
    execution_package_path: str
    execution_package_sha256: Sha256
    execution_package_hash: Sha256
    mr_p4_status: Literal["CLOSED"]
    mr_p5_status: Literal["FROZEN"]
    mr_p6_status: Literal["PASS"]
    main_a_status: Literal["NOT_STARTED"]
    measured_main_a_trajectory_count: Literal[0]
    authorization_hash: Sha256


class MainExecutionFreezeReport(_FrozenModel):
    package_id: str
    package_sha256: Sha256
    package_hash: Sha256
    status: Literal["FROZEN"]
    mr_p4_status: Literal["CLOSED"]
    mr_p5_status: Literal["FROZEN"]
    measured_main_a_trajectory_count: Literal[0]


class MainAuthorizationReport(_FrozenModel):
    authorization_id: str
    authorization_sha256: Sha256
    authorization_hash: Sha256
    execution_package_sha256: Sha256
    status: Literal["AUTHORIZED_EXECUTION"]
    mr_p6_status: Literal["PASS"]
    main_a_status: Literal["NOT_STARTED"]
    measured_main_a_trajectory_count: Literal[0]
