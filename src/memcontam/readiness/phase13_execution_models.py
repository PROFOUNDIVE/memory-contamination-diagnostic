from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Task = Literal["game24", "math_equation_balancer", "word_sorting"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactRef(StrictModel):
    path: str
    sha256: Sha256


class TimingContract(StrictModel):
    L_min: PositiveInt
    tau_star_rule: Literal["minimum_static_feasible_position"]
    tau_star: PositiveInt
    H_run: PositiveInt
    absolute_trial_start: PositiveInt
    absolute_trial_end: PositiveInt
    event_time_start: NonNegativeInt
    event_time_end: NonNegativeInt
    minimum_stream_length: PositiveInt


class SuffixIdentity(StrictModel):
    seed_id: Annotated[int, Field(ge=10000, le=10011)]
    source_ordered_stream_sha256: Sha256
    suffix_length: Literal[10]


class TaskStream(StrictModel):
    task: Task
    calibration_path: str
    calibration_sha256: Sha256
    source_main_v1_path: str
    source_main_v1_sha256: Sha256
    prospective_main_v2_ordered_signatures_sha256: Sha256
    suffixes: tuple[SuffixIdentity, ...]

    @field_validator("suffixes", mode="before")
    @classmethod
    def _suffix_tuple(cls, value: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(value)


class ExecutionIdentities(StrictModel):
    provider_id: Identifier
    model_snapshot_id: Identifier
    decoding_contract_id: Identifier
    prompt_contract_id: Identifier
    tool_contract_id: Identifier
    parser_contract_id: Identifier
    verifier_contract_id: Identifier
    resource_contract_id: Identifier
    session_contract_id: Identifier
    failure_contract_id: Identifier
    retry_contract_id: Identifier
    checkpoint_serializer_id: Identifier
    task_stream_contract_id: Identifier
    native_capacity_registry_id: Identifier


class MemoryArm(StrictModel):
    arm_key: Literal["Clean", "Correct", "Irrelevant", "Contam"]
    branch_constructor_id: Identifier
    candidate_or_control_id: Identifier


class NoMemArm(StrictModel):
    arm_key: Literal["star_NoMem"]
    persistent_memory: Literal[False]
    singleton: Literal[True]


class NativeCapacity(StrictModel):
    baseline: Literal["fh_bounded", "rag_frozen", "bot_style", "reflexion_style"]
    capacity_contract_id: Identifier
    transition_policy: Literal[
        "oldest_first_truncation",
        "frozen_corpus_no_online_admission",
        "bounded_template_eviction",
        "bounded_reflection_eviction",
    ]
    configured_limit: PositiveInt
    insertion_supported: Literal[True]


class CallComponent(StrictModel):
    component_id: Identifier
    owner_kind: Literal["prefix", "execution"]
    owner_id: Identifier
    phase: Literal["burn_init", "trial"]
    nominal_calls_per_activation: NonNegativeInt
    raw_maximum_calls_per_activation: NonNegativeInt


class AnalysisWindow(StrictModel):
    analysis_window_id: Identifier
    source_execution_contract_id: Identifier
    window_length: Literal[2, 5, 10]
    event_time_start: Literal[0]
    event_time_end: Annotated[int, Field(ge=1, le=9)]
    outcome_family: Literal["verified_accuracy", "recurrence", "persistence", "propagation", "collapse_like"]
    evidence_status: Literal[
        "confirmatory_primary", "confirmatory_secondary", "prespecified_sensitivity",
        "descriptive", "exploratory",
    ]
    multiplicity_status: Literal[
        "primary_holm_family", "estimation_only", "descriptive_no_inferential_family"
    ]
    realization_disposition: Literal["prefix_view", "source_execution", "not_exchangeable"]
    provider_execution_multiplicity: Literal[0, 1]


class BlockedCapacity(StrictModel):
    status: Literal["blocked_pending_operator_authorization"]
    value: None
    unit: Identifier
    reason_code: Literal["OPERATOR_AUTHORIZATION_REQUIRED"]


class OperatorCapacity(StrictModel):
    maximum_cost_microusd: BlockedCapacity
    per_request_timeout_seconds: BlockedCapacity
    maximum_latency_milliseconds: BlockedCapacity
    maximum_archive_bytes: BlockedCapacity
    provider_requests_per_minute: BlockedCapacity
    provider_concurrency: BlockedCapacity
    maximum_wall_clock_seconds: BlockedCapacity
    retention_days: BlockedCapacity
    sealing_deadline_seconds: BlockedCapacity


class CallIllustration(StrictModel):
    task_seed_count: PositiveInt
    nominal_semantic_calls: PositiveInt
    raw_maximum_semantic_calls: PositiveInt
    reserved_semantic_calls: PositiveInt
    raw_maximum_transport_attempts: PositiveInt
    reserved_transport_attempts: PositiveInt
    maximum_input_tokens: PositiveInt
    maximum_output_tokens: PositiveInt


class PlanningIllustrations(StrictModel):
    reserve_percent: Literal[5]
    maximum_transport_attempts_per_semantic_call: Literal[4]
    maximum_input_tokens_per_transport_attempt: Literal[4096]
    maximum_output_tokens_per_transport_attempt: Literal[2048]
    main: CallIllustration
    calibration: CallIllustration


class ExecutionRegistry(StrictModel):
    schema_version: Literal["phase13_execution_registry_v1"]
    registry_id: Literal["phase13-execution-registry-v1"]
    source_partition: ArtifactRef
    timing: TimingContract
    execution_contract_id: Literal["phase13-main-a-h10-execution-v1"]
    checkpoint_registry_id: Literal["phase13-common-checkpoint-v1"]
    execution_suffix_registry_id: Literal["phase13-calibration-v2-suffix-v1"]
    analysis_window_registry_id: Literal["phase13-analysis-window-registry-v1"]
    primary_analysis_window_id: Identifier
    identities: ExecutionIdentities
    task_streams: tuple[TaskStream, ...]
    memory_arms: tuple[MemoryArm, ...]
    nomem: NoMemArm
    native_capacities: tuple[NativeCapacity, ...]
    prefix_owner_id: Identifier
    execution_owner_id: Identifier
    call_components: tuple[CallComponent, ...]
    analysis_windows: tuple[AnalysisWindow, ...]
    operator_capacity: OperatorCapacity
    planning_illustrations: PlanningIllustrations
    registry_hash: Sha256

    @field_validator(
        "task_streams", "memory_arms", "native_capacities", "call_components",
        "analysis_windows", mode="before",
    )
    @classmethod
    def _tuple_fields(cls, value: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(value)
