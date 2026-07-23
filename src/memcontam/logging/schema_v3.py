from __future__ import annotations

from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from memcontam.phase12_types import RunFamily


LOGGING_V3: Literal["logging_v3"] = "logging_v3"
ProtocolIndex = Literal["clean", "contam", "filter"]
ExperimentalArm = Literal["clean", "correct", "irrelevant", "contam", "filter"]
ProtocolVersion = Literal["phase12_primary_v1", "phase12_code_exploratory_v1"]


class Phase12SchemaError(ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScientificAdmissionReference(_StrictModel):
    p12i_certificate_id: str
    bfv2_certificate_id: str | None = None
    readiness_bundle_hash: str | None = None

    @model_validator(mode="after")
    def _validate_reference(self) -> ScientificAdmissionReference:
        if not self.p12i_certificate_id:
            raise ValueError("SCIENTIFIC_ADMISSION_REQUIRED")
        return self


class PrefixExecutionKey(_StrictModel):
    kind: Literal["branch_free_prefix"]


class MemoryArmExecutionKey(_StrictModel):
    kind: Literal["memory_arm"]
    arm: ExperimentalArm


class NoMemExecutionKey(_StrictModel):
    kind: Literal["nomem_singleton"]
    key: Literal["*"]


ExecutionKey = Annotated[
    PrefixExecutionKey | MemoryArmExecutionKey | NoMemExecutionKey,
    Field(discriminator="kind"),
]


class BaseSensitivityCellRef(_StrictModel):
    kind: Literal["base"]
    cell_id: str


class TimingSensitivityCellRef(_StrictModel):
    kind: Literal["timing"]
    cell_id: str
    base_cell_id: str | None = None
    timing_quantile: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_unrelated_fields(cls, value: Any) -> Any:
        if isinstance(value, dict) and "horizon" in value:
            raise ValueError("UNRELATED_SENSITIVITY_FIELD")
        return value


class HorizonSensitivityCellRef(_StrictModel):
    kind: Literal["horizon"]
    cell_id: str
    base_cell_id: str | None = None
    horizon: int | None = Field(default=None, gt=0)


class AffinitySensitivityCellRef(_StrictModel):
    kind: Literal["affinity"]
    cell_id: str
    base_cell_id: str
    affinity_band: str


class FhBudgetSensitivityCellRef(_StrictModel):
    kind: Literal["fh_budget"]
    cell_id: str
    base_cell_id: str
    fh_context_budget_id: str


class EmbeddingSensitivityCellRef(_StrictModel):
    kind: Literal["embedding"]
    cell_id: str
    base_cell_id: str
    embedding_contract_id: str


class BehavioralSensitivityCellRef(_StrictModel):
    kind: Literal["behavior"]
    cell_id: str
    base_cell_id: str
    behavior_test_id: str


SensitivityCellRef = Annotated[
    BaseSensitivityCellRef
    | TimingSensitivityCellRef
    | HorizonSensitivityCellRef
    | AffinitySensitivityCellRef
    | FhBudgetSensitivityCellRef
    | EmbeddingSensitivityCellRef
    | BehavioralSensitivityCellRef,
    Field(discriminator="kind"),
]


def _validate_arm_projection(record: Any) -> None:
    execution_key = record.execution_key
    if not hasattr(record, "protocol_index_or_none"):
        return
    if isinstance(execution_key, MemoryArmExecutionKey):
        projection = (
            execution_key.arm if execution_key.arm in {"clean", "contam", "filter"} else None
        )
        if record.protocol_index_or_none != projection:
            if execution_key.arm == "clean" and record.protocol_index_or_none is None:
                raise Phase12SchemaError("NOMEM_ARM_FORBIDDEN")
            raise Phase12SchemaError("INVALID_ARM_PROTOCOL_PROJECTION")
    elif record.protocol_index_or_none is not None:
        raise Phase12SchemaError("INVALID_ARM_PROTOCOL_PROJECTION")


class _RunMetadataBase(_StrictModel):
    schema_version: Literal["logging_v3"] = LOGGING_V3
    contract_level: Literal["phase12"] = "phase12"
    protocol_version: ProtocolVersion
    evidence_layer: Literal["build", "calibration", "main", "extension"]
    run_family: RunFamily
    run_template_id: str
    prefix_template_key_or_none: str | None
    task_family: str
    baseline_condition_id: str
    execution_key: ExecutionKey
    protocol_index_or_none: ProtocolIndex | None
    trajectory_seed: int
    abstract_seed_slot_or_none: str | None
    sensitivity_cell_ref: SensitivityCellRef
    metric_registry_version: str
    embedding_contract_hash: str
    tool_contract_hash: str
    candidate_registry_version: str
    split_manifest_version: str
    behavior_registry_version: str
    run_template_registry_version: str
    rerun_policy_version: str

    @model_validator(mode="after")
    def _validate_execution_key(self) -> _RunMetadataBase:
        _validate_arm_projection(self)
        if (
            isinstance(self.execution_key, PrefixExecutionKey)
            and not self.prefix_template_key_or_none
        ):
            raise ValueError("PREFIX_EXECUTION_KEY_REQUIRED")
        if (
            isinstance(self.execution_key, NoMemExecutionKey)
            and self.prefix_template_key_or_none is not None
        ):
            raise ValueError("NOMEM_ARM_FORBIDDEN")
        return self


class PreRouteRunMetadata(_RunMetadataBase):
    metadata_kind: Literal["pre_route"]
    scientific_result: bool
    scientific_admission_ref_or_none: ScientificAdmissionReference | None
    route_selection_manifest_id: str | None = None
    seed_allocation_manifest_id: str | None = None

    @model_validator(mode="after")
    def _validate_pre_route(self) -> PreRouteRunMetadata:
        if self.protocol_version != "phase12_primary_v1":
            raise ValueError("MIXED_PROTOCOL_VERSION")
        if self.route_selection_manifest_id is not None:
            raise ValueError("ROUTE_SELECTION_FORBIDDEN_PRE_ROUTE")
        if self.seed_allocation_manifest_id is not None:
            raise ValueError("SEED_ALLOCATION_FORBIDDEN_PRE_ROUTE")
        if self.scientific_result and self.scientific_admission_ref_or_none is None:
            raise ValueError("SCIENTIFIC_ADMISSION_REQUIRED")
        if not self.scientific_result and self.scientific_admission_ref_or_none is not None:
            raise ValueError("ADMISSION_EVIDENCE_FORBIDDEN")
        return self


class SelectedRouteRunMetadata(_RunMetadataBase):
    metadata_kind: Literal["selected_route"]
    scientific_result: Literal[True]
    scientific_admission_ref: ScientificAdmissionReference | None = None
    route_selection_manifest_id: str | None = None
    seed_allocation_manifest_id: str | None = None

    @model_validator(mode="after")
    def _validate_selected_route(self) -> SelectedRouteRunMetadata:
        if self.protocol_version != "phase12_primary_v1":
            raise ValueError("MIXED_PROTOCOL_VERSION")
        if self.scientific_admission_ref is None:
            raise ValueError("SCIENTIFIC_ADMISSION_REQUIRED")
        if not self.route_selection_manifest_id:
            raise ValueError("ROUTE_SELECTION_REQUIRED")
        if not self.seed_allocation_manifest_id:
            raise ValueError("SEED_ALLOCATION_REQUIRED")
        return self


class NonScientificExploratoryCodeRunMetadata(_RunMetadataBase):
    metadata_kind: Literal["exploratory_code_non_scientific"]
    scientific_result: Literal[False]
    scientific_admission_ref_or_none: ScientificAdmissionReference | None = None
    source_route_selection_manifest_id: str | None = None
    source_seed_allocation_manifest_id: str | None = None
    exploratory_activation_manifest_id: str | None = None

    @model_validator(mode="after")
    def _validate_non_scientific_exploratory(self) -> NonScientificExploratoryCodeRunMetadata:
        if self.protocol_version != "phase12_code_exploratory_v1":
            raise ValueError("MIXED_PROTOCOL_VERSION")
        if self.scientific_admission_ref_or_none is not None:
            raise ValueError("ADMISSION_EVIDENCE_FORBIDDEN")
        if self.source_route_selection_manifest_id is not None:
            raise ValueError("SOURCE_ROUTE_SELECTION_FORBIDDEN")
        if self.source_seed_allocation_manifest_id is not None:
            raise ValueError("SOURCE_SEED_ALLOCATION_FORBIDDEN")
        if self.exploratory_activation_manifest_id is not None:
            raise ValueError("EXPLORATORY_ACTIVATION_FORBIDDEN")
        return self


class ScientificExploratoryCodeRunMetadata(_RunMetadataBase):
    metadata_kind: Literal["exploratory_code_scientific"]
    scientific_result: Literal[True]
    scientific_admission_ref: ScientificAdmissionReference | None = None
    source_route_selection_manifest_id: str | None = None
    source_seed_allocation_manifest_id: str | None = None
    exploratory_activation_manifest_id: str | None = None

    @model_validator(mode="after")
    def _validate_scientific_exploratory(self) -> ScientificExploratoryCodeRunMetadata:
        if self.protocol_version != "phase12_code_exploratory_v1":
            raise ValueError("MIXED_PROTOCOL_VERSION")
        if self.scientific_admission_ref is None:
            raise ValueError("SCIENTIFIC_ADMISSION_REQUIRED")
        if not self.source_route_selection_manifest_id:
            raise ValueError("ROUTE_SELECTION_REQUIRED")
        if not self.source_seed_allocation_manifest_id:
            raise ValueError("SEED_ALLOCATION_REQUIRED")
        if not self.exploratory_activation_manifest_id:
            raise ValueError("EXPLORATORY_ACTIVATION_REQUIRED")
        return self


RunMetadataV3 = Annotated[
    PreRouteRunMetadata
    | SelectedRouteRunMetadata
    | NonScientificExploratoryCodeRunMetadata
    | ScientificExploratoryCodeRunMetadata,
    Field(discriminator="metadata_kind"),
]


class _TrialLogBase(_StrictModel):
    schema_version: Literal["logging_v3"] = LOGGING_V3
    contract_level: Literal["phase12"] = "phase12"
    absolute_trial_index: int = Field(ge=0)
    event_time: int | str
    parse_status: str
    execution_status: str
    failure_class: str | None
    analysis_inclusion: str
    inclusion_reason: str
    context_event_id_or_none: str | None
    retrieval_event_ids: list[str]
    tool_event_ids: list[str]
    auxiliary_context_inclusion_or_none: dict[str, object] | None
    operational_attribution_or_none: dict[str, object] | None


class PrefixTrialLog(_TrialLogBase):
    trial_kind: Literal["branch_free_prefix"]
    execution_key: PrefixExecutionKey
    prefix_run_id: str
    checkpoint_event_ids: list[str]
    admission_event_ids: list[str]
    memory_event_ids: list[str]

    @model_validator(mode="after")
    def _validate_prefix(self) -> PrefixTrialLog:
        _validate_arm_projection(self)
        return self


class MemoryBranchTrialLog(_TrialLogBase):
    trial_kind: Literal["memory_branch"]
    execution_key: MemoryArmExecutionKey
    branch_id: str
    prefix_run_id: str
    checkpoint_id: str
    checkpoint_index: int = Field(ge=0)
    candidate_triplet_id_or_none: str | None
    native_render_id_or_none: str | None
    intervention_event_id_or_none: str | None = None
    admission_event_ids: list[str]
    memory_event_ids: list[str]

    @model_validator(mode="after")
    def _validate_branch(self) -> MemoryBranchTrialLog:
        _validate_arm_projection(self)
        if self.execution_key.arm == "clean":
            if any(
                value is not None
                for value in (
                    self.candidate_triplet_id_or_none,
                    self.native_render_id_or_none,
                    self.intervention_event_id_or_none,
                )
            ):
                raise ValueError("INTERVENTION_FORBIDDEN_FOR_CLEAN")
        elif any(
            value is None
            for value in (
                self.candidate_triplet_id_or_none,
                self.native_render_id_or_none,
                self.intervention_event_id_or_none,
            )
        ):
            raise ValueError("INTERVENTION_REQUIRED")
        return self


class NoMemTrialLog(_TrialLogBase):
    trial_kind: Literal["nomem_singleton"]
    execution_key: NoMemExecutionKey

    @model_validator(mode="after")
    def _validate_nomem(self) -> NoMemTrialLog:
        _validate_arm_projection(self)
        return self


TrialLogV3 = Annotated[
    PrefixTrialLog | MemoryBranchTrialLog | NoMemTrialLog,
    Field(discriminator="trial_kind"),
]


class _EventBase(_StrictModel):
    schema_version: Literal["logging_v3"] = LOGGING_V3
    contract_level: Literal["phase12"] = "phase12"
    event_id: str
    run_id: str
    trial_id: str | None = None
    event_seq: int = Field(ge=0)


class ToolEvent(_EventBase):
    record_type: Literal["tool_event"]
    tool_mode: Literal["python_sandbox"]
    action: Literal["execute_python"]
    code_hash: str
    output: str
    stderr: str
    exit_code: int
    status: Literal["completed"]
    duration_ms: int = Field(ge=0)
    executor_identity: str
    parent_call_id: str
    continuation_call_id: str


class RetrievalEvent(_EventBase):
    record_type: Literal["retrieval_event"]
    retrieval_id: str
    query_hash: str
    retrieved_entry_ids: list[str]
    retrieved_scores: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_scores(self) -> RetrievalEvent:
        if self.retrieved_scores and len(self.retrieved_scores) != len(self.retrieved_entry_ids):
            raise ValueError("RETRIEVAL_SCORE_COUNT_MISMATCH")
        return self


class ContextEvent(_EventBase):
    record_type: Literal["context_event"]
    context_id: str
    final_entry_ids: list[str]
    removed_entry_ids: list[str] = Field(default_factory=list)


class AdmissionEvent(_EventBase):
    record_type: Literal["admission_event"]
    admission_id: str
    decision: Literal["admit", "quarantine", "reject"]


class InterventionEvent(_EventBase):
    record_type: Literal["intervention_event"]
    intervention_id: str
    arm: ExperimentalArm
    candidate_triplet_id: str
    native_render_id: str


class CheckpointEvent(_EventBase):
    record_type: Literal["checkpoint_event"]
    checkpoint_id: str
    checkpoint_index: int = Field(ge=0)
    memory_hash: str


class EligibilityEvent(_EventBase):
    record_type: Literal["eligibility_event"]
    eligibility_id: str
    eligible: bool


class FailureEventV3(_EventBase):
    record_type: Literal["failure_event"]
    failure_class: str


V3Event = Annotated[
    ToolEvent
    | RetrievalEvent
    | ContextEvent
    | AdmissionEvent
    | InterventionEvent
    | CheckpointEvent
    | EligibilityEvent
    | FailureEventV3,
    Field(discriminator="record_type"),
]
Phase12Record = RunMetadataV3 | TrialLogV3 | V3Event

_RUN_METADATA_ADAPTER: TypeAdapter[Any] = TypeAdapter(RunMetadataV3)
_TRIAL_LOG_ADAPTER: TypeAdapter[Any] = TypeAdapter(TrialLogV3)
_EVENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(V3Event)
_EVENT_TYPES = {
    "tool_event",
    "retrieval_event",
    "context_event",
    "admission_event",
    "intervention_event",
    "checkpoint_event",
    "eligibility_event",
    "failure_event",
}


def parse_log_record_v3(record: Mapping[str, Any]) -> Phase12Record:
    payload = dict(record)
    if (
        payload.get("schema_version", LOGGING_V3) != LOGGING_V3
        or payload.get("contract_level", "phase12") != "phase12"
    ):
        raise Phase12SchemaError("SCHEMA_CONTRACT_MISMATCH")
    if "metadata_kind" in payload:
        return _RUN_METADATA_ADAPTER.validate_python(payload)
    if "trial_kind" in payload:
        execution_kind = payload.get("execution_key", {}).get("kind")
        if payload["trial_kind"] == "branch_free_prefix" and execution_kind != "branch_free_prefix":
            raise Phase12SchemaError("PREFIX_EXECUTION_KEY_REQUIRED")
        if payload["trial_kind"] == "nomem_singleton" and execution_kind != "nomem_singleton":
            raise Phase12SchemaError("NOMEM_ARM_FORBIDDEN")
        return _TRIAL_LOG_ADAPTER.validate_python(payload)
    if "record_type" in payload:
        if payload["record_type"] not in _EVENT_TYPES:
            raise Phase12SchemaError("UNKNOWN_V3_RECORD_TYPE")
        return _EVENT_ADAPTER.validate_python(payload)
    raise Phase12SchemaError("UNKNOWN_V3_RECORD_TYPE")
