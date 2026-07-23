from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.memory.stores import MemoryEntry


LOGGING_V1 = "logging_v1"
LOGGING_V2 = "logging_v2"
LOGGING_V3 = "logging_v3"

BadMemoryUptakeLabel = Literal[
    "not_applicable", "not_evaluable", "no_uptake_detected", "uptake_detected"
]
RepeatedFailureLabel = Literal["not_applicable", "first_failure", "repeated_failure"]
RecoveryAfterFilterLabel = Literal["not_applicable", "recovered", "not_recovered"]
ExposureStatus = Literal["supported", "not_applicable", "not_evaluable"]
ExposureMode = Literal["clean", "final_prompt", "not_in_final_prompt", "not_evaluable"]
EvaluationRegime = Literal["online", "frozen"]
MemoryUpdateMode = Literal["enabled", "disabled", "not_applicable"]
ContaminationClass = Literal["clean", "injected", "derived", "natural"]
LineageStatus = Literal["exact", "approximate", "unavailable"]
LineageBasis = Literal[
    "seed", "recorded_parent", "recorded_source", "version_edge", "signature", "none"
]


class EvaluationLawSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_law_id: str
    regime: EvaluationRegime
    task_law_id: str
    inference_law_id: str
    checkpoint_policy_id: str | None = None


class TargetContaminationSetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_set_id: str
    definition_version: str
    included_classes: list[ContaminationClass]
    require_exact_lineage: bool


class CheckpointPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    policy_id: str | None = None
    interval: int | None = Field(default=None, gt=0)
    artifact_root: str | None = None


class CheckpointRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    checkpoint_trial_index: int = Field(ge=0)
    checkpoint_memory_hash: str
    checkpoint_source_run_id: str
    artifact_path: str | None = None


class LineageEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    child_entry_id: str
    parent_entry_id: str
    relation: str
    lineage_status: LineageStatus
    lineage_basis: LineageBasis
    injected_root_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_lineage_basis(self) -> LineageEdge:
        if self.lineage_status == "exact" and self.lineage_basis == "signature":
            raise ValueError("signature basis cannot claim exact lineage")
        return self


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_metadata_id: str
    run_id: str
    git_commit: str
    config_hash: str
    provider: str
    model_snapshots: dict[str, str]
    query_date: str
    start_date: str
    seed: int | str | None
    order: int | str | list[str]
    decoding_defaults: dict[str, Any]
    sample_set_hash: str
    sample_order_hash: str
    stage: str
    schema_version: str
    contract_level: Literal["phase10", "phase11"] = "phase10"
    evaluation_law: EvaluationLawSpec | None = None
    target_contamination_set: TargetContaminationSetSpec | None = None
    checkpoint_policy: CheckpointPolicySpec = Field(default_factory=CheckpointPolicySpec)
    prompt_version: str
    memory_policy_version: str
    contamination_catalog_version: str
    retry_policy_version: str

    @model_validator(mode="after")
    def _validate_logging_contract(self) -> RunMetadata:
        if self.schema_version == LOGGING_V1:
            if self.contract_level != "phase10":
                raise ValueError("logging_v1 requires contract_level=phase10")
            return self
        if self.schema_version == LOGGING_V2:
            if self.contract_level != "phase11":
                raise ValueError("logging_v2 requires contract_level=phase11")
            if self.evaluation_law is None:
                raise ValueError("logging_v2 requires evaluation_law")
            if self.target_contamination_set is None:
                raise ValueError("logging_v2 requires target_contamination_set")
            return self
        raise ValueError(f"schema_version must be {LOGGING_V1} or {LOGGING_V2}")


class EventContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_metadata_id: str
    run_id: str
    trial_id: str
    trial_seq: int = Field(ge=0)
    event_seq: int = Field(ge=0)
    stage: str


class PromptSourceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    rendered_hash: str
    entry_id: str
    parent_call_id: str | None = None
    source_ids: list[str]
    parent_ids: list[str]
    lineage_id: str
    version: str
    origin: str
    clean_or_contaminated: Literal["clean", "contaminated"]
    contamination_class: ContaminationClass | None = None
    injected_root_ids: list[str] = Field(default_factory=list)
    lineage_status: LineageStatus | None = None
    lineage_basis: LineageBasis | None = None
    direct_parent_ids: list[str] = Field(default_factory=list)
    target_set_id: str | None = None
    is_target_contamination: bool | None = None

    @model_validator(mode="after")
    def _require_nonempty_span(self) -> PromptSourceSpan:
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        _validate_phase11_lineage_fields(self)
        _validate_contamination_compatibility(
            self.clean_or_contaminated, self.contamination_class
        )
        return self


class CallEvent(EventContext):
    call_id: str
    method_stage: str = "unknown"
    messages: list[dict[str, str]]
    model: str
    decoding_params: dict[str, Any]
    response_text: str | None
    token_usage: dict[str, int]
    latency_ms: int | None = Field(strict=True, ge=0)
    retry_count: int = Field(ge=0)
    source_spans: list[PromptSourceSpan]
    created_at: str
    error_type: str | None = None
    failure_function: str | None = None
    failure_module: str | None = None
    failure_line: int | None = Field(default=None, ge=0)
    origin: Literal["provider_call", "parser", "verifier", "runner"] | None = None


class FailureEvent(EventContext):
    failure_id: str
    origin: Literal["provider_call", "parser", "verifier", "runner"]
    error_type: str
    failure_function: str | None
    failure_module: str | None
    failure_line: int | None = Field(default=None, ge=0)
    retry_count: int = Field(ge=0)
    disposition: str
    created_at: str


class FilterEvent(EventContext):
    filter_id: str
    arm: Literal["clean", "contaminated", "contaminated_filter"]
    baseline: str
    decisions: list[dict[str, Any]]
    kept_source_ids: list[str]
    removed_source_ids: list[str]
    pre_source_ids: list[str]
    post_source_ids: list[str]
    ground_truth_contaminated_ids: list[str]
    action: Literal["apply", "outcome"]
    final_answer_source_ids: list[str]
    verdict: str | None
    created_at: str


class MemoryEvent(EventContext):
    memory_id: str
    event_type: str
    operation: str
    baseline: str
    source_trial_id: str | None
    parent_entry_ids: list[str]
    source_entry_ids: list[str]
    contaminated_source_ids: list[str]
    before_entry_ids: list[str]
    after_entry_ids: list[str]
    before_snapshot_hash: str | None
    after_snapshot_hash: str | None
    new_entry_ids: list[str]
    updated_entry_ids: list[str]
    removed_entry_ids: list[str]
    creation_origin: str | None
    memory_version: str | None
    status: str
    created_at: str
    lineage_edges: list[LineageEdge] = Field(default_factory=list)


class MemoryItemLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str
    content_hash: str
    memory_type: str
    clean_or_contaminated: str
    source_trial_id: str | None
    parent_entry_ids: list[str]
    source_entry_ids: list[str]
    lineage_id: str
    version: str
    creation_origin: str
    metadata: dict[str, Any]
    contamination_class: ContaminationClass | None = None
    injected_root_ids: list[str] = Field(default_factory=list)
    lineage_status: LineageStatus | None = None
    lineage_basis: LineageBasis | None = None
    direct_parent_ids: list[str] = Field(default_factory=list)
    target_set_id: str | None = None
    is_target_contamination: bool | None = None

    @model_validator(mode="after")
    def _validate_phase11_lineage(self) -> MemoryItemLog:
        _validate_phase11_lineage_fields(self)
        _validate_contamination_compatibility(
            self.clean_or_contaminated, self.contamination_class
        )
        return self

    @classmethod
    def from_memory_entry(
        cls, entry: MemoryEntry, entries: list[MemoryEntry] | None = None
    ) -> MemoryItemLog:
        from memcontam.logging.provenance import canonical_lineage_for_entry, canonical_metadata

        metadata = canonical_metadata(entry.metadata)
        parent_entry_ids = _string_list(metadata.get("parent_entry_ids"))
        source_entry_ids = _string_list(metadata.get("source_entry_ids"))
        if not source_entry_ids:
            source_entry_ids = _string_list(metadata.get("source_contaminated_entry_ids"))

        lineage_id = metadata.get("lineage_id")
        if not isinstance(lineage_id, str) or not lineage_id:
            lineage = metadata.get("lineage")
            lineage_id = (
                lineage
                if isinstance(lineage, str) and lineage not in {"clean", "contaminated"}
                else entry.source_trial_id or entry.entry_id
            )

        version = metadata.get("memory_version", metadata.get("version", "v0"))
        creation_origin = metadata.get("creation_origin", metadata.get("origin"))
        if not isinstance(creation_origin, str) or not creation_origin:
            reflection_lineage = metadata.get("reflection_lineage")
            if isinstance(reflection_lineage, dict) and isinstance(reflection_lineage.get("stage"), str):
                creation_origin = reflection_lineage["stage"]
            else:
                creation_origin = "seed" if entry.source_trial_id is None else entry.memory_type

        lineage = canonical_lineage_for_entry(entry, entries)
        target_set_id = metadata.get("target_set_id")
        is_target_contamination = metadata.get("is_target_contamination")
        has_phase11_lineage = any(
            field_name in metadata
            for field_name in (
                "contamination_class",
                "lineage_status",
                "lineage_basis",
                "direct_parent_ids",
                "injected_root_ids",
                "memory_error_status",
            )
        )
        contamination_class = lineage.contamination_class
        if not has_phase11_lineage and entry.clean_or_contaminated == "contaminated":
            contamination_class = None
        canonical_binary_class = (
            entry.clean_or_contaminated
            if contamination_class is None
            else "clean" if contamination_class == "clean" else "contaminated"
        )
        return cls(
            entry_id=entry.entry_id,
            content_hash=hashlib.sha256(entry.content.encode("utf-8")).hexdigest(),
            memory_type=entry.memory_type,
            clean_or_contaminated=canonical_binary_class,
            source_trial_id=entry.source_trial_id,
            parent_entry_ids=parent_entry_ids,
            source_entry_ids=source_entry_ids,
            lineage_id=lineage_id,
            version=str(version),
            creation_origin=creation_origin,
            metadata=metadata,
            contamination_class=contamination_class,
            injected_root_ids=lineage.injected_root_ids if contamination_class is not None else [],
            lineage_status=lineage.lineage_status if contamination_class is not None else None,
            lineage_basis=lineage.lineage_basis if contamination_class is not None else None,
            direct_parent_ids=lineage.direct_parent_ids if contamination_class is not None else [],
            target_set_id=target_set_id if isinstance(target_set_id, str) and target_set_id else None,
            is_target_contamination=is_target_contamination
            if isinstance(is_target_contamination, bool)
            else None,
        )


def _string_list(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


class ContaminationExposure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: Literal["clean", "contaminated", "contaminated_filter"] = "clean"
    status: ExposureStatus = "not_applicable"
    is_exposed: bool | None = None
    answer_call_id: str | None = None
    target_entry_ids: list[str] = Field(default_factory=list)
    source_entry_ids: list[str] = Field(default_factory=list)
    exposed_source_ids: list[str] = Field(default_factory=list)
    exposure_mode: ExposureMode = "clean"
    reason: str = "clean arm has no contaminated memory sources"
    target_set_id: str | None = None
    exposed_entry_ids: list[str] = Field(default_factory=list)
    exposed_injected_root_ids: list[str] = Field(default_factory=list)
    evidence_lineage_status: LineageStatus | None = None

    @model_validator(mode="before")
    @classmethod
    def _downgrade_legacy_proxy_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        legacy_keys = {
            "contamination_types",
            "memory_before_entry_ids",
            "retrieved_entry_ids",
        }
        if legacy_keys.intersection(value) and "status" not in value:
            return {
                "condition": value.get("condition", "clean"),
                "status": "not_evaluable",
                "is_exposed": None,
                "answer_call_id": None,
                "target_entry_ids": [],
                "source_entry_ids": _string_list(value.get("source_entry_ids")),
                "exposed_source_ids": [],
                "exposure_mode": "not_evaluable",
                "reason": "legacy proxy exposure has no final-call source spans",
            }
        return value

    @model_validator(mode="after")
    def _validate_exposure_fields(self) -> ContaminationExposure:
        if self.status == "supported":
            if self.is_exposed is None:
                raise ValueError("supported exposure requires is_exposed")
            if not self.answer_call_id:
                raise ValueError("supported exposure requires answer_call_id")
            expected_mode = "final_prompt" if self.is_exposed else "not_in_final_prompt"
            if self.exposure_mode != expected_mode:
                raise ValueError(f"supported exposure requires exposure_mode={expected_mode}")
            if self.is_exposed and not self.exposed_source_ids:
                raise ValueError("exposed final-prompt evidence requires exposed_source_ids")
        elif self.status == "not_applicable":
            if self.is_exposed is not None or self.exposure_mode != "clean":
                raise ValueError("not_applicable exposure must be clean and unevaluable")
        elif self.is_exposed is not None or self.exposure_mode != "not_evaluable":
            raise ValueError("not_evaluable exposure must not report an exposure result")
        return self


class VerifierResult(BaseModel):
    is_correct: bool
    parsed_answer: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRecord(BaseModel):
    document_id: str
    rank: int
    score: float
    text: str
    title_or_type: str
    clean_or_contaminated: str
    source: str
    corpus_hash: str
    embedding_model_id: str
    embedding_revision: str
    embedding_library_version: str


class MethodCall(BaseModel):
    call_id: str | None = None
    stage: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    raw_response: str | None
    model: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    latency_ms: int | None = Field(default=None, strict=True, ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)
    retry_count: int = 0
    error_type: str | None = None
    retrieved_records: list[RetrievalRecord] = Field(default_factory=list)
    source_spans: list[PromptSourceSpan] = Field(default_factory=list)


class TrialLog(BaseModel):
    trial_id: str
    run_id: str
    task_name: str
    sample_id: str
    baseline: str
    arm: Literal["clean", "contaminated", "contaminated_filter"]
    backbone: str
    input: dict[str, Any]
    gold_or_verifier_spec: dict[str, Any]
    prompt_messages: list[dict[str, str]]
    memory_before: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_memory: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_scores: list[float] = Field(default_factory=list)
    filter_decision: dict[str, Any] | None = None
    raw_response: str | None
    parsed_answer: str | None = None
    verifier_result: VerifierResult | None
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_write_event: dict[str, Any] | None = None
    memory_after: list[dict[str, Any]] = Field(default_factory=list)
    method_calls: list[MethodCall] = Field(default_factory=list)
    contamination_exposure: ContaminationExposure = Field(default_factory=ContaminationExposure)
    bad_memory_uptake_label: BadMemoryUptakeLabel | None = None
    repeated_failure_label: RepeatedFailureLabel | None = None
    recovery_after_filter_label: RecoveryAfterFilterLabel | None = None
    latency_ms: int | None = Field(default=None, strict=True, ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_estimate: float | None = None
    retry_count: int = 0
    error_type: str | None = None
    schema_version: Literal["legacy", "logging_v1", "logging_v2"] = "legacy"
    stage: str = "legacy"
    status: Literal["legacy", "succeeded", "failed"] = "legacy"
    run_metadata_id: str | None = None
    trial_seq: int | None = Field(default=None, ge=0)
    event_seq: int | None = Field(default=None, ge=0)
    answer_call_id: str | None = None
    failure_id: str | None = None
    evaluation_law_id: str | None = None
    target_set_id: str | None = None
    memory_update_mode: MemoryUpdateMode | None = None
    trajectory_pair_id: str | None = None
    checkpoint_index: int | None = Field(default=None, ge=0)
    pair_id: str | None = None
    checkpoint_ref: CheckpointRef | None = None

    @model_validator(mode="before")
    @classmethod
    def _adapt_legacy_row(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if data.get("schema_version") not in {LOGGING_V1, LOGGING_V2}:
            data["schema_version"] = "legacy"
            data["stage"] = "legacy"
            data["status"] = "legacy"
        return data

    @model_validator(mode="after")
    def _validate_versioned_contract(self) -> TrialLog:
        if self.schema_version == "legacy":
            if self.contamination_exposure.status == "supported":
                raise ValueError("legacy rows cannot report supported exposure without answer-call source spans")
            return self

        version = self.schema_version
        if not self.stage or self.stage == "legacy":
            raise ValueError(f"{version} requires stage")
        if self.status not in {"succeeded", "failed"}:
            raise ValueError(f"{version} requires status")
        if not self.run_metadata_id:
            raise ValueError(f"{version} requires run_metadata_id")
        if self.trial_seq is None:
            raise ValueError(f"{version} requires trial_seq")
        if self.event_seq is None:
            raise ValueError(f"{version} requires final event_seq")
        if not self.answer_call_id:
            raise ValueError(f"{version} requires answer_call_id")
        if not self.prompt_messages:
            raise ValueError(f"{version} requires exact answer prompt_messages")

        answer_call = next((call for call in self.method_calls if call.call_id == self.answer_call_id), None)
        if answer_call is None:
            raise ValueError("answer_call_id must identify a method_calls entry")
        if answer_call.messages != self.prompt_messages:
            raise ValueError("prompt_messages must equal the exact answer call messages")

        if self.status == "succeeded":
            for field_name in ("raw_response", "parsed_answer", "verifier_result"):
                if getattr(self, field_name) is None:
                    raise ValueError(f"succeeded logging_v1 requires {field_name}")
        elif not self.failure_id:
            raise ValueError("failed logging_v1 requires failure_id linkage")

        exposure = self.contamination_exposure
        if exposure.answer_call_id is not None and exposure.answer_call_id != self.answer_call_id:
            raise ValueError("contamination exposure must reference answer_call_id")
        if exposure.status == "supported":
            if exposure.answer_call_id != self.answer_call_id:
                raise ValueError("supported exposure must reference answer_call_id")
            span_source_ids = {
                source_id
                for span in answer_call.source_spans
                for source_id in [*span.source_ids, span.entry_id]
            }
            if exposure.is_exposed and not set(exposure.exposed_source_ids).issubset(span_source_ids):
                raise ValueError("exposed source IDs must be present in final-call source spans")
        if self.schema_version == LOGGING_V2:
            self._validate_logging_v2_contract(answer_call)
        return self

    def _validate_logging_v2_contract(self, answer_call: MethodCall) -> None:
        required = {
            "evaluation_law_id": self.evaluation_law_id,
            "target_set_id": self.target_set_id,
            "memory_update_mode": self.memory_update_mode,
            "trajectory_pair_id": self.trajectory_pair_id,
            "pair_id": self.pair_id,
        }
        for field_name, value in required.items():
            if value is None or value == "":
                raise ValueError(f"logging_v2 requires {field_name}")
        if self.contamination_exposure.target_set_id != self.target_set_id:
            raise ValueError("logging_v2 contamination exposure must reference target_set_id")
        if self.contamination_exposure.status == "supported":
            if self.contamination_exposure.evidence_lineage_status is None:
                raise ValueError("logging_v2 supported exposure requires evidence_lineage_status")
            if self.contamination_exposure.is_exposed and not self.contamination_exposure.exposed_entry_ids:
                raise ValueError("logging_v2 exposed evidence requires exposed_entry_ids")
        for span in answer_call.source_spans:
            for field_name in (
                "contamination_class",
                "lineage_status",
                "lineage_basis",
                "target_set_id",
                "is_target_contamination",
            ):
                if getattr(span, field_name) is None:
                    raise ValueError(f"logging_v2 source spans require {field_name}")
            if span.target_set_id != self.target_set_id:
                raise ValueError("logging_v2 source span target_set_id must match trial target_set_id")
        expected_source_ids = _unique_span_entry_ids(answer_call.source_spans)
        if self.contamination_exposure.source_entry_ids != expected_source_ids:
            raise ValueError("logging_v2 source_entry_ids must equal rendered answer span entry IDs")
        writing_baselines = {
            "full_history",
            "reflexion_style",
            "bot_style",
            "dynamic_cheatsheet_optional",
            "dynamic_cheatsheet_rs_optional",
        }
        if self.memory_update_mode == "enabled":
            if self.checkpoint_ref is not None:
                raise ValueError("online logging_v2 trial must not include checkpoint_ref")
        elif self.memory_update_mode == "disabled":
            if self.baseline in writing_baselines:
                raise ValueError("frozen logging_v2 rejects memory-writing baselines")
            if self.checkpoint_ref is None:
                raise ValueError("frozen logging_v2 trial requires checkpoint_ref")
        elif self.checkpoint_ref is not None:
            raise ValueError("not_applicable memory_update_mode must not include checkpoint_ref")


def _validate_phase11_lineage_fields(value: Any) -> None:
    if value.lineage_status == "exact" and value.lineage_basis == "signature":
        raise ValueError("signature basis cannot claim exact lineage")
    if value.contamination_class == "derived" and value.lineage_status == "exact":
        if not value.direct_parent_ids:
            raise ValueError("exact derived lineage requires direct_parent_ids")
        if not value.injected_root_ids:
            raise ValueError("exact derived lineage requires injected_root_ids")


def _validate_contamination_compatibility(
    clean_or_contaminated: str, contamination_class: ContaminationClass | None
) -> None:
    if contamination_class is None:
        return
    if clean_or_contaminated == "clean" and contamination_class != "clean":
        raise ValueError("clean_or_contaminated must agree with contamination_class")
    if clean_or_contaminated == "contaminated" and contamination_class == "clean":
        raise ValueError("clean_or_contaminated must agree with contamination_class")


def _unique_span_entry_ids(spans: list[PromptSourceSpan]) -> list[str]:
    entry_ids: list[str] = []
    for span in spans:
        if span.entry_id not in entry_ids:
            entry_ids.append(span.entry_id)
    return entry_ids


def _v2_target_entry_ids(
    memory_before: list[dict[str, Any]], target_set: TargetContaminationSetSpec
) -> list[str]:
    from memcontam.logging.provenance import target_set_membership

    entries = [MemoryEntry.model_validate(entry) for entry in memory_before]
    return [
        item.entry_id
        for item in (MemoryItemLog.from_memory_entry(entry, entries) for entry in entries)
        if target_set_membership(item, target_set)
    ]
