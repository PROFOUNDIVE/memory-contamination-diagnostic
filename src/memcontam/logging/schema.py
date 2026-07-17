from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memcontam.memory.stores import MemoryEntry


LOGGING_V1 = "logging_v1"

BadMemoryUptakeLabel = Literal[
    "not_applicable", "not_evaluable", "no_uptake_detected", "uptake_detected"
]
RepeatedFailureLabel = Literal["not_applicable", "first_failure", "repeated_failure"]
RecoveryAfterFilterLabel = Literal["not_applicable", "recovered", "not_recovered"]
ExposureStatus = Literal["supported", "not_applicable", "not_evaluable"]
ExposureMode = Literal["clean", "final_prompt", "not_in_final_prompt", "not_evaluable"]


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
    prompt_version: str
    memory_policy_version: str
    contamination_catalog_version: str
    retry_policy_version: str

    @field_validator("schema_version")
    @classmethod
    def _require_logging_v1(cls, value: str) -> str:
        if value != LOGGING_V1:
            raise ValueError(f"schema_version must be {LOGGING_V1}")
        return value


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

    @model_validator(mode="after")
    def _require_nonempty_span(self) -> PromptSourceSpan:
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
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

    @classmethod
    def from_memory_entry(cls, entry: MemoryEntry) -> MemoryItemLog:
        metadata = dict(entry.metadata)
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

        return cls(
            entry_id=entry.entry_id,
            content_hash=hashlib.sha256(entry.content.encode("utf-8")).hexdigest(),
            memory_type=entry.memory_type,
            clean_or_contaminated=entry.clean_or_contaminated,
            source_trial_id=entry.source_trial_id,
            parent_entry_ids=parent_entry_ids,
            source_entry_ids=source_entry_ids,
            lineage_id=lineage_id,
            version=str(version),
            creation_origin=creation_origin,
            metadata=metadata,
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
    schema_version: Literal["legacy", "logging_v1"] = "legacy"
    stage: str = "legacy"
    status: Literal["legacy", "succeeded", "failed"] = "legacy"
    run_metadata_id: str | None = None
    trial_seq: int | None = Field(default=None, ge=0)
    event_seq: int | None = Field(default=None, ge=0)
    answer_call_id: str | None = None
    failure_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _adapt_legacy_row(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if data.get("schema_version") != LOGGING_V1:
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

        if not self.stage or self.stage == "legacy":
            raise ValueError("logging_v1 requires stage")
        if self.status not in {"succeeded", "failed"}:
            raise ValueError("logging_v1 requires status")
        if not self.run_metadata_id:
            raise ValueError("logging_v1 requires run_metadata_id")
        if self.trial_seq is None:
            raise ValueError("logging_v1 requires trial_seq")
        if self.event_seq is None:
            raise ValueError("logging_v1 requires final event_seq")
        if not self.answer_call_id:
            raise ValueError("logging_v1 requires answer_call_id")
        if not self.prompt_messages:
            raise ValueError("logging_v1 requires exact answer prompt_messages")

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
        return self
