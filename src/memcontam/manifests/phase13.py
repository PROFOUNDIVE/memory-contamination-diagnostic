from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1)]
EvidenceStatus = Literal[
    "confirmatory_primary", "confirmatory_secondary", "prespecified_sensitivity",
    "descriptive", "exploratory",
]
MultiplicityStatus = Literal[
    "primary_holm_family", "estimation_only", "descriptive_no_inferential_family"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AnalysisWindowBinding(StrictModel):
    analysis_window_id: Identifier
    window_length: Literal[2, 5]
    event_time_start: Literal[0]
    event_time_end: Literal[1, 4]
    outcome_family: Identifier
    evidence_status: EvidenceStatus
    multiplicity_status: MultiplicityStatus

    @model_validator(mode="after")
    def _exact_range(self) -> AnalysisWindowBinding:
        if self.event_time_end != self.window_length - 1:
            raise ValueError("WINDOW_EVENT_RANGE_INVALID")
        return self


class SourceEvent(StrictModel):
    event_time: Annotated[int, Field(ge=0, le=9)]
    absolute_trial_index: Annotated[int, Field(ge=2, le=11)]
    baseline: Identifier
    arm: Identifier
    source_checkpoint_id: Identifier
    branch_checkpoint_id: Identifier
    suffix_id: Identifier
    task: Identifier
    model: Identifier
    decoding_contract_id: Identifier
    prompt_contract_id: Identifier
    tool_contract_id: Identifier
    parser_contract_id: Identifier
    verifier_contract_id: Identifier
    native_semantics_id: Identifier
    session_id: Identifier
    randomness_contract_id: Identifier
    future_feedback_cutoff: Literal[0]
    intervention_id: str | None
    execution_owner_id: Identifier
    status: Literal["succeeded", "failed"]
    verified_score: Literal[0, 1]
    state_before_sha256: Sha256
    state_after_sha256: Sha256


class ConformanceCheck(StrictModel):
    check_id: Identifier
    verdict: Literal["pass", "fail"]
    evidence_sha256: Sha256
    checker_version: Literal["phase13-prefix-checker-v2"]
    source_run_id: Identifier
    source_manifest_id: Identifier
    source_raw_sha256: Sha256


class DerivedWindowRow(StrictModel):
    analysis_window: AnalysisWindowBinding
    source_run_id: Identifier
    source_manifest_id: Identifier
    source_raw_sha256: Sha256
    source_execution_contract_id: Literal["phase13-main-a-h10-execution-v1"]
    source_execution_owner_id: Literal["phase13-h10-execution-owner-v1"]
    source_ordered_stream_sha256: Sha256
    event_time_range: tuple[Literal[0], Literal[1, 4]]
    events: tuple[SourceEvent, ...]
    conformance_id: Literal["phase13-ten-condition-prefix-v1"]
    realization_disposition: Literal["prefix_view"]
    no_new_provider_execution: Literal[True]
    provider_calls: Literal[0]
    task_presentations: Literal[0]
    memory_evolutions: Literal[0]

    @property
    def analysis_window_id(self) -> str:
        return self.analysis_window.analysis_window_id

    @property
    def window_length(self) -> int:
        return self.analysis_window.window_length


class PrefixDerivationArtifact(StrictModel):
    schema_version: Literal["phase13_prefix_derivation_v2"]
    conformance_id: Literal["phase13-ten-condition-prefix-v1"]
    execution_registry_hash: Sha256
    source_raw_sha256: Sha256
    checks: tuple[ConformanceCheck, ...]
    rows: tuple[DerivedWindowRow, ...]


class NotExchangeableWindow(StrictModel):
    analysis_window_id: Identifier
    evidence_status: EvidenceStatus
    multiplicity_status: MultiplicityStatus
    realization_disposition: Literal["not_exchangeable"]


class NotExchangeable(StrictModel):
    status: Literal["not_exchangeable"]
    checks: tuple[ConformanceCheck, ...]
    registered_windows: tuple[NotExchangeableWindow, ...]
    derived_artifact: None
    provider_calls: Literal[0]
    task_presentations: Literal[0]
    memory_evolutions: Literal[0]


__all__ = (
    "AnalysisWindowBinding", "ConformanceCheck", "DerivedWindowRow", "NotExchangeable",
    "NotExchangeableWindow", "PrefixDerivationArtifact", "SourceEvent",
)
