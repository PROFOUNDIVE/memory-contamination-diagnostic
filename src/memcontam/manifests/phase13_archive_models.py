from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1)]


class StrictArchiveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AuthorityBinding(StrictArchiveModel):
    path: Identifier
    sha256: Sha256


class ArchiveAuthorities(StrictArchiveModel):
    execution: AuthorityBinding
    analysis: AuthorityBinding
    historical: AuthorityBinding
    checkpoint: AuthorityBinding


class SourceArchiveEvent(StrictArchiveModel):
    event_id: Identifier
    event_time: Annotated[int, Field(ge=0, le=9)]
    absolute_trial_index: Annotated[int, Field(ge=2, le=11)]
    source_checkpoint_id: Identifier
    baseline: Identifier
    arm: Identifier
    task: Identifier
    model: Identifier
    session_id: Identifier
    native_state_id: Identifier
    intervention_id: str | None
    state_before_sha256: Sha256
    state_after_sha256: Sha256
    write_event_ids: tuple[Identifier, ...]
    retention_event_ids: tuple[Identifier, ...]
    eviction_event_ids: tuple[Identifier, ...]
    lineage_parent_ids: tuple[Identifier, ...]
    semantic_call_id: Identifier
    call_owner_id: Identifier
    verified_score: Literal[0, 1]
    status: Literal["succeeded", "failed"]


class SourceAttemptRow(StrictArchiveModel):
    attempt_id: Identifier
    source_run_id: Identifier
    source_manifest_id: Identifier | None = None
    source_ordered_stream_sha256: Sha256 | None = None
    execution_contract_id: Identifier | None = None
    status: Literal["completed", "invalidated"]
    invalidated_reason: str | None
    raw_evidence_sha256: Sha256
    rerun_parent_id: str | None
    source_raw_path: Identifier
    source_raw_sha256: Sha256
    raw_record_range: tuple[int, int]
    events: tuple[SourceArchiveEvent, ...]


class DerivedWindowArchiveRow(StrictArchiveModel):
    window_id: Identifier
    analysis_window_id: Identifier
    source_run_id: Identifier
    source_raw_sha256: Sha256
    source_event_range: tuple[int, int]
    event_ids: tuple[Identifier, ...]
    window_length: Literal[2, 5]
    evidence_status: Identifier
    multiplicity_status: Identifier
    provider_calls: Literal[0]
    owner_id: Identifier


class ProviderLedgerRow(StrictArchiveModel):
    semantic_call_id: Identifier
    execution_owner_id: Identifier
    transport_attempt_ids: tuple[Identifier, ...]


class OfflineLedgerRow(StrictArchiveModel):
    operation: Literal["prefix_derivation", "paired_seed_bootstrap", "report_rendering"]
    owner_id: Identifier
    provider_calls: Annotated[int, Field(ge=0)]
    cost_microusd: Annotated[int, Field(ge=0)] = 0


class AggregateArchiveRow(StrictArchiveModel):
    aggregate_id: Identifier
    source_ids: tuple[Identifier, ...]
    status: Literal["ESTIMABLE", "NOT_ESTIMABLE"]
    family_id: Identifier
    original_weights: dict[Identifier, float]
    weights: dict[Identifier, float]
    estimate: float | Literal["NOT_ESTIMABLE"]


class ClaimArchiveRow(StrictArchiveModel):
    claim_id: Identifier
    aggregate_id: Identifier
    status: Literal["supported", "nonclaim"]
    family_id: Identifier
    estimate: float | Literal["NOT_ESTIMABLE"]


class HistoricalReference(StrictArchiveModel):
    run_id: Identifier
    availability: Identifier
    imported: bool


class Phase13Archive(StrictArchiveModel):
    schema_version: Literal["phase13_archive_v2"]
    authorities: ArchiveAuthorities
    source_attempts: tuple[SourceAttemptRow, ...]
    derived_windows: tuple[DerivedWindowArchiveRow, ...]
    provider_ledger: tuple[ProviderLedgerRow, ...]
    offline_ledger: tuple[OfflineLedgerRow, ...]
    aggregates: tuple[AggregateArchiveRow, ...]
    claims: tuple[ClaimArchiveRow, ...]
    historical_reference: HistoricalReference


__all__ = (
    "AggregateArchiveRow",
    "ClaimArchiveRow",
    "DerivedWindowArchiveRow",
    "Phase13Archive",
    "SourceAttemptRow",
)
