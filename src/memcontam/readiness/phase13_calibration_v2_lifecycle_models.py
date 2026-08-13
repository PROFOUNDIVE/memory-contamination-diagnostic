from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping


LifecycleTerminal = Literal[
    "CALIBRATION_V2_EXTERNAL_BLOCK",
    "CALIBRATION_V2_INVALIDATED",
    "CALIBRATION_V2_COMPLETED",
]


@dataclass(frozen=True, slots=True)
class LifecycleInvocation:
    config_path: Path
    report_path: Path
    request_path: Path | None
    authorization_path: Path | None
    expected_authorization_sha256: str | None
    allow_live_calls: bool
    environment: Mapping[str, str]
    now: datetime | None = None


@dataclass(frozen=True, slots=True)
class SourceTrajectoryEvidence:
    source_run_id: str
    task: str
    seed_id: int
    event_times: tuple[int, ...]
    state_lineage: tuple[tuple[str, str], ...]
    source_execution_count: int
    accounting_status: Literal["closed_complete", "closed_partial"]
    provider_owner_id: str
    offline_owner_id: str
    provider_call_ids: tuple[str, ...]
    settled_call_ids: tuple[str, ...]
    transport_attempt_ids: tuple[str, ...]
    short_window_provider_calls: int
    derived_window_executions: int
    archive_valid: bool
    claim_status: Literal["synthetic_qa_only", "absent"]


@dataclass(frozen=True, slots=True)
class AuthorizedLifecycleEvidence:
    sources: tuple[SourceTrajectoryEvidence, ...]
    support_successes: tuple[tuple[str, int], ...]
    attempted_seeds: tuple[tuple[str, int], ...]
    observed_input_tokens: int
    observed_output_tokens: int
    observed_cost_microusd: int
    observed_latency_milliseconds: int
    observed_storage_bytes: int
    observed_wall_clock_seconds: int


@dataclass(frozen=True, slots=True)
class ResourceValues:
    maximum_semantic_calls: int
    maximum_transport_attempts: int
    maximum_input_tokens: int
    maximum_output_tokens: int
    maximum_cost_microusd: int
    maximum_latency_milliseconds: int
    maximum_storage_bytes: int
    maximum_wall_clock_seconds: int


@dataclass(frozen=True, slots=True)
class LifecycleReport:
    schema_version: Literal["phase13_calibration_v2_lifecycle_report_v1"]
    terminal: LifecycleTerminal
    reason: str | None
    main_terminal: Literal["MAIN_A_EXECUTION_FORBIDDEN"]
    synthetic_qa_only: bool
    unavailable_fields: tuple[str, ...]
    deterministic_evidence: dict[str, str]
    identities: dict[str, str | None]
    provider_construction_count: int
    provider_dispatch_count: int
    run_root: str | None
    archive_status: Literal["absent", "valid", "invalid"]
    claim_status: Literal["absent", "synthetic_qa_only"]
    trajectory_count: int
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    events_per_source: int
    source_execution_count: int
    short_window_provider_calls: int
    derived_window_executions: int
    settled_semantic_calls: int
    settled_transport_attempts: int
    provider_owner_id: str | None
    offline_owner_id: str | None
    archive_valid: bool
    support_cp95: tuple[tuple[str, str], ...]
    attempted_seeds: tuple[tuple[str, int], ...]
    observed_resources: ResourceValues
    capacity_ceilings: ResourceValues
    timestamp: str
    implementation_commit: str
    report_sha256: str


__all__ = (
    "AuthorizedLifecycleEvidence",
    "LifecycleInvocation",
    "LifecycleReport",
    "ResourceValues",
    "SourceTrajectoryEvidence",
)
