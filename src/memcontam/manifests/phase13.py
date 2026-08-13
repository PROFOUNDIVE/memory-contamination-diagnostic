from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1)]
Verdict = Literal["pass", "fail"]
EvidenceStatus = Literal[
    "confirmatory_primary",
    "confirmatory_secondary",
    "prespecified_sensitivity",
    "descriptive",
    "exploratory",
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
    evidence_status: EvidenceStatus
    multiplicity_status: MultiplicityStatus


class _TrajectoryIdentity(StrictModel):
    source_run_id: Identifier
    source_manifest_id: Identifier
    source_checkpoint_id: Identifier
    source_checkpoint_sha256: Sha256
    source_suffix_id: Identifier
    source_ordered_stream_sha256: Sha256
    task: Identifier
    model_snapshot_id: Identifier
    decoding_contract_id: Identifier
    prompt_contract_id: Identifier
    tool_contract_id: Identifier
    parser_contract_id: Identifier
    verifier_contract_id: Identifier
    native_semantics_id: Identifier
    session_contract_id: Identifier
    randomness_contract_id: Identifier
    intervention_id: Identifier
    future_feedback_cutoff: Annotated[int, Field(ge=0)]
    source_execution_contract_id: Literal["phase13-main-a-h10-execution-v1"]
    source_execution_owner_id: Identifier


class ConformanceAuthority(_TrajectoryIdentity):
    schema_version: Literal["phase13_prefix_conformance_authority_v1"]
    authority_id: Identifier
    conformance_id: Literal["phase13-ten-condition-prefix-v1"]
    checker_version: Literal["phase13-prefix-checker-v1"]
    checker_script_sha256: Sha256
    checker_config_sha256: Sha256
    repository_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    source_manifest_sha256: Sha256
    analysis_windows: tuple[AnalysisWindowBinding, ...]

    @field_validator("analysis_windows", mode="before")
    @classmethod
    def _window_tuple(cls, value: list[dict[str, str | int]]) -> tuple[dict[str, str | int], ...]:
        return tuple(value)


class SourceTrajectoryManifest(_TrajectoryIdentity):
    schema_version: Literal["phase13_source_trajectory_v1"]
    source_raw_path: str
    source_raw_sha256: Sha256
    event_count: Annotated[int, Field(gt=0)]


class SourceEvent(StrictModel):
    event_index: Annotated[int, Field(ge=0, le=9)]
    status: Identifier
    source_checkpoint_id: Identifier
    source_suffix_id: Identifier
    task: Identifier
    model_snapshot_id: Identifier
    session_contract_id: Identifier
    intervention_id: Identifier
    state_before_sha256: Sha256
    state_after_sha256: Sha256


class ConformanceCheck(StrictModel):
    check_id: Identifier
    verdict: Verdict
    evidence_sha256: Sha256
    checker_version: Literal["phase13-prefix-checker-v1"]
    source_run_id: Identifier
    source_manifest_id: Identifier
    source_manifest_sha256: Sha256


class DerivedWindowRow(StrictModel):
    analysis_window_id: Identifier
    conformance_id: Literal["phase13-ten-condition-prefix-v1"]
    checker_script_sha256: Sha256
    checker_config_sha256: Sha256
    repository_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    source_run_id: Identifier
    source_manifest_id: Identifier
    source_manifest_sha256: Sha256
    source_checkpoint_id: Identifier
    source_checkpoint_sha256: Sha256
    source_suffix_id: Identifier
    source_ordered_stream_sha256: Sha256
    source_raw_path: str
    source_raw_sha256: Sha256
    source_execution_contract_id: Literal["phase13-main-a-h10-execution-v1"]
    analysis_window: AnalysisWindowBinding
    window_length: Literal[2, 5]
    event_time_range: tuple[Literal[0], Literal[1, 4]]
    events: tuple[SourceEvent, ...]
    evidence_status: EvidenceStatus
    multiplicity_status: MultiplicityStatus
    realization_disposition: Literal["prefix_view"]
    no_new_provider_execution: Literal[True]
    provider_calls: Literal[0]
    task_presentations: Literal[0]
    memory_evolutions: Literal[0]


class PrefixDerivationArtifact(StrictModel):
    schema_version: Literal["phase13_prefix_derivation_v1"]
    conformance_id: Literal["phase13-ten-condition-prefix-v1"]
    checker_script_sha256: Sha256
    checker_config_sha256: Sha256
    repository_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
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


def read_exact_no_follow(path: Path, expected_sha256: str, error_prefix: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            raw = stream.read()
    except OSError as error:
        raise ValueError(f"{error_prefix}_READ_INVALID") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{error_prefix}_HASH_MISMATCH")
    return raw


def load_conformance_authority(path: Path, expected_sha256: str) -> ConformanceAuthority:
    raw = read_exact_no_follow(path, expected_sha256, "AUTHORITY")
    return ConformanceAuthority.model_validate(json.loads(raw))


def load_source_manifest(path: Path, expected_sha256: str) -> SourceTrajectoryManifest:
    raw = read_exact_no_follow(path, expected_sha256, "SOURCE_MANIFEST")
    return SourceTrajectoryManifest.model_validate(json.loads(raw))


__all__ = (
    "AnalysisWindowBinding",
    "ConformanceAuthority",
    "ConformanceCheck",
    "DerivedWindowRow",
    "NotExchangeable",
    "NotExchangeableWindow",
    "PrefixDerivationArtifact",
    "SourceEvent",
    "SourceTrajectoryManifest",
    "load_conformance_authority",
    "load_source_manifest",
    "read_exact_no_follow",
)
