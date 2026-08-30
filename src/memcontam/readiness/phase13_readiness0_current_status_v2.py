from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_readiness0_current_status import (
    CurrentReadiness0StatusError,
    PreliveArtifactBytes,
    build_current_readiness0_status,
)
from memcontam.readiness.phase13_readiness0_evidence_validate import (
    EvidenceValidationError,
    validate_pass_evidence,
)
from memcontam.readiness.phase13_readiness0_f1c_models import F1CReport
from memcontam.readiness.phase13_readiness0_live import READINESS0_CASES
from memcontam.readiness.phase13_readiness0_live_models import (
    ArtifactBinding,
    CaseEvidence,
    EvidenceManifest,
)


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CurrentReadiness0StatusV2Error(ValueError):
    def __init__(self, code: str = "READINESS0_CURRENT_STATUS_V2_INVALID") -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CurrentReadiness0ArtifactsV2(_FrozenModel):
    live_request: ArtifactBinding
    live_authorization: ArtifactBinding
    f1c_registry: ArtifactBinding
    f1c_report: ArtifactBinding
    implementation_manifest: ArtifactBinding
    window_proof: ArtifactBinding
    live_evidence_manifest: ArtifactBinding
    live_evidence_cases: ArtifactBinding


@dataclass(frozen=True, slots=True)
class ClosedReadiness0ArtifactBytes:
    prelive: PreliveArtifactBytes
    evidence_manifest: bytes
    evidence_cases: bytes


class CurrentReadiness0StatusV2(_FrozenModel):
    schema_version: Literal["phase13_readiness0_current_status_v2"]
    status: Literal["PASS"]
    failure_code: None
    external_blockers: tuple[()]
    f1c_status: Literal["PASS"]
    provider_calls_issued: Literal[12]
    output_directory_created: Literal[True]
    scientific_result: Literal[False]
    main_result: Literal[False]
    measured_main_a_trajectory_count: Literal[0]
    mr_p4_status: Literal["CLOSED"]
    mr_p4_closure_claimed: Literal[True]
    mr_p5_status: Literal["NOT_STARTED"]
    mr_p6_status: Literal["NOT_AUTHORIZED"]
    main_a_status: Literal["NOT_STARTED"]
    main_execution_authorized: Literal[False]
    artifacts: CurrentReadiness0ArtifactsV2
    status_hash: Sha256


def build_current_readiness0_status_v2(
    inputs: ClosedReadiness0ArtifactBytes,
) -> CurrentReadiness0StatusV2:
    artifacts = _validate_artifacts(inputs)
    status = CurrentReadiness0StatusV2(
        schema_version="phase13_readiness0_current_status_v2",
        status="PASS",
        failure_code=None,
        external_blockers=(),
        f1c_status="PASS",
        provider_calls_issued=12,
        output_directory_created=True,
        scientific_result=False,
        main_result=False,
        measured_main_a_trajectory_count=0,
        mr_p4_status="CLOSED",
        mr_p4_closure_claimed=True,
        mr_p5_status="NOT_STARTED",
        mr_p6_status="NOT_AUTHORIZED",
        main_a_status="NOT_STARTED",
        main_execution_authorized=False,
        artifacts=artifacts,
        status_hash="0" * 64,
    )
    return status.model_copy(
        update={"status_hash": _canonical_hash(status.model_dump(mode="json", exclude={"status_hash"}))}
    )


def validate_current_readiness0_status_v2(
    status_raw: bytes,
    inputs: ClosedReadiness0ArtifactBytes,
) -> CurrentReadiness0StatusV2:
    try:
        status = CurrentReadiness0StatusV2.model_validate_json(status_raw)
    except ValidationError as error:
        raise CurrentReadiness0StatusV2Error() from error
    if status != build_current_readiness0_status_v2(inputs):
        raise CurrentReadiness0StatusV2Error()
    return status


def _validate_artifacts(inputs: ClosedReadiness0ArtifactBytes) -> CurrentReadiness0ArtifactsV2:
    try:
        prelive = build_current_readiness0_status(inputs.prelive)
        manifest = EvidenceManifest.model_validate_json(inputs.evidence_manifest)
        rows = tuple(
            CaseEvidence.model_validate_json(line) for line in inputs.evidence_cases.splitlines()
        )
        f1c_report = F1CReport.model_validate_json(inputs.prelive.f1c_report)
        validate_pass_evidence(manifest, rows, READINESS0_CASES)
    except (CurrentReadiness0StatusError, EvidenceValidationError, ValidationError) as error:
        raise CurrentReadiness0StatusV2Error() from error
    request_sha256 = _sha256(inputs.prelive.live_request)
    authorization_sha256 = _sha256(inputs.prelive.live_authorization)
    f1c_sha256 = _sha256(inputs.prelive.f1c_registry)
    cases_sha256 = _sha256(inputs.evidence_cases)
    if (
        "READINESS0_REAUTHORIZATION_REQUIRED" in prelive.external_blockers
        or manifest.request_sha256 != request_sha256
        or manifest.authorization_sha256 != authorization_sha256
        or manifest.f1c_registry_sha256 != f1c_sha256
        or manifest.cases.path != "cases.jsonl"
        or manifest.cases.sha256 != cases_sha256
        or manifest.manifest_hash
        != _canonical_hash(manifest.model_dump(mode="json", exclude={"manifest_hash"}))
        or manifest.f1c_runtime.model_dump(mode="json")
        != f1c_report.runtime.model_dump(mode="json")
        or tuple(row.case_id for row in rows)
        != tuple(case.case_id for case in READINESS0_CASES)
    ):
        raise CurrentReadiness0StatusV2Error()
    return CurrentReadiness0ArtifactsV2(
        live_request=prelive.artifacts.live_request,
        live_authorization=prelive.artifacts.live_authorization,
        f1c_registry=prelive.artifacts.f1c_registry,
        f1c_report=prelive.artifacts.f1c_report,
        implementation_manifest=prelive.artifacts.implementation_manifest,
        window_proof=prelive.artifacts.window_proof,
        live_evidence_manifest=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_live_evidence_v1/evidence_manifest.json",
            sha256=_sha256(inputs.evidence_manifest),
        ),
        live_evidence_cases=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_live_evidence_v1/cases.jsonl",
            sha256=cases_sha256,
        ),
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(payload: dict[str, JsonValue]) -> str:
    return _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


__all__ = [
    "ClosedReadiness0ArtifactBytes",
    "CurrentReadiness0StatusV2",
    "CurrentReadiness0StatusV2Error",
    "build_current_readiness0_status_v2",
    "validate_current_readiness0_status_v2",
]
