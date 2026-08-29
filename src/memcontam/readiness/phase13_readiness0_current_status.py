from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_readiness0_live_models import (
    ArtifactBinding,
    LiveAuthorization,
    LiveRequest,
)
from memcontam.readiness.phase13_readiness0_f1c_report import validate_f1c_registry
from memcontam.readiness.phase13_readiness0_package import (
    validate_implementation_manifest,
    validate_window_proof,
)


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CurrentReadiness0StatusError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CurrentReadiness0Artifacts(_FrozenModel):
    live_request: ArtifactBinding
    live_authorization: ArtifactBinding
    f1c_registry: ArtifactBinding
    f1c_report: ArtifactBinding
    implementation_manifest: ArtifactBinding
    window_proof: ArtifactBinding


@dataclass(frozen=True, slots=True)
class PreliveArtifactBytes:
    live_request: bytes
    live_authorization: bytes
    f1c_registry: bytes
    f1c_report: bytes
    implementation_manifest: bytes
    window_proof: bytes
    repository_root: Path
    credential_present: bool


class CurrentReadiness0Status(_FrozenModel):
    schema_version: Literal["phase13_readiness0_current_status_v1"]
    status: Literal["PRE_DISPATCH_BLOCKED"]
    failure_code: Literal["READINESS0_PRE_DISPATCH_BLOCKED"]
    external_blockers: tuple[
        Literal["OPENAI_API_KEY_MISSING", "READINESS0_REAUTHORIZATION_REQUIRED"], ...
    ]
    f1c_status: Literal["PASS"]
    provider_calls_issued: Literal[0]
    output_directory_created: Literal[False]
    scientific_result: Literal[False]
    main_result: Literal[False]
    measured_main_a_trajectory_count: Literal[0]
    mr_p4_status: Literal["OPEN"]
    mr_p4_closure_claimed: Literal[False]
    mr_p5_status: Literal["NOT_STARTED"]
    mr_p6_status: Literal["NOT_AUTHORIZED"]
    main_a_status: Literal["NOT_STARTED"]
    main_execution_authorized: Literal[False]
    credential_present: bool
    artifacts: CurrentReadiness0Artifacts
    status_hash: Sha256


def build_current_readiness0_status(inputs: PreliveArtifactBytes) -> CurrentReadiness0Status:
    artifacts, authorization_current = _validate_prelive_artifacts(inputs)
    blockers = (
        *( () if inputs.credential_present else ("OPENAI_API_KEY_MISSING",) ),
        *( () if authorization_current else ("READINESS0_REAUTHORIZATION_REQUIRED",) ),
    )
    status = CurrentReadiness0Status(
        schema_version="phase13_readiness0_current_status_v1",
        status="PRE_DISPATCH_BLOCKED",
        failure_code="READINESS0_PRE_DISPATCH_BLOCKED",
        external_blockers=blockers,
        f1c_status="PASS",
        provider_calls_issued=0,
        output_directory_created=False,
        scientific_result=False,
        main_result=False,
        measured_main_a_trajectory_count=0,
        mr_p4_status="OPEN",
        mr_p4_closure_claimed=False,
        mr_p5_status="NOT_STARTED",
        mr_p6_status="NOT_AUTHORIZED",
        main_a_status="NOT_STARTED",
        main_execution_authorized=False,
        credential_present=inputs.credential_present,
        artifacts=artifacts,
        status_hash="0" * 64,
    )
    return status.model_copy(
        update={"status_hash": _canonical_hash(status.model_dump(mode="json", exclude={"status_hash"}))}
    )


def validate_current_readiness0_status(
    status_raw: bytes,
    inputs: PreliveArtifactBytes,
) -> CurrentReadiness0Status:
    try:
        status = CurrentReadiness0Status.model_validate_json(status_raw)
    except ValidationError as error:
        raise CurrentReadiness0StatusError("READINESS0_CURRENT_STATUS_INVALID") from error
    artifacts, authorization_current = _validate_prelive_artifacts(inputs)
    blockers = (
        *( () if inputs.credential_present else ("OPENAI_API_KEY_MISSING",) ),
        *( () if authorization_current else ("READINESS0_REAUTHORIZATION_REQUIRED",) ),
    )
    if (
        status.artifacts != artifacts
        or status.external_blockers != blockers
        or status.credential_present != inputs.credential_present
        or status.status_hash
        != _canonical_hash(status.model_dump(mode="json", exclude={"status_hash"}))
    ):
        raise CurrentReadiness0StatusError("READINESS0_CURRENT_STATUS_INVALID")
    return status


def _validate_prelive_artifacts(
    inputs: PreliveArtifactBytes,
) -> tuple[CurrentReadiness0Artifacts, bool]:
    try:
        request = LiveRequest.model_validate_json(inputs.live_request)
        authorization = LiveAuthorization.model_validate_json(inputs.live_authorization)
        f1c = validate_f1c_registry(
            inputs.f1c_registry,
            inputs.f1c_report,
            inputs.repository_root,
        )
    except ValidationError as error:
        raise CurrentReadiness0StatusError("READINESS0_CURRENT_STATUS_INVALID") from error
    request_sha256 = _sha256(inputs.live_request)
    f1c_sha256 = _sha256(inputs.f1c_registry)
    implementation_sha256 = _sha256(inputs.implementation_manifest)
    window_sha256 = _sha256(inputs.window_proof)
    report_sha256 = _sha256(inputs.f1c_report)
    root = inputs.repository_root / "data/phase13/main/mr_p4"
    validate_implementation_manifest(inputs.implementation_manifest, inputs.repository_root)
    validate_window_proof(inputs.window_proof, root, inputs.repository_root)
    if (
        request.request_hash
        != _canonical_hash(request.model_dump(mode="json", exclude={"request_hash"}))
        or f1c.f1c_hash != _canonical_hash(f1c.model_dump(mode="json", exclude={"f1c_hash"}))
        or request.f1c_registry.sha256 != f1c_sha256
        or request.implementation_manifest.sha256 != implementation_sha256
        or request.window_proof.sha256 != window_sha256
        or f1c.report.sha256 != report_sha256
    ):
        raise CurrentReadiness0StatusError("READINESS0_CURRENT_STATUS_INVALID")
    return CurrentReadiness0Artifacts(
        live_request=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_live_request_v1.json",
            sha256=request_sha256,
        ),
        live_authorization=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_live_authorization_v1.json",
            sha256=_sha256(inputs.live_authorization),
        ),
        f1c_registry=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_f1c_registry_v1.json",
            sha256=f1c_sha256,
        ),
        f1c_report=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_f1c_report_v1.json",
            sha256=report_sha256,
        ),
        implementation_manifest=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_live_implementation_manifest_v1.json",
            sha256=implementation_sha256,
        ),
        window_proof=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_window_proof_v1.json",
            sha256=window_sha256,
        ),
    ), authorization.request_sha256 == request_sha256


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(payload: dict[str, JsonValue]) -> str:
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


__all__ = [
    "CurrentReadiness0Status",
    "CurrentReadiness0StatusError",
    "PreliveArtifactBytes",
    "build_current_readiness0_status",
    "validate_current_readiness0_status",
]
