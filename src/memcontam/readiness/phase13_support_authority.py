from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from memcontam.manifests.phase13 import PrefixDerivationArtifact
from memcontam.readiness.phase13_analysis_contract import (
    EXECUTION_REGISTRY_HASH,
)
from memcontam.readiness.phase13_calibration_v2_runtime_models import (
    CompletedTrajectory,
    TrajectoryRequest,
)
from memcontam.readiness.phase13_prefix_reuse import derive_prefix_windows


ANALYSIS_REGISTRY_HASH = "82960a8f65d316c53bcf55da3e215f0c4b62781643c21155307b40aa9adf4eee"


class SupportAuthorityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def authenticate_conformance(
    certificate: PrefixDerivationArtifact,
    request: TrajectoryRequest | None,
    source: CompletedTrajectory | None,
) -> None:
    if request is None or source is None:
        raise SupportAuthorityError("CONFORMANCE_AUTHORITY_REQUIRED")
    if source.status != "completed":
        raise SupportAuthorityError("CONFORMANCE_SOURCE_NOT_COMPLETED")
    if source.sealed is not True:
        raise SupportAuthorityError("CONFORMANCE_SOURCE_NOT_SEALED")
    if (
        request.stream_id != source.stream_id
        or request.stream_id != source.source_manifest_id
        or source.source_manifest_id != source.source_seal.source_manifest_id
    ):
        raise SupportAuthorityError("CONFORMANCE_SOURCE_IDENTITY_MISMATCH")
    if (
        request.verified.execution.registry_hash != EXECUTION_REGISTRY_HASH
        or source.source_seal.execution_registry_hash != EXECUTION_REGISTRY_HASH
    ):
        raise SupportAuthorityError("CONFORMANCE_EXECUTION_AUTHORITY_MISMATCH")
    if (
        request.verified.analysis.registry_hash != ANALYSIS_REGISTRY_HASH
        or source.source_seal.analysis_registry_hash != ANALYSIS_REGISTRY_HASH
    ):
        raise SupportAuthorityError("CONFORMANCE_ANALYSIS_AUTHORITY_MISMATCH")
    raw_hash = hashlib.sha256(
        b"".join(
            (json.dumps(asdict(event), sort_keys=True, separators=(",", ":")) + "\n").encode()
            for event in source.events
        )
    ).hexdigest()
    if raw_hash != source.source_raw_sha256 or raw_hash != source.source_seal.source_raw_sha256:
        raise SupportAuthorityError("CONFORMANCE_SOURCE_HASH_MISMATCH")
    trusted = derive_prefix_windows(request, source)
    if not isinstance(trusted, PrefixDerivationArtifact):
        raise SupportAuthorityError("CONFORMANCE_NOT_PASSED")
    if certificate != trusted:
        raise SupportAuthorityError("CONFORMANCE_ARTIFACT_MISMATCH")


__all__ = ("SupportAuthorityError", "authenticate_conformance")
