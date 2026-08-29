from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import ValidationError

from memcontam.readiness.phase13_readiness0_f1c_build import build_f1c_report
from memcontam.readiness.phase13_readiness0_f1c_contract import (
    F1CReportError,
    canonical_hash,
)
from memcontam.readiness.phase13_readiness0_f1c_validate import (
    validate_f1c_report,
    validate_f1c_reproducibility,
)
from memcontam.readiness.phase13_readiness0_live_models import ArtifactBinding, F1CRegistry


def build_f1c_registry(report_raw: bytes, legacy_manifest_raw: bytes) -> F1CRegistry:
    report = validate_f1c_report(
        report_raw,
        Path(__file__).parents[3],
    )
    registry = F1CRegistry(
        schema_version="phase13_readiness0_f1c_registry_v1",
        status="PASS",
        cache_environment_variable="MEMCONTAM_BGE_CACHE_DIR",
        local_files_only=True,
        model_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        normalize_embeddings=True,
        vector_dimension=1024,
        report=ArtifactBinding(
            path="data/phase13/main/mr_p4/readiness0_f1c_report_v1.json",
            sha256=hashlib.sha256(report_raw).hexdigest(),
        ),
        runtime_hash=report.runtime.runtime_hash,
        legacy_rag_manifest=ArtifactBinding(
            path="data/phase13/rag/legacy/manifest.json",
            sha256=hashlib.sha256(legacy_manifest_raw).hexdigest(),
        ),
        ready_legacy_cells=("game24", "math_equation_balancer", "word_sorting"),
        f1c_hash="0" * 64,
    )
    payload = registry.model_dump(mode="json", exclude={"f1c_hash"})
    return registry.model_copy(update={"f1c_hash": canonical_hash(payload)})


def validate_f1c_registry(
    registry_raw: bytes,
    report_raw: bytes,
    repository_root: Path,
) -> F1CRegistry:
    try:
        registry = F1CRegistry.model_validate_json(registry_raw)
    except ValidationError as error:
        raise F1CReportError("READINESS0_F1C_REGISTRY_INVALID") from error
    report = validate_f1c_report(report_raw, repository_root)
    expected = build_f1c_registry(
        report_raw,
        (repository_root / registry.legacy_rag_manifest.path).read_bytes(),
    )
    if registry != expected or registry.runtime_hash != report.runtime.runtime_hash:
        raise F1CReportError("READINESS0_F1C_REGISTRY_INVALID")
    return registry


__all__ = [
    "F1CReportError",
    "build_f1c_registry",
    "build_f1c_report",
    "validate_f1c_registry",
    "validate_f1c_report",
    "validate_f1c_reproducibility",
]
