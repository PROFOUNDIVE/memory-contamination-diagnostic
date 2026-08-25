from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from memcontam.contamination.phase12.registry import load_candidate_registry

from .phase13_legacy_rag_audit import OpaqueExclusionRegistry
from .phase13_legacy_rag_bytes import JsonValue, canonical_json_bytes
from .phase13_legacy_rag_models import (
    LegacyRagMaterializationReport,
    PackageManifest,
    RepeatabilityReport,
)
from .phase13_legacy_rag_serialization import (
    TRIPLET_REGISTRY_PATH,
    sha256_file,
    write_json,
)
from .phase13_legacy_rag_task_materialize import (
    LegacyRagMaterializationRequest,
    materialize_stage,
    package_status,
)


class LegacyRagMaterializationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


def materialize_legacy_rag_package(
    request: LegacyRagMaterializationRequest,
) -> LegacyRagMaterializationReport:
    output = request.output
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-stage-") as directory:
        temporary_root = Path(directory)
        primary = temporary_root / "primary"
        repeat = temporary_root / "repeat"
        primary.mkdir()
        repeat.mkdir()
        opaque = OpaqueExclusionRegistry.model_validate_json(
            request.opaque_exclusion_path.read_bytes()
        )
        if not request.allow_unfrozen_meb_threshold_for_tests:
            require_legacy_rag_materialization_ready(opaque)
        triplets = {
            triplet.task: triplet
            for triplet in load_candidate_registry(
                request.repository_root / TRIPLET_REGISTRY_PATH
            ).triplets
        }
        materialize_stage(primary, request, opaque, triplets)
        materialize_stage(repeat, request, opaque, triplets)
        test_only = (
            request.allow_test_embedder
            or request.allow_unfrozen_meb_threshold_for_tests
        )
        compared_hashes = _identical_artifact_hashes(primary, repeat)
        aggregate_payload: dict[str, JsonValue] = dict(compared_hashes)
        aggregate = hashlib.sha256(canonical_json_bytes(aggregate_payload)).hexdigest()
        write_json(
            primary / "repeatability_report.json",
            RepeatabilityReport(
                schema_version="phase13_legacy_rag_repeatability_v1",
                status="PASS",
                compared_artifact_hashes=compared_hashes,
                first_materialization_sha256=aggregate,
                repeat_materialization_sha256=aggregate,
            ).model_dump(mode="json"),
        )
        status = package_status(test_only=test_only)
        artifact_hashes = {
            str(path.relative_to(primary)): sha256_file(path)
            for path in sorted(primary.rglob("*"))
            if path.is_file()
        }
        write_json(
            primary / "manifest.json",
            PackageManifest(
                schema_version="phase13_legacy_rag_manifest_v1",
                package_status=status.package_status,
                materialization_profile=("test_only" if test_only else "production_bge_m3"),
                artifact_hashes=artifact_hashes,
            ).model_dump(mode="json"),
        )
        from .phase13_legacy_rag_validate import validate_legacy_rag_package

        report = validate_legacy_rag_package(
            primary,
            request.repository_root,
            sha256_file(primary / "manifest.json"),
            allow_test_package=test_only,
        )
        primary.replace(output)
        return report


def _identical_artifact_hashes(first: Path, repeat: Path) -> dict[str, str]:
    first_hashes = {
        str(path.relative_to(first)): sha256_file(path)
        for path in sorted(first.rglob("*"))
        if path.is_file()
    }
    repeat_hashes = {
        str(path.relative_to(repeat)): sha256_file(path)
        for path in sorted(repeat.rglob("*"))
        if path.is_file()
    }
    if first_hashes != repeat_hashes:
        raise LegacyRagMaterializationError("LEGACY_RAG_REPEAT_MATERIALIZATION_MISMATCH")
    return first_hashes


def require_legacy_rag_materialization_ready(opaque: OpaqueExclusionRegistry) -> None:
    if opaque.status != "PASS" or any(status != "PASS" for status in opaque.task_statuses.values()):
        raise LegacyRagMaterializationError("MEB_STRUCTURAL_SIMILARITY_THRESHOLD_UNFROZEN")


__all__ = [
    "LegacyRagMaterializationError",
    "LegacyRagMaterializationRequest",
    "materialize_legacy_rag_package",
    "require_legacy_rag_materialization_ready",
]
