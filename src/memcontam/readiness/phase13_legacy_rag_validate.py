from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import ValidationError

from memcontam.contamination.phase12.registry import load_candidate_registry

from .phase13_legacy_rag_audit import (
    LegacyRagAuditError,
    OpaqueExclusionRegistry,
    validate_opaque_registry_identity,
)
from .phase13_legacy_rag_bytes import JsonValue, canonical_json_bytes
from .phase13_legacy_rag_errors import LegacyRagValidationError, fail_validation
from .phase13_legacy_rag_models import (
    MATERIALIZED_TASKS,
    TASKS,
    BuildRegistry,
    CorpusBundle,
    IndexBundle,
    LegacyRagMaterializationReport,
    PackageManifest,
    PackageStatus,
    RepeatabilityReport,
)
from .phase13_legacy_rag_registry_validation import RegistryCheckSource, validate_registry
from .phase13_legacy_rag_serialization import TRIPLET_REGISTRY_PATH, sha256_file
from .phase13_legacy_rag_validation_checks import (
    CorpusCheckSource,
    IndexCheckSource,
    validate_corpus,
    validate_indices,
)


def validate_legacy_rag_package(
    root: Path,
    repository_root: Path,
    expected_manifest_sha256: str,
    *,
    allow_test_package: bool = False,
) -> LegacyRagMaterializationReport:
    try:
        actual_manifest_sha256 = sha256_file(root / "manifest.json")
    except OSError as error:
        raise LegacyRagValidationError("LEGACY_RAG_PACKAGE_SCHEMA_INVALID") from error
    if actual_manifest_sha256 != expected_manifest_sha256:
        fail_validation("LEGACY_RAG_MANIFEST_IDENTITY_MISMATCH")
    manifest = _load(PackageManifest, root / "manifest.json", "LEGACY_RAG_PACKAGE_SCHEMA_INVALID")
    _validate_artifact_hashes(root, manifest)
    status = _load(PackageStatus, root / "package_status.json", "LEGACY_RAG_PACKAGE_STATUS_INVALID")
    opaque = _load(
        OpaqueExclusionRegistry,
        root / "opaque_exclusion_registry.json",
        "LEGACY_RAG_PACKAGE_SCHEMA_INVALID",
    )
    repeatability = _load(
        RepeatabilityReport,
        root / "repeatability_report.json",
        "LEGACY_RAG_REPEAT_MATERIALIZATION_INVALID",
    )
    _validate_status(manifest, status, allow_test_package=allow_test_package)
    try:
        validate_opaque_registry_identity(opaque)
    except LegacyRagAuditError as error:
        raise LegacyRagValidationError(error.code) from error
    _validate_repeatability(root, repeatability)
    triplets = {
        triplet.task: triplet
        for triplet in load_candidate_registry(
            repository_root / TRIPLET_REGISTRY_PATH
        ).triplets
    }
    opaque_hash = sha256_file(root / "opaque_exclusion_registry.json")
    for task in MATERIALIZED_TASKS:
        task_root = root / task
        registry = _load(
            BuildRegistry,
            task_root / "build_registry.json",
            "LEGACY_RAG_BUILD_REGISTRY_INVALID",
        )
        corpus = _load(
            CorpusBundle,
            task_root / "corpus.json",
            "LEGACY_RAG_CORPUS_INVALID",
        )
        indices = _load(
            IndexBundle,
            task_root / "indices.json",
            "LEGACY_RAG_INDEX_INVALID",
        )
        candidates = validate_registry(
            RegistryCheckSource(root, repository_root, opaque, opaque_hash, task, registry)
        )
        validate_corpus(
            CorpusCheckSource(
                repository_root,
                task,
                registry,
                corpus,
                triplets[task],
                candidates,
            )
        )
        validate_indices(IndexCheckSource(task, corpus, indices))
    return LegacyRagMaterializationReport(
        package_status=status.package_status,
        tasks=status.tasks,
    )


def _load(model, path: Path, code: str):
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise LegacyRagValidationError(code) from error


def _validate_artifact_hashes(root: Path, manifest: PackageManifest) -> None:
    expected_paths = {
        "opaque_exclusion_registry.json",
        "package_status.json",
        "repeatability_report.json",
        "math_equation_balancer/calibration_registry.json",
        "math_equation_balancer/structural_threshold.json",
        "word_sorting/leakage_calibration.json",
        *(
            f"{task}/{artifact}.json"
            for task in MATERIALIZED_TASKS
            for artifact in ("build_registry", "corpus", "indices")
        ),
    }
    if any(
        Path(relative).is_absolute() or ".." in Path(relative).parts
        for relative in manifest.artifact_hashes
    ) or set(manifest.artifact_hashes) != expected_paths:
        fail_validation("LEGACY_RAG_ARTIFACT_HASH_MISMATCH")
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_paths != set(manifest.artifact_hashes) or any(
        sha256_file(root / relative) != expected
        for relative, expected in manifest.artifact_hashes.items()
    ):
        fail_validation("LEGACY_RAG_ARTIFACT_HASH_MISMATCH")


def _validate_status(
    manifest: PackageManifest,
    status: PackageStatus,
    *,
    allow_test_package: bool,
) -> None:
    if manifest.materialization_profile == "test_only" and not allow_test_package:
        fail_validation("LEGACY_RAG_TEST_ARTIFACT_PROMOTION_FORBIDDEN")
    expected = (
        "TEST_ONLY_NOT_READY"
        if allow_test_package
        else "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
    )
    expected_profile = "test_only" if allow_test_package else "production_bge_m3"
    if (
        manifest.package_status != status.package_status
        or status.package_status != expected
        or manifest.materialization_profile != expected_profile
        or set(status.tasks) != set(TASKS)
        or any(task.status != expected for task in status.tasks.values())
    ):
        fail_validation("LEGACY_RAG_PACKAGE_STATUS_INVALID")


def _validate_repeatability(root: Path, report: RepeatabilityReport) -> None:
    actual = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.json", "repeatability_report.json"}
    }
    aggregate_payload: dict[str, JsonValue] = dict(actual)
    aggregate = hashlib.sha256(canonical_json_bytes(aggregate_payload)).hexdigest()
    if (
        report.compared_artifact_hashes != actual
        or report.first_materialization_sha256 != aggregate
        or report.repeat_materialization_sha256 != aggregate
    ):
        fail_validation("LEGACY_RAG_REPEAT_MATERIALIZATION_INVALID")


__all__ = ["LegacyRagValidationError", "validate_legacy_rag_package"]
