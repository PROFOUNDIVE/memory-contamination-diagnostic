from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from pydantic import ValidationError

from .phase13_legacy_rag_audit import OpaqueExclusionRegistry
from .phase13_legacy_rag_calibration import (
    MebCalibrationRegistry,
    WordSortingLeakageCalibration,
    build_meb_calibration_registry,
    build_word_sorting_leakage_calibration,
    calibration_artifact_sha256,
)
from .phase13_legacy_rag_construction import (
    CALIBRATION_PATHS,
    BuildRegistrySource,
    Candidate,
    build_registry,
    calibration_path,
    calibration_signatures,
    generated_candidates,
)
from .phase13_legacy_rag_errors import fail_validation
from .phase13_legacy_rag_generators import meb_candidates
from .phase13_legacy_rag_models import ArtifactReference, BuildRegistry, FeasibleTaskName
from .phase13_legacy_rag_serialization import sha256_file


@dataclass(frozen=True, slots=True)
class RegistryCheckSource:
    package_root: Path
    repository_root: Path
    opaque: OpaqueExclusionRegistry
    opaque_hash: str
    task: FeasibleTaskName
    registry: BuildRegistry


def validate_registry(source: RegistryCheckSource) -> tuple[Candidate, ...]:
    opaque_signatures = frozenset(source.opaque.signature_hashes[source.task])
    historical_pilot_status = None
    leakage_artifact = None
    match source.task:
        case "game24":
            calibration = calibration_path(source.repository_root, source.task)
            signatures = calibration_signatures(source.task, calibration)
            candidates = generated_candidates(
                source.task, frozenset((*opaque_signatures, *signatures))
            )
            calibration_registry_path = CALIBRATION_PATHS[source.task]
            calibration_registry_id = "legacy_game24_pilot_calibration_registry_v1"
            calibration_registry_sha256 = sha256_file(calibration)
            calibration_selection_law = "existing_compatible_pilot_registry"
            build_partition_law = "first_64_noncolliding_certified_candidates"
        case "math_equation_balancer":
            eligible = meb_candidates(opaque_signatures, limit=80)
            expected_calibration = build_meb_calibration_registry(eligible[:16])
            calibration_registry_path = f"{source.task}/calibration_registry.json"
            observed_calibration = _load(
                MebCalibrationRegistry,
                source.package_root / calibration_registry_path,
                "LEGACY_RAG_BUILD_REGISTRY_INVALID",
            )
            if observed_calibration != expected_calibration:
                fail_validation("LEGACY_RAG_BUILD_REGISTRY_INVALID")
            signatures = tuple(row.canonical_signature for row in eligible[:16])
            candidates = eligible[16:]
            calibration_registry_id = expected_calibration.registry_id
            calibration_registry_sha256 = sha256_file(
                source.package_root / calibration_registry_path
            )
            calibration_selection_law = expected_calibration.selection_law
            build_partition_law = expected_calibration.partition_law
            historical_pilot_status = expected_calibration.historical_rhs_completion_pilot.status
        case "word_sorting":
            calibration = calibration_path(source.repository_root, source.task)
            signatures = calibration_signatures(source.task, calibration)
            candidates = generated_candidates(
                source.task, frozenset((*opaque_signatures, *signatures))
            )
            expected_leakage = build_word_sorting_leakage_calibration(calibration)
            leakage_path = f"{source.task}/leakage_calibration.json"
            observed_leakage = _load(
                WordSortingLeakageCalibration,
                source.package_root / leakage_path,
                "LEGACY_RAG_BUILD_REGISTRY_INVALID",
            )
            if observed_leakage != expected_leakage:
                fail_validation("LEGACY_RAG_BUILD_REGISTRY_INVALID")
            leakage_artifact = ArtifactReference(
                path=leakage_path,
                sha256=calibration_artifact_sha256(expected_leakage),
            )
            calibration_registry_path = CALIBRATION_PATHS[source.task]
            calibration_registry_id = "legacy_word_sorting_pilot_calibration_registry_v1"
            calibration_registry_sha256 = sha256_file(calibration)
            calibration_selection_law = "existing_compatible_pilot_registry"
            build_partition_law = "first_64_noncolliding_candidates"
        case unreachable:
            assert_never(unreachable)
    expected = build_registry(
        BuildRegistrySource(
            repository_root=source.repository_root,
            task=source.task,
            calibration_hashes=signatures,
            evaluation_exclusion_hashes=tuple(opaque_signatures),
            calibration_registry_path=calibration_registry_path,
            calibration_registry_id=calibration_registry_id,
            calibration_registry_sha256=calibration_registry_sha256,
            calibration_selection_law=calibration_selection_law,
            build_partition_law=build_partition_law,
            historical_pilot_status=historical_pilot_status,
            leakage_calibration_artifact=leakage_artifact,
            opaque_hash=source.opaque_hash,
            candidates=candidates,
        )
    )
    if source.registry != expected:
        fail_validation("LEGACY_RAG_BUILD_REGISTRY_INVALID")
    return candidates


def _load(model, path: Path, code: str):
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        fail_validation(code)
        raise AssertionError from error


__all__ = ["RegistryCheckSource", "validate_registry"]
