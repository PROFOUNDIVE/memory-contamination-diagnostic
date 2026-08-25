from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import assert_never

from memcontam.contamination.phase12.models import CandidateTriplet
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import BranchCorpusSet, CleanCorpus, build_branch_corpora

from .phase13_legacy_rag_audit import OpaqueExclusionRegistry
from .phase13_legacy_rag_calibration import (
    build_meb_calibration_registry,
    build_word_sorting_leakage_calibration,
    calibration_artifact_sha256,
)
from .phase13_legacy_rag_construction import (
    CALIBRATION_PATHS,
    BuildRegistrySource,
    build_registry,
    calibration_path,
    calibration_signatures,
    generated_candidates,
)
from .phase13_legacy_rag_documents import clean_documents
from .phase13_legacy_rag_errors import LegacyRagValidationError
from .phase13_legacy_rag_generators import meb_candidates
from .phase13_legacy_rag_models import (
    BRANCHES,
    ArtifactReference,
    FeasibleTaskName,
    MATERIALIZED_TASKS,
    PackageStatus,
    TaskStatus,
)
from .phase13_legacy_rag_serialization import (
    CorpusBundleSource,
    IndexBundleSource,
    MetadataEmbeddingProvider,
    build_corpus_bundle,
    build_index_bundle,
    sha256_file,
    write_index,
    write_json,
)


@dataclass(frozen=True, slots=True)
class LegacyRagMaterializationRequest:
    output: Path
    repository_root: Path
    opaque_exclusion_path: Path
    embedder: MetadataEmbeddingProvider
    allow_test_embedder: bool = False
    allow_unfrozen_meb_threshold_for_tests: bool = False


class _CachedEmbedder:
    def __init__(self, delegate: MetadataEmbeddingProvider) -> None:
        self._delegate = delegate

    @property
    def metadata(self):
        return self._delegate.metadata

    @lru_cache(maxsize=None)
    def encode_document(self, text: str) -> list[float]:
        return self._delegate.encode_document(text)

    @lru_cache(maxsize=None)
    def encode_query(self, text: str) -> list[float]:
        return self._delegate.encode_query(text)


@dataclass(frozen=True, slots=True)
class _MaterializationContext:
    request: LegacyRagMaterializationRequest
    opaque: OpaqueExclusionRegistry
    triplets: dict[str, CandidateTriplet]
    embedder: _CachedEmbedder


def materialize_stage(
    root: Path,
    request: LegacyRagMaterializationRequest,
    opaque: OpaqueExclusionRegistry,
    triplets: dict[str, CandidateTriplet],
) -> None:
    if not request.allow_test_embedder and not isinstance(
        request.embedder, BgeM3EmbeddingProvider
    ):
        raise LegacyRagValidationError("LEGACY_RAG_RUNTIME_IDENTITY_INVALID")
    shutil.copyfile(request.opaque_exclusion_path, root / "opaque_exclusion_registry.json")
    context = _MaterializationContext(request, opaque, triplets, _CachedEmbedder(request.embedder))
    for task in MATERIALIZED_TASKS:
        _materialize_task(root, task, context)
    write_json(
        root / "package_status.json",
        package_status(
            test_only=(
                request.allow_test_embedder
                or request.allow_unfrozen_meb_threshold_for_tests
            )
        ).model_dump(mode="json"),
    )


def _materialize_task(
    root: Path,
    task: FeasibleTaskName,
    context: _MaterializationContext,
) -> None:
    task_root = root / task
    task_root.mkdir()
    opaque_signatures = frozenset(context.opaque.signature_hashes[task])
    historical_pilot_status = None
    leakage_artifact = None
    structural_threshold_artifact = None
    match task:
        case "game24":
            calibration = calibration_path(context.request.repository_root, task)
            calibration_hashes = calibration_signatures(task, calibration)
            candidates = generated_candidates(
                task, frozenset((*opaque_signatures, *calibration_hashes))
            )
            calibration_registry_path = CALIBRATION_PATHS[task]
            calibration_registry_id = "legacy_game24_pilot_calibration_registry_v1"
            calibration_registry_sha256 = sha256_file(calibration)
            calibration_selection_law = "existing_compatible_pilot_registry"
            build_partition_law = "first_64_noncolliding_certified_candidates"
        case "math_equation_balancer":
            threshold_path = f"{task}/structural_threshold.json"
            write_json(
                root / threshold_path,
                context.opaque.meb_structural_threshold.model_dump(mode="json"),
            )
            structural_threshold_artifact = ArtifactReference(
                path=threshold_path,
                sha256=sha256_file(root / threshold_path),
            )
            eligible = meb_candidates(opaque_signatures, limit=80)
            calibration_registry = build_meb_calibration_registry(eligible[:16])
            calibration_registry_path = f"{task}/calibration_registry.json"
            write_json(root / calibration_registry_path, calibration_registry.model_dump(mode="json"))
            calibration_hashes = tuple(row.canonical_signature for row in eligible[:16])
            candidates = eligible[16:]
            calibration_registry_id = calibration_registry.registry_id
            calibration_registry_sha256 = sha256_file(root / calibration_registry_path)
            calibration_selection_law = calibration_registry.selection_law
            build_partition_law = calibration_registry.partition_law
            historical_pilot_status = calibration_registry.historical_rhs_completion_pilot.status
        case "word_sorting":
            calibration = calibration_path(context.request.repository_root, task)
            calibration_hashes = calibration_signatures(task, calibration)
            candidates = generated_candidates(
                task, frozenset((*opaque_signatures, *calibration_hashes))
            )
            leakage = build_word_sorting_leakage_calibration(calibration)
            leakage_path = f"{task}/leakage_calibration.json"
            write_json(root / leakage_path, leakage.model_dump(mode="json"))
            leakage_artifact = ArtifactReference(
                path=leakage_path,
                sha256=calibration_artifact_sha256(leakage),
            )
            calibration_registry_path = CALIBRATION_PATHS[task]
            calibration_registry_id = "legacy_word_sorting_pilot_calibration_registry_v1"
            calibration_registry_sha256 = sha256_file(calibration)
            calibration_selection_law = "existing_compatible_pilot_registry"
            build_partition_law = "first_64_noncolliding_candidates"
        case unreachable:
            assert_never(unreachable)
    registry = build_registry(
        BuildRegistrySource(
            repository_root=context.request.repository_root,
            task=task,
            calibration_hashes=calibration_hashes,
            evaluation_exclusion_hashes=tuple(opaque_signatures),
            calibration_registry_path=calibration_registry_path,
            calibration_registry_id=calibration_registry_id,
            calibration_registry_sha256=calibration_registry_sha256,
            calibration_selection_law=calibration_selection_law,
            build_partition_law=build_partition_law,
            historical_pilot_status=historical_pilot_status,
            leakage_calibration_artifact=leakage_artifact,
            structural_threshold_artifact=structural_threshold_artifact,
            opaque_hash=sha256_file(root / "opaque_exclusion_registry.json"),
            candidates=candidates,
        )
    )
    documents = clean_documents(task, candidates, registry)
    clean = CleanCorpus.from_documents(
        [{"id": document.document_id, "text": document.text} for document in documents],
        corpus_id=f"phase13_legacy_rag_v1::{task}",
    )
    built = build_branch_corpora(clean, context.triplets[task])
    corpora = BranchCorpusSet(
        clean=clean,
        branches={branch: built.branches[branch] for branch in BRANCHES},
        serialization_id=built.serialization_id,
    )
    indices = build_branch_indices(corpora, context.embedder, None)
    write_json(task_root / "build_registry.json", registry.model_dump(mode="json"))
    write_json(
        task_root / "corpus.json",
        build_corpus_bundle(
            CorpusBundleSource(
                context.request.repository_root,
                task,
                documents,
                context.triplets[task],
                corpora,
            )
        ).model_dump(mode="json"),
    )
    write_index(
        task_root / "indices.json",
        build_index_bundle(
            IndexBundleSource(task, corpora, indices, context.embedder)
        ).model_dump(mode="json"),
    )


def package_status(*, test_only: bool = False) -> PackageStatus:
    status = (
        "TEST_ONLY_NOT_READY"
        if test_only
        else "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
    )
    complete = TaskStatus(status=status)
    return PackageStatus(
        schema_version="phase13_legacy_rag_package_status_v1",
        package_status=status,
        tasks={task: complete for task in MATERIALIZED_TASKS},
    )


__all__ = ["LegacyRagMaterializationRequest", "materialize_stage", "package_status"]
