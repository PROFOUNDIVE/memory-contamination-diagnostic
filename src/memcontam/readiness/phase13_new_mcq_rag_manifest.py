from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .phase13_new_mcq_rag_frozen import EXPECTED_CLASSES

REMAINING_OBJECTS = (
    "authority_required_leakage_gate_artifacts",
    "task_local_candidate_selection_and_certification",
    "task_local_intervention_relevance",
    "clean_correct_irrelevant_contam_branch_indices",
)
_SEMANTIC_STRATA = (
    "requirement_quantifier_constraint_interpretation",
    "option_wise_evidence_comparison_elimination",
    "contradiction_counterexample_consistency_checking",
    "uncertainty_management_final_answer_verification",
)


class ManifestError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Artifact(_FrozenModel):
    path: str
    sha256: str


class TaskArtifacts(_FrozenModel):
    documents: Literal[24]
    candidate: Artifact
    review: Artifact
    accepted: Artifact
    index: Artifact
    corpus_hash: str
    index_hashes: dict[Literal["clean"], str]


class Promotion(_FrozenModel):
    status: Literal["NOT_READY"]
    reason: Literal["NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"]
    remaining_objects: tuple[
        Literal[
            "authority_required_leakage_gate_artifacts",
            "verified_bge_m3_snapshot_tree_and_runtime_binding",
            "task_local_candidate_selection_and_certification",
            "task_local_intervention_relevance",
            "clean_correct_irrelevant_contam_branch_indices",
        ],
        ...,
    ]


class PackageManifest(_FrozenModel):
    schema_version: Literal["new_mcq_rag_package_manifest_v1"]
    source_registry: Artifact
    authoring_contract: Artifact
    required_artifacts: dict[str, tuple[Artifact, ...]]
    tasks: dict[str, TaskArtifacts]
    package_reconstruction_identity: str
    promotion: Promotion


class _CandidatePackageStatus(_FrozenModel):
    path: Literal["data/phase13/rag/new_mcq/package_manifest_v1.json"]
    sha256: str
    status: Literal["CLEAN_PACKAGE_NOT_READY"]
    reconstruction_identity: str


class _CellStatus(_FrozenModel):
    status: Literal["NOT_READY"]
    reason: Literal["NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"]
    entry_condition_met: Literal[False]
    missing_objects: tuple[str, ...]
    index_hashes: dict[Literal["clean"], str]


class _RetrievalContract(_FrozenModel):
    corpus_scope: Literal["same_task_only"]
    embedding_model: Literal["BAAI/bge-m3"]
    embedding_revision: Literal["5617a9f61b028005a4858fdac845db406aefb181"]
    reranker: None
    score_threshold: None
    similarity: Literal["cosine"]
    tie_break: Literal["lexical_document_id"]
    top_k: Literal[3]
    update_mode: Literal["frozen_read_only"]


class _ScientificContract(_FrozenModel):
    answer_free: Literal[True]
    atomic_documents: Literal[True]
    documents_per_stratum: Literal[6]
    documents_per_task: Literal[24]
    procedural_only: Literal[True]
    semantic_strata: tuple[str, ...]
    task_specific: Literal[True]


class _PackageStatus(_FrozenModel):
    schema_version: Literal["phase13_new_mcq_rag_status_v1"]
    authority_sha256: Literal["880ba261285758b8c5fea697a105690ffd1c0e4b0b6ab8409673f8408d457b11"]
    candidate_package: _CandidatePackageStatus
    cells: dict[str, _CellStatus]
    cutoff: Literal["2026-08-22T18:00:00+09:00"]
    cutoff_applied: Literal[False]
    retrieval_contract: _RetrievalContract
    scientific_contract: _ScientificContract


@dataclass(frozen=True, slots=True)
class ManifestEvidence:
    candidate_hashes: dict[str, str]
    review_hashes: dict[str, str]
    candidate_corpus_hashes: dict[str, str]


def validate_package_manifest(root: Path, evidence: ManifestEvidence) -> PackageManifest:
    manifest = PackageManifest.model_validate_json((root / "package_manifest_v1.json").read_bytes())
    _require_artifact(root, manifest.source_registry)
    _require_artifact(root, manifest.authoring_contract)
    if (
        set(manifest.required_artifacts) != set(EXPECTED_CLASSES)
        or set(manifest.tasks) != set(evidence.candidate_hashes)
        or manifest.promotion.remaining_objects != REMAINING_OBJECTS
    ):
        raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
    for artifacts in manifest.required_artifacts.values():
        if not artifacts:
            raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
        for artifact in artifacts:
            _require_artifact(root, artifact)
    if {
        key: {artifact.path for artifact in artifacts}
        for key, artifacts in manifest.required_artifacts.items()
    } != _expected_required_paths():
        raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
    for task, artifacts in manifest.tasks.items():
        for artifact in (artifacts.candidate, artifacts.review, artifacts.accepted, artifacts.index):
            _require_artifact(root, artifact)
        if (
            artifacts.candidate.sha256 != evidence.candidate_hashes[task]
            or artifacts.review.sha256 != evidence.review_hashes[task]
            or artifacts.corpus_hash != evidence.candidate_corpus_hashes[task]
            or set(artifacts.index_hashes) != {"clean"}
        ):
            raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
    expected_identity = package_reconstruction_identity(_manifest_artifacts(manifest))
    if manifest.package_reconstruction_identity != expected_identity:
        raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
    _validate_status(root, manifest)
    return manifest


def package_reconstruction_identity(artifacts: tuple[Artifact, ...]) -> str:
    by_path: dict[str, str] = {}
    for artifact in artifacts:
        existing = by_path.get(artifact.path)
        if existing is not None and existing != artifact.sha256:
            raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
        by_path[artifact.path] = artifact.sha256
    return hashlib.sha256(
        "".join(f"{path}\0{by_path[path]}\n" for path in sorted(by_path)).encode()
    ).hexdigest()


def _manifest_artifacts(manifest: PackageManifest) -> tuple[Artifact, ...]:
    task_artifacts = tuple(
        artifact
        for task in manifest.tasks.values()
        for artifact in (task.candidate, task.review, task.accepted, task.index)
    )
    required = tuple(
        artifact
        for artifacts in manifest.required_artifacts.values()
        for artifact in artifacts
    )
    return (manifest.source_registry, manifest.authoring_contract, *required, *task_artifacts)


def _expected_required_paths() -> dict[str, set[str]]:
    tasks = ("mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond")
    return {
        "complete_source_eligibility_registry": {"source_eligibility_registry_v1.json"},
        "accepted_document_registry": {f"accepted/{task}.jsonl" for task in tasks},
        "verified_embedding_runtime_artifact": {"embedding_runtime_v1.json"},
        "serialized_clean_index_artifacts": {f"indices/{task}.json" for task in tasks},
        "partial_clean_document_leakage_evidence": {"leakage_report_v1.json"},
    }


def _validate_status(root: Path, manifest: PackageManifest) -> None:
    status = _PackageStatus.model_validate_json(
        (root.parent / "new_mcq_rag_status_v1.json").read_bytes()
    )
    package = status.candidate_package
    if (
        package.sha256 != _sha256(root / "package_manifest_v1.json")
        or package.reconstruction_identity != manifest.package_reconstruction_identity
        or set(status.cells) != set(manifest.tasks)
        or status.scientific_contract.semantic_strata != _SEMANTIC_STRATA
        or any(
            cell.missing_objects != REMAINING_OBJECTS
            or cell.index_hashes != manifest.tasks[task].index_hashes
            for task, cell in status.cells.items()
        )
    ):
        raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")


def _require_artifact(root: Path, artifact: Artifact) -> None:
    if (
        Path(artifact.path).is_absolute()
        or ".." in Path(artifact.path).parts
        or _sha256(root / artifact.path) != artifact.sha256
    ):
        raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "REMAINING_OBJECTS",
    "ManifestError",
    "ManifestEvidence",
    "PackageManifest",
    "package_reconstruction_identity",
    "validate_package_manifest",
]
