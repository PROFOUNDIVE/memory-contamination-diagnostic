from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ManifestError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Artifact(_FrozenModel):
    path: str
    sha256: str


class ProvisionalArtifact(Artifact):
    status: Literal["INCOMPLETE", "PARTIAL"]


class TaskArtifacts(_FrozenModel):
    documents: int
    candidate_path: str
    candidate_sha256: str
    review_path: str
    review_sha256: str
    candidate_corpus_hash: str
    provisional_clean_index_hash: str


class Promotion(_FrozenModel):
    status: Literal["NOT_READY"]
    reason: Literal["NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"]
    remaining_objects: tuple[str, ...]


class PackageManifest(_FrozenModel):
    schema_version: Literal["new_mcq_rag_package_manifest_v1"]
    source_registry: Artifact
    authoring_contract: Artifact
    provisional_artifacts: dict[str, ProvisionalArtifact]
    tasks: dict[str, TaskArtifacts]
    promotion: Promotion


@dataclass(frozen=True, slots=True)
class ManifestEvidence:
    candidate_hashes: dict[str, str]
    review_hashes: dict[str, str]
    candidate_corpus_hashes: dict[str, str]
    remaining_objects: tuple[str, ...]


def validate_package_manifest(root: Path, evidence: ManifestEvidence) -> None:
    manifest = PackageManifest.model_validate_json((root / "package_manifest_v1.json").read_bytes())
    _require_artifact(root, manifest.source_registry)
    _require_artifact(root, manifest.authoring_contract)
    for artifact in manifest.provisional_artifacts.values():
        _require_artifact(root, artifact)
    if (
        set(manifest.tasks) != set(evidence.candidate_hashes)
        or manifest.promotion.remaining_objects != evidence.remaining_objects
    ):
        raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
    for task, artifacts in manifest.tasks.items():
        if (
            artifacts.documents != 24
            or artifacts.candidate_sha256 != evidence.candidate_hashes[task]
            or artifacts.review_sha256 != evidence.review_hashes[task]
            or artifacts.candidate_corpus_hash != evidence.candidate_corpus_hashes[task]
            or _sha256(root / artifacts.candidate_path) != artifacts.candidate_sha256
            or _sha256(root / artifacts.review_path) != artifacts.review_sha256
        ):
            raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")


def _require_artifact(root: Path, artifact: Artifact) -> None:
    if _sha256(root / artifact.path) != artifact.sha256:
        raise ManifestError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["ManifestError", "ManifestEvidence", "validate_package_manifest"]
