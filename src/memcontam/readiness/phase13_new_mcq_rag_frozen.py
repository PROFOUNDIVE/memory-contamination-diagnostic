from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY, BranchIndex, EmbeddingProvider
from memcontam.rag.phase12_corpus import BranchCorpus, Document
from memcontam.readiness.phase13_new_mcq_bge import (
    validate_runtime_artifact,
    verify_runtime_binding,
)
from memcontam.readiness.phase13_new_mcq_p0_4_evidence import validate_candidate_evidence

TASKS = ("mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond")
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = (
    "complete_source_eligibility_registry",
    "accepted_document_registry",
    "verified_embedding_runtime_artifact",
    "serialized_clean_index_artifacts",
    "partial_clean_document_leakage_evidence",
    "partial_task_local_candidate_evidence",
)


class FrozenArtifactError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AcceptedDocument(_FrozenModel):
    schema_version: Literal["new_mcq_rag_clean_doc_v1"]
    document_id: str
    task_id: str
    semantic_stratum: str
    document_ordinal: int
    text: str
    source_registry_ids: tuple[str, ...]
    authoring_template_id: Literal["new_mcq_procedural_atomic_v1"]
    review_status: Literal["accepted"]
    content_hash: str


class SerializedCleanIndex(_FrozenModel):
    schema_version: Literal["new_mcq_rag_serialized_clean_index_v1"]
    task_id: str
    corpus_serialization_id: str
    corpus_content_hash: str
    index_serialization_id: str
    index_artifact_hash: str
    embedding_contract: dict[str, str | int | bool]
    top_k: Literal[3]
    documents: tuple[dict[str, str], ...]
    vectors: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class FrozenRagState:
    state: RagFrozenStateV3
    index_artifact_hash: str
    reconstruction_identity: str


def validate_frozen_artifacts(
    root: Path,
    evaluation_root: Path,
    expected_corpus_hashes: dict[str, str],
) -> dict[str, str]:
    accepted = {task: _accepted(root, task) for task in TASKS}
    if any(
        canonical_json_hash([{"id": row.document_id, "text": row.text} for row in accepted[task]])
        != expected_corpus_hashes[task]
        for task in TASKS
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_ACCEPTED_DOCUMENT_REGISTRY_INVALID")
    _validate_source_eligibility(root)
    validate_runtime_artifact(root)
    _validate_leakage(root, evaluation_root)
    validate_candidate_evidence(root, evaluation_root)
    hashes: dict[str, str] = {}
    for task in TASKS:
        serialized = _serialized_index(root, task)
        _validate_clean_index(serialized, accepted[task])
        hashes[task] = serialized.index_artifact_hash
    return hashes


def load_frozen_clean_state(
    root: Path,
    task: str,
    embedder: EmbeddingProvider,
    *,
    allow_test_embedder: bool = False,
) -> FrozenRagState:
    return _load_frozen_clean_state(root, task, embedder, allow_test_embedder, True)


def _load_frozen_clean_state_for_test(
    root: Path,
    task: str,
    embedder: EmbeddingProvider,
) -> FrozenRagState:
    return _load_frozen_clean_state(root, task, embedder, True, False)


def _load_frozen_clean_state(
    root: Path,
    task: str,
    embedder: EmbeddingProvider,
    allow_test_embedder: bool,
    verify_snapshot: bool,
) -> FrozenRagState:
    metadata = getattr(embedder, "metadata", {})
    if (
        task not in TASKS
        or metadata.get("model_id") != BgeM3EmbeddingProvider.MODEL_ID
        or metadata.get("revision") != BgeM3EmbeddingProvider.REVISION
        or metadata.get("vector_dimension") != BgeM3EmbeddingProvider.VECTOR_DIMENSION
        or metadata.get("normalize_embeddings") is not True
        or (not allow_test_embedder and not isinstance(embedder, BgeM3EmbeddingProvider))
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_RUNTIME_IDENTITY_INVALID")
    if verify_snapshot:
        if not isinstance(embedder, BgeM3EmbeddingProvider):
            raise FrozenArtifactError("NEW_MCQ_RAG_RUNTIME_SNAPSHOT_UNVERIFIED")
        verify_runtime_binding(root, embedder)
    serialized = _serialized_index(root, task)
    _validate_clean_index(serialized, _accepted(root, task))
    documents = tuple(Document.from_mapping(row) for row in serialized.documents)
    corpus = BranchCorpus(
        branch="clean",
        documents=documents,
        active_document_ids=tuple(row.document_id for row in documents),
        serialization_id=serialized.corpus_serialization_id,
    )
    index = BranchIndex(
        branch="clean",
        documents=documents,
        embedding_contract=serialized.embedding_contract,
        vectors=serialized.vectors,
        serialization_id=serialized.index_serialization_id,
        _embedder=embedder,
    )
    if corpus.content_hash != serialized.corpus_content_hash or index.artifact_hash != serialized.index_artifact_hash:
        raise FrozenArtifactError("NEW_MCQ_RAG_SERIALIZED_INDEX_INVALID")
    identity = canonical_json_hash(
        {"task": task, "branch": "clean", "index": index.artifact_hash, "schema": serialized.schema_version}
    )
    return FrozenRagState(RagFrozenStateV3("clean", corpus, index), index.artifact_hash, identity)


def _accepted(root: Path, task: str) -> tuple[AcceptedDocument, ...]:
    rows = tuple(
        AcceptedDocument.model_validate_json(line)
        for line in (root / "accepted" / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
    )
    if (
        len(rows) != 24
        or len({row.document_id for row in rows}) != 24
        or any(
            row.task_id != task or row.content_hash != canonical_json_hash(row.text)
            for row in rows
        )
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_ACCEPTED_DOCUMENT_REGISTRY_INVALID")
    return rows


def _validate_source_eligibility(root: Path) -> None:
    source = json.loads((root / "source_eligibility_registry_v1.json").read_bytes())
    if source.get("status") != "COMPLETE" or set(source.get("tasks", {})) != set(TASKS):
        raise FrozenArtifactError("NEW_MCQ_RAG_SOURCE_ELIGIBILITY_INVALID")


def _validate_leakage(root: Path, evaluation_root: Path) -> None:
    report = json.loads((root / "leakage_report_v1.json").read_bytes())
    artifacts = report.get("evaluation_artifacts")
    review_evidence = report.get("procedural_review_evidence")
    completed_checks = {
        "document_id_uniqueness": "PASS",
        "exact_document_duplicate": "PASS",
        "canonical_document_duplicate": "PASS",
        "cross_task_exact_or_canonical_duplicate": "PASS",
        "exact_evaluation_question_or_option_overlap": "PASS",
    }
    if (
        report.get("status") != "NOT_READY_REQUIRED_LEAKAGE_GATE_UNFROZEN"
        or report.get("scope") != "accepted_clean_documents_only"
        or report.get("completed_deterministic_checks") != completed_checks
        or report.get("missing_objects")
        != [
            "task_specific_canonicalizers",
            "displayed_permutation_equivalence",
            "near_duplicate_threshold",
            "structural_similarity_threshold",
            "lexical_overlap_threshold",
            "source_span_registry",
            "exclusion_manifest",
        ]
        or report.get("evaluation_manifest_sha256")
        != _sha256(evaluation_root / "manifest.json")
        or not isinstance(artifacts, dict)
        or not isinstance(review_evidence, dict)
        or review_evidence.get("review_contract_id") != "new_mcq_procedural_review_v1"
        or not isinstance(review_evidence.get("task_review_sha256"), dict)
        or any(
            artifacts.get(task) != _sha256(evaluation_root / f"{task}.jsonl")
            or review_evidence["task_review_sha256"].get(task)
            != _sha256(root / "reviews" / f"{task}.json")
            for task in TASKS
        )
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_LEAKAGE_AUDIT_INVALID")


def _serialized_index(root: Path, task: str) -> SerializedCleanIndex:
    value = SerializedCleanIndex.model_validate_json((root / "indices" / f"{task}.json").read_bytes())
    if value.task_id != task:
        raise FrozenArtifactError("NEW_MCQ_RAG_SERIALIZED_INDEX_INVALID")
    return value


def _validate_clean_index(
    serialized: SerializedCleanIndex,
    accepted: tuple[AcceptedDocument, ...],
) -> None:
    documents = tuple(Document.from_mapping(row) for row in serialized.documents)
    expected_documents = tuple(
        {"id": row.document_id, "text": row.text} for row in accepted
    )
    expected_ids = {row.document_id for row in accepted}
    computed_corpus_hash = canonical_json_hash([document.payload() for document in documents])
    computed_index_hash = canonical_json_hash(
        {
            "documents": [document.payload() for document in documents],
            "embedding_contract": serialized.embedding_contract,
            "vectors": {document_id: list(vector) for document_id, vector in serialized.vectors.items()},
        }
    )
    if (
        serialized.embedding_contract.get("production_identity") != BGE_M3_PRIMARY_IDENTITY
        or len(documents) != 24
        or tuple(document.payload() for document in documents) != expected_documents
        or {row.document_id for row in documents} != expected_ids
        or set(serialized.vectors) != expected_ids
        or any(len(vector) != 1024 for vector in serialized.vectors.values())
        or computed_corpus_hash != serialized.corpus_content_hash
        or computed_index_hash != serialized.index_artifact_hash
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_SERIALIZED_INDEX_INVALID")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "EXPECTED_CLASSES",
    "FrozenArtifactError",
    "FrozenRagState",
    "SerializedCleanIndex",
    "load_frozen_clean_state",
    "validate_frozen_artifacts",
]
