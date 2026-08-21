from __future__ import annotations

from pathlib import Path

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY, BRANCH_INDEX_VERSION
from memcontam.rag.phase12_corpus import BRANCH_CORPUS_VERSION, Document

from .phase13_new_mcq_rag_models import (
    BRANCHES,
    AcceptedDocument,
    BranchName,
    FrozenArtifactError,
    SerializedBranchIndex,
    SerializedIndexBundle,
    TaskInterventions,
)


class SerializedIndexError(FrozenArtifactError):
    def __init__(self) -> None:
        super().__init__("NEW_MCQ_RAG_SERIALIZED_INDEX_INVALID")


def load_serialized_indices(root: Path, task: str) -> SerializedIndexBundle:
    bundle = SerializedIndexBundle.model_validate_json(
        (root / "indices" / f"{task}.json").read_bytes()
    )
    if bundle.task_id != task or set(bundle.branches) != set(BRANCHES):
        raise SerializedIndexError
    return bundle


def validate_serialized_bundle(
    bundle: SerializedIndexBundle,
    accepted: tuple[AcceptedDocument, ...],
    interventions: TaskInterventions,
) -> None:
    for branch in BRANCHES:
        validate_serialized_branch(
            bundle.task_id,
            branch,
            bundle.branches[branch],
            accepted,
            interventions,
        )


def validate_serialized_branch(
    task: str,
    branch: BranchName,
    serialized: SerializedBranchIndex,
    accepted: tuple[AcceptedDocument, ...],
    interventions: TaskInterventions,
) -> None:
    expected = [{"id": row.document_id, "text": row.text} for row in accepted]
    if branch != "clean":
        intervention = interventions.documents[branch]
        expected.append({"id": intervention.document_id, "text": intervention.text})
    documents = tuple(Document.from_mapping(row) for row in serialized.documents)
    payloads = [document.payload() for document in documents]
    computed_index_hash = canonical_json_hash(
        {
            "documents": payloads,
            "embedding_contract": serialized.embedding_contract,
            "vectors": {key: list(value) for key, value in serialized.vectors.items()},
        }
    )
    if (
        serialized.branch != branch
        or serialized.corpus_serialization_id
        != f"new_mcq_rag_v1::{task}|{branch}|{BRANCH_CORPUS_VERSION}"
        or serialized.index_serialization_id
        != f"new_mcq_rag_v1::{task}|base|{branch}|{BRANCH_INDEX_VERSION}"
        or serialized.embedding_contract.get("production_identity") != BGE_M3_PRIMARY_IDENTITY
        or payloads != expected
        or set(serialized.vectors) != {row.document_id for row in documents}
        or any(len(vector) != 1024 for vector in serialized.vectors.values())
        or canonical_json_hash(payloads) != serialized.corpus_content_hash
        or computed_index_hash != serialized.index_artifact_hash
    ):
        raise SerializedIndexError


__all__ = [
    "load_serialized_indices",
    "validate_serialized_branch",
    "validate_serialized_bundle",
]
