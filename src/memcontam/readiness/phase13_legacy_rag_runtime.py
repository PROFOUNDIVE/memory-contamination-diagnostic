from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.rag.branch_index import BranchIndex
from memcontam.rag.phase12_corpus import BranchCorpus, Document

from .phase13_legacy_rag_errors import LegacyRagValidationError
from .phase13_legacy_rag_models import (
    BranchName,
    CorpusBundle,
    EmbeddingRuntimeIdentity,
    FeasibleTaskName,
    IndexBundle,
)
from .phase13_legacy_rag_serialization import MetadataEmbeddingProvider
from .phase13_legacy_rag_validate import validate_legacy_rag_package


@dataclass(frozen=True, slots=True)
class LegacyRagRuntimeRequest:
    package_root: Path
    repository_root: Path
    task: FeasibleTaskName
    branch: BranchName
    embedder: MetadataEmbeddingProvider
    expected_manifest_sha256: str
    allow_test_embedder: bool = False
    allow_test_package: bool = False


@dataclass(frozen=True, slots=True)
class LoadedLegacyRagState:
    state: RagFrozenStateV3
    index_artifact_hash: str


def load_legacy_rag_state(request: LegacyRagRuntimeRequest) -> LoadedLegacyRagState:
    validate_legacy_rag_package(
        request.package_root,
        request.repository_root,
        request.expected_manifest_sha256,
        allow_test_package=request.allow_test_package,
    )
    metadata = request.embedder.metadata
    if (
        metadata.get("model_id") != BgeM3EmbeddingProvider.MODEL_ID
        or metadata.get("revision") != BgeM3EmbeddingProvider.REVISION
        or metadata.get("normalize_embeddings") is not True
        or (
            not request.allow_test_embedder
            and not isinstance(request.embedder, BgeM3EmbeddingProvider)
        )
    ):
        raise LegacyRagValidationError("LEGACY_RAG_RUNTIME_IDENTITY_INVALID")
    corpus_bundle = CorpusBundle.model_validate_json(
        (request.package_root / request.task / "corpus.json").read_bytes()
    )
    index_bundle = IndexBundle.model_validate_json(
        (request.package_root / request.task / "indices.json").read_bytes()
    )
    runtime_identity = EmbeddingRuntimeIdentity.model_validate(metadata)
    if runtime_identity != index_bundle.embedding_runtime:
        raise LegacyRagValidationError("LEGACY_RAG_RUNTIME_IDENTITY_INVALID")
    corpus_data = corpus_bundle.branches[request.branch]
    index_data = index_bundle.branches[request.branch]
    dimension = index_data.embedding_contract.get("dimension")
    if (
        corpus_bundle.task_id != request.task
        or index_bundle.task_id != request.task
        or metadata.get("vector_dimension") != dimension
        or index_data.corpus_serialization_id != corpus_data.serialization_id
        or index_data.documents != corpus_data.documents
        or index_data.branch != request.branch
    ):
        raise LegacyRagValidationError("LEGACY_RAG_RUNTIME_BINDING_INVALID")
    documents = tuple(
        Document.from_mapping(document.model_dump(mode="json"))
        for document in corpus_data.documents
    )
    corpus = BranchCorpus(
        branch=request.branch,
        documents=documents,
        active_document_ids=corpus_data.active_document_ids,
        serialization_id=corpus_data.serialization_id,
    )
    index = BranchIndex(
        branch=request.branch,
        documents=documents,
        embedding_contract=index_data.embedding_contract,
        vectors=index_data.vectors,
        serialization_id=index_data.index_serialization_id,
        _embedder=request.embedder,
    )
    if index.artifact_hash != index_data.index_artifact_hash:
        raise LegacyRagValidationError("LEGACY_RAG_RUNTIME_BINDING_INVALID")
    return LoadedLegacyRagState(
        state=RagFrozenStateV3(branch=request.branch, corpus=corpus, index=index),
        index_artifact_hash=index.artifact_hash,
    )


__all__ = [
    "LegacyRagRuntimeRequest",
    "LoadedLegacyRagState",
    "load_legacy_rag_state",
]
