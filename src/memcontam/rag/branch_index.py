from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.memory.embeddings import normalized_dot_top_k
from memcontam.rag.phase12_corpus import BranchCorpus, BranchCorpusSet, Document, MetadataVariantCorpus


BRANCH_INDEX_VERSION = "index-v3"
BGE_M3_PRIMARY_IDENTITY = "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181"


class EmbeddingProvider(Protocol):
    def encode_document(self, text: str) -> list[float]: ...

    def encode_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievalResult:
    document_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class BranchIndex:
    branch: str
    documents: tuple[Document, ...]
    embedding_contract: Mapping[str, Any]
    vectors: Mapping[str, tuple[float, ...]]
    serialization_id: str
    _embedder: EmbeddingProvider = field(repr=False, compare=False)
    index_version: str = BRANCH_INDEX_VERSION

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(
            {
                "documents": [document.payload() for document in self.documents],
                "embedding_contract": dict(self.embedding_contract),
                "vectors": {document_id: list(vector) for document_id, vector in self.vectors.items()},
            }
        )

    def retrieve(self, query: str, k: int) -> tuple[RetrievalResult, ...]:
        scores = normalized_dot_top_k(
            self._embedder.encode_query(query),
            [list(self.vectors[document.document_id]) for document in self.documents],
            [document.document_id for document in self.documents],
            k,
        )
        return tuple(
            RetrievalResult(document_id=document_id, score=score, rank=rank)
            for rank, (document_id, score) in enumerate(scores, start=1)
        )

    def final_inclusion(
        self, retrieved: tuple[RetrievalResult, ...], allowed_document_ids: set[str]
    ) -> tuple[RetrievalResult, ...]:
        return tuple(result for result in retrieved if result.document_id in allowed_document_ids)


@dataclass(frozen=True)
class BranchIndexSet:
    branches: dict[str, BranchIndex]
    serialization_id: str
    top_k: int = 3


def build_branch_indices(
    corpora: BranchCorpusSet | MetadataVariantCorpus,
    embedder: EmbeddingProvider,
    filter_policy: object | None,
) -> BranchIndexSet:
    del filter_policy
    source = corpora.reference if isinstance(corpora, MetadataVariantCorpus) else corpora
    contract = _embedding_contract(embedder)
    indices = {
        name: _build_index(corpus, embedder, contract, corpora.serialization_id)
        for name, corpus in source.branches.items()
    }
    return BranchIndexSet(branches=indices, serialization_id=corpora.serialization_id)


def _build_index(
    corpus: BranchCorpus,
    embedder: EmbeddingProvider,
    contract: Mapping[str, Any],
    set_serialization_id: str,
) -> BranchIndex:
    documents = corpus.active_documents
    vectors = {
        document.document_id: tuple(float(value) for value in embedder.encode_document(document.text))
        for document in documents
    }
    dimension = contract.get("dimension")
    if not isinstance(dimension, int) or any(len(vector) != dimension for vector in vectors.values()):
        raise ValueError("EMBEDDING_DIMENSION_MISMATCH")
    if contract.get("normalized") is not True:
        raise ValueError("EMBEDDING_NORMALIZATION_REQUIRED")
    return BranchIndex(
        branch=corpus.branch,
        documents=documents,
        embedding_contract=contract,
        vectors=vectors,
        serialization_id=f"{set_serialization_id}|{corpus.branch}|{BRANCH_INDEX_VERSION}",
        _embedder=embedder,
    )


def _embedding_contract(embedder: EmbeddingProvider) -> Mapping[str, Any]:
    contract = getattr(embedder, "embedding_contract", None)
    if isinstance(contract, Mapping):
        return _validate_embedding_contract(dict(contract))
    metadata = getattr(embedder, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("EMBEDDING_CONTRACT_REQUIRED")
    model_id = metadata.get("model_id")
    revision = metadata.get("revision")
    dimension = metadata.get("vector_dimension")
    normalized = metadata.get("normalize_embeddings")
    if not all(isinstance(value, str) for value in (model_id, revision)) or not isinstance(dimension, int):
        raise ValueError("EMBEDDING_CONTRACT_REQUIRED")
    return _validate_embedding_contract(
        {
        "dimension": dimension,
        "normalized": normalized is True,
        "production_identity": f"{model_id}@{revision}",
        "provider": model_id,
        }
    )


def _validate_embedding_contract(contract: dict[str, Any]) -> Mapping[str, Any]:
    if contract.get("production_identity") != BGE_M3_PRIMARY_IDENTITY:
        raise ValueError("BGE_M3_IDENTITY_MISMATCH")
    return contract
