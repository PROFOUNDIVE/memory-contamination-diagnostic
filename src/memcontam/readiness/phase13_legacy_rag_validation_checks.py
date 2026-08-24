from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from memcontam.contamination.phase12.models import CandidateTriplet
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, normalized_dot_top_k
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY, BRANCH_INDEX_VERSION
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora

from .phase13_legacy_rag_construction import (
    Candidate,
)
from .phase13_legacy_rag_documents import clean_documents, semantic_registry_hash
from .phase13_legacy_rag_errors import fail_validation
from .phase13_legacy_rag_models import (
    BRANCHES,
    BuildRegistry,
    CorpusBundle,
    FeasibleTaskName,
    IndexBundle,
    SerializedDocument,
)
from .phase13_legacy_rag_serialization import (
    TRIPLET_REGISTRY_PATH,
    hash_json,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class CorpusCheckSource:
    repository_root: Path
    task: FeasibleTaskName
    registry: BuildRegistry
    corpus: CorpusBundle
    triplet: CandidateTriplet
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class IndexCheckSource:
    task: FeasibleTaskName
    corpus: CorpusBundle
    bundle: IndexBundle


def validate_corpus(source: CorpusCheckSource) -> None:
    expected_documents = clean_documents(
        source.task,
        source.candidates,
        source.registry,
    )
    clean = CleanCorpus.from_documents(
        [
            {"id": document.document_id, "text": document.text}
            for document in expected_documents
        ],
        corpus_id=f"phase13_legacy_rag_v1::{source.task}",
    )
    expected_branches = build_branch_corpora(clean, source.triplet)
    clean_payload = [document.payload() for document in clean.documents]
    triplet_registry = source.repository_root / TRIPLET_REGISTRY_PATH
    if (
        source.corpus.task_id != source.task
        or source.corpus.clean_documents != expected_documents
        or [
            sum(document.semantic_stratum == stratum for document in expected_documents)
            for stratum in "ABCD"
        ]
        != [6, 6, 6, 6]
        or source.corpus.semantic_registry_sha256 != semantic_registry_hash(source.task)
        or source.corpus.clean_corpus_sha256 != hash_json(clean_payload)
        or source.corpus.triplet_registry.path != TRIPLET_REGISTRY_PATH
        or source.corpus.triplet_registry.sha256 != sha256_file(triplet_registry)
        or source.corpus.triplet_id != source.triplet.triplet_id
        or source.corpus.triplet_artifact_hash != hash_json(asdict(source.triplet))
        or set(source.corpus.branches) != set(BRANCHES)
    ):
        fail_validation("LEGACY_RAG_CORPUS_INVALID")
    for branch in BRANCHES:
        observed = source.corpus.branches[branch]
        expected = expected_branches.branches[branch]
        expected_documents_payload = tuple(
            SerializedDocument.model_validate(document.payload())
            for document in expected.documents
        )
        if (
            observed.branch != branch
            or observed.serialization_id != expected.serialization_id
            or observed.clean_base_hash != source.corpus.clean_corpus_sha256
            or observed.documents != expected_documents_payload
            or observed.active_document_ids != expected.active_document_ids
        ):
            fail_validation("LEGACY_RAG_CORPUS_INVALID")


def validate_indices(source: IndexCheckSource) -> None:
    bundle = source.bundle
    if (
        bundle.task_id != source.task
        or set(bundle.branches) != set(BRANCHES)
        or bundle.embedding_runtime.model_id != BgeM3EmbeddingProvider.MODEL_ID
        or bundle.embedding_runtime.revision != BgeM3EmbeddingProvider.REVISION
        or bundle.embedding_runtime.vector_dimension != BgeM3EmbeddingProvider.VECTOR_DIMENSION
        or bundle.embedding_runtime.normalize_embeddings is not True
        or not bundle.embedding_runtime.embedding_library_version
    ):
        fail_validation("LEGACY_RAG_INDEX_INVALID")
    clean_ids = source.corpus.branches["clean"].active_document_ids
    clean_vectors = bundle.branches["clean"].vectors
    dimensions: set[int] = set()
    for branch in BRANCHES:
        index = bundle.branches[branch]
        corpus = source.corpus.branches[branch]
        dimension = index.embedding_contract.get("dimension")
        document_payloads = [document.model_dump(mode="json") for document in index.documents]
        vector_payload = {key: list(value) for key, value in index.vectors.items()}
        artifact_payload = {
            "documents": document_payloads,
            "embedding_contract": index.embedding_contract,
            "vectors": vector_payload,
        }
        if isinstance(dimension, int):
            dimensions.add(dimension)
        if (
            index.branch != branch
            or index.corpus_serialization_id != corpus.serialization_id
            or index.index_serialization_id
            != f"phase13_legacy_rag_v1::{source.task}|base|{branch}|{BRANCH_INDEX_VERSION}"
            or index.embedding_contract.get("production_identity") != BGE_M3_PRIMARY_IDENTITY
            or index.embedding_contract.get("normalized") is not True
            or index.documents != corpus.documents
            or set(index.vectors) != set(corpus.active_document_ids)
            or not isinstance(dimension, int)
            or dimension != bundle.embedding_runtime.vector_dimension
            or any(len(vector) != dimension for vector in index.vectors.values())
            or not all(_unit_vector(vector) for vector in index.vectors.values())
            or any(index.vectors.get(document_id) != clean_vectors[document_id] for document_id in clean_ids)
            or index.corpus_content_hash != hash_json(document_payloads)
            or index.index_artifact_hash != _legacy_index_hash(artifact_payload)
        ):
            fail_validation("LEGACY_RAG_INDEX_INVALID")
    if len(dimensions) != 1:
        fail_validation("LEGACY_RAG_INDEX_INVALID")
    _validate_retrieval_diagnostic(source)


def _validate_retrieval_diagnostic(source: IndexCheckSource) -> None:
    diagnostic = source.bundle.retrieval_diagnostic
    clean = source.bundle.branches["clean"]
    if diagnostic.query_document_id not in clean.vectors:
        fail_validation("LEGACY_RAG_RETRIEVAL_DIAGNOSTIC_INVALID")
    results = normalized_dot_top_k(
        list(clean.vectors[diagnostic.query_document_id]),
        [list(clean.vectors[document.id]) for document in clean.documents],
        [document.id for document in clean.documents],
        3,
    )
    expected_ids = (results[0][0], results[1][0], results[2][0])
    expected_scores = (results[0][1], results[1][1], results[2][1])
    if (
        len(clean.documents) != 24
        or diagnostic.returned_document_ids != expected_ids
        or any(
            not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
            for observed, expected in zip(diagnostic.scores, expected_scores, strict=True)
        )
        or len(set(expected_ids)) != 3
        or any(not document_id.startswith(f"legacy-rag::{source.task}::") for document_id in expected_ids)
    ):
        fail_validation("LEGACY_RAG_RETRIEVAL_DIAGNOSTIC_INVALID")


def _unit_vector(vector: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in vector) and math.isclose(
        math.sqrt(sum(value * value for value in vector)),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def _legacy_index_hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CorpusCheckSource",
    "IndexCheckSource",
    "validate_corpus",
    "validate_indices",
]
