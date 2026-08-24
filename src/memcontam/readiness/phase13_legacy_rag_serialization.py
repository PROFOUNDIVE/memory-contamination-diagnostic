from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from memcontam.contamination.phase12.models import CandidateTriplet
from memcontam.memory.embeddings import EmbeddingProvider, normalized_dot_top_k
from memcontam.rag.branch_index import BranchIndexSet
from memcontam.rag.phase12_corpus import BranchCorpusSet

from .phase13_legacy_rag_bytes import JsonValue, canonical_json_bytes
from .phase13_legacy_rag_documents import semantic_registry_hash
from .phase13_legacy_rag_models import (
    BRANCHES,
    ArtifactReference,
    CorpusBundle,
    CorpusDocument,
    EmbeddingRuntimeIdentity,
    FeasibleTaskName,
    IndexBundle,
    RetrievalDiagnostic,
    SerializedBranchCorpus,
    SerializedBranchIndex,
    SerializedDocument,
)


TRIPLET_REGISTRY_PATH = "data/phase12/registries/candidate_registry_v1.json"


MetadataEmbeddingProvider = EmbeddingProvider


@dataclass(frozen=True, slots=True)
class CorpusBundleSource:
    repository_root: Path
    task: FeasibleTaskName
    documents: tuple[CorpusDocument, ...]
    triplet: CandidateTriplet
    corpora: BranchCorpusSet


@dataclass(frozen=True, slots=True)
class IndexBundleSource:
    task: FeasibleTaskName
    corpora: BranchCorpusSet
    indices: BranchIndexSet
    embedder: MetadataEmbeddingProvider


def build_corpus_bundle(source: CorpusBundleSource) -> CorpusBundle:
    clean_payload = [
        document.payload() for document in source.corpora.branches["clean"].documents
    ]
    triplet_registry = source.repository_root / TRIPLET_REGISTRY_PATH
    clean_hash = hash_json(clean_payload)
    return CorpusBundle(
        schema_version="phase13_legacy_rag_corpus_bundle_v1",
        task_id=source.task,
        clean_documents=source.documents,
        semantic_registry_sha256=semantic_registry_hash(source.task),
        clean_corpus_sha256=clean_hash,
        triplet_registry=ArtifactReference(
            path=TRIPLET_REGISTRY_PATH,
            sha256=sha256_file(triplet_registry),
        ),
        triplet_id=source.triplet.triplet_id,
        triplet_artifact_hash=hash_json(asdict(source.triplet)),
        branches={
            branch: SerializedBranchCorpus(
                branch=branch,
                serialization_id=source.corpora.branches[branch].serialization_id,
                clean_base_hash=clean_hash,
                documents=tuple(
                    SerializedDocument.model_validate(document.payload())
                    for document in source.corpora.branches[branch].documents
                ),
                active_document_ids=source.corpora.branches[branch].active_document_ids,
            )
            for branch in BRANCHES
        },
    )


def build_index_bundle(source: IndexBundleSource) -> IndexBundle:
    metadata = source.embedder.metadata
    clean_index = source.indices.branches["clean"]
    query_document = clean_index.documents[0]
    results = normalized_dot_top_k(
        list(clean_index.vectors[query_document.document_id]),
        [list(clean_index.vectors[document.document_id]) for document in clean_index.documents],
        [document.document_id for document in clean_index.documents],
        3,
    )
    if len(results) != 3:
        message = "legacy RAG retrieval competition requires three ranked documents"
        raise ValueError(message)
    return IndexBundle(
        schema_version="phase13_legacy_rag_serialized_indices_v1",
        task_id=source.task,
        top_k=3,
        similarity="cosine",
        reranker=None,
        score_threshold=None,
        tie_break="document_id_lexical",
        update_mode="frozen_read_only",
        corpus_scope="same_task_only",
        embedding_runtime=EmbeddingRuntimeIdentity.model_validate(metadata),
        retrieval_diagnostic=RetrievalDiagnostic(
            status="PASS",
            query_document_id=query_document.document_id,
            returned_document_ids=(results[0][0], results[1][0], results[2][0]),
            scores=(results[0][1], results[1][1], results[2][1]),
        ),
        branches={
            branch: SerializedBranchIndex(
                branch=branch,
                corpus_serialization_id=source.corpora.branches[branch].serialization_id,
                corpus_content_hash=hash_json(
                    [document.payload() for document in source.indices.branches[branch].documents]
                ),
                index_serialization_id=source.indices.branches[branch].serialization_id,
                index_artifact_hash=source.indices.branches[branch].artifact_hash,
                embedding_contract=dict(
                    source.indices.branches[branch].embedding_contract
                ),
                documents=tuple(
                    SerializedDocument.model_validate(document.payload())
                    for document in source.indices.branches[branch].documents
                ),
                vectors=dict(source.indices.branches[branch].vectors),
            )
            for branch in BRANCHES
        },
    )


def hash_json(value) -> str:
    normalized: JsonValue = json.loads(json.dumps(value))
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def write_json(path: Path, value: JsonValue) -> None:
    path.write_bytes(canonical_json_bytes(value))


def write_index(path: Path, value) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "CorpusBundleSource",
    "IndexBundleSource",
    "MetadataEmbeddingProvider",
    "TRIPLET_REGISTRY_PATH",
    "build_corpus_bundle",
    "build_index_bundle",
    "hash_json",
    "sha256_file",
    "write_index",
    "write_json",
]
