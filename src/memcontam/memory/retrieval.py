from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import TypedDict

from memcontam.baselines.contracts import CorpusIdentity
from memcontam.logging.schema import RetrievalRecord
from memcontam.memory.embeddings import EmbeddingProvider
from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.embeddings import normalized_dot_top_k
from memcontam.memory.stores import MemoryEntry


INDEX_FORMAT = "memcontam-dense-json-v1"
NORMALIZE_EMBEDDINGS = True


class RetrievedRecord(TypedDict):
    entry_id: str
    content: str
    score: float
    rank: int
    memory_type: str
    clean_or_contaminated: str
    source_trial_id: str | None
    metadata: dict
    memory_entry: MemoryEntry


class DenseIndex:
    def __init__(
        self,
        entries: list[MemoryEntry],
        *,
        provider: EmbeddingProvider | None = None,
        cache_dir: str | Path = "data/embedding_cache",
        corpus_identity: CorpusIdentity | None = None,
    ) -> None:
        if provider is None:
            raise ValueError("DenseIndex requires an explicit embedding provider")
        self.entries = list(entries)
        self.provider = provider
        self.corpus_identity = corpus_identity
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "dense_index_manifest.json"
        self.vectors_path = self.cache_dir / "dense_index_vectors.json"
        self._entries_by_id = _entries_by_id(self.entries)
        self.manifest = _manifest(self.entries, self.provider)
        self.document_vectors = self._load_or_build_vectors()

    @property
    def corpus_hash(self) -> str:
        return str(self.manifest["corpus_hash"])

    def retrieve(self, query: str, k: int) -> list[RetrievalRecord]:
        if k <= 0 or not self.entries:
            return []
        query_vector = self.provider.encode_query(query)
        document_ids = [entry.entry_id for entry in self.entries]
        scores = normalized_dot_top_k(query_vector, self.document_vectors, document_ids, k)
        return [self._retrieval_record(document_id, score, rank) for rank, (document_id, score) in enumerate(scores, 1)]

    def _retrieval_record(self, document_id: str, score: float, rank: int) -> RetrievalRecord:
        entry = self._entries_by_id[document_id]
        metadata = self.provider.metadata
        return RetrievalRecord(
            document_id=entry.entry_id,
            rank=rank,
            score=score,
            text=entry.content,
            title_or_type=entry.memory_type,
            clean_or_contaminated=entry.clean_or_contaminated,
            source=str(entry.metadata.get("source") or entry.source_trial_id or "memory_entry"),
            corpus_hash=str(self.manifest["corpus_hash"]),
            embedding_model_id=str(metadata["model_id"]),
            embedding_revision=str(metadata["revision"]),
            embedding_library_version=str(metadata["embedding_library_version"]),
        )

    def _load_or_build_vectors(self) -> list[list[float]]:
        if not self.manifest_path.exists() and not self.vectors_path.exists():
            return self._build_vectors()
        if not self.manifest_path.exists() or not self.vectors_path.exists():
            raise ValueError("stale dense index cache: missing manifest or vectors file")
        cached_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if cached_manifest != self.manifest:
            raise ValueError("stale dense index cache: manifest does not match corpus/provider")
        vectors = json.loads(self.vectors_path.read_text(encoding="utf-8"))
        if not isinstance(vectors, list) or len(vectors) != len(self.entries):
            raise ValueError("stale dense index cache: vector count does not match corpus")
        _validate_vectors(vectors, [entry.entry_id for entry in self.entries])
        return vectors

    def _build_vectors(self) -> list[list[float]]:
        vectors = [self.provider.encode_document(entry.content) for entry in self.entries]
        _validate_vectors(vectors, [entry.entry_id for entry in self.entries])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, sort_keys=True, indent=2), encoding="utf-8")
        self.vectors_path.write_text(json.dumps(vectors), encoding="utf-8")
        return vectors


def render_retrieved_record(record: RetrievedRecord) -> str:
    provenance = ", ".join(
        [
            f"memory_type={record['memory_type']}",
            f"clean_or_contaminated={record['clean_or_contaminated']}",
            f"source_trial_id={record['source_trial_id']}",
            f"metadata={record['metadata']}",
        ]
    )
    return (
        f"#{record['rank']} entry_id={record['entry_id']} score={record['score']:.6f} "
        f"{provenance}\n{record['content']}"
    )


def retrieve_records(query: str, entries: list[MemoryEntry], k: int = 3) -> list[RetrievedRecord]:
    if k <= 0 or not entries:
        return []
    with tempfile.TemporaryDirectory() as cache_dir:
        records = DenseIndex(entries, provider=FakeEmbeddingProvider(), cache_dir=cache_dir).retrieve(query, k)
    return [_legacy_record(record, _entries_by_id(entries)[record.document_id]) for record in records]


def lexical_retrieve(query: str, entries: list[MemoryEntry], k: int = 3) -> list[tuple[MemoryEntry, float]]:
    # ponytail: compatibility wrapper for older tuple callers; new code should use retrieve_records().
    return [(record["memory_entry"], record["score"]) for record in retrieve_records(query, entries, k=k)]


def _legacy_record(record: RetrievalRecord, entry: MemoryEntry) -> RetrievedRecord:
    return {
        "entry_id": entry.entry_id,
        "content": entry.content,
        "score": (record.score + 1.0) / 2.0,
        "rank": record.rank,
        "memory_type": entry.memory_type,
        "clean_or_contaminated": entry.clean_or_contaminated,
        "source_trial_id": entry.source_trial_id,
        "metadata": entry.metadata,
        "memory_entry": entry,
    }


def _manifest(entries: list[MemoryEntry], provider: EmbeddingProvider) -> dict[str, object]:
    metadata = provider.metadata
    if metadata.get("normalize_embeddings", True) is not True:
        raise ValueError("embedding provider must use normalized embeddings")
    vector_dimension = metadata["vector_dimension"]
    if not isinstance(vector_dimension, int):
        raise ValueError("embedding provider vector_dimension must be an int")
    manifest = {
        "corpus_hash": _corpus_hash(entries),
        "embedding_model_id": str(metadata["model_id"]),
        "embedding_revision": str(metadata["revision"]),
        "embedding_library_version": str(metadata["embedding_library_version"]),
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "vector_dimension": vector_dimension,
        "index_format": INDEX_FORMAT,
    }
    manifest["manifest_hash"] = _hash_json(manifest)
    return manifest


def _corpus_hash(entries: list[MemoryEntry]) -> str:
    return _hash_json(
        [
            {
                "entry_id": entry.entry_id,
                "content": entry.content,
                "memory_type": entry.memory_type,
                "clean_or_contaminated": entry.clean_or_contaminated,
                "source_trial_id": entry.source_trial_id,
                "metadata": entry.metadata,
            }
            for entry in entries
        ]
    )


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _entries_by_id(entries: list[MemoryEntry]) -> dict[str, MemoryEntry]:
    by_id: dict[str, MemoryEntry] = {}
    for entry in entries:
        if entry.entry_id in by_id:
            raise ValueError(f"duplicate document_id: {entry.entry_id}")
        by_id[entry.entry_id] = entry
    return by_id


def _validate_vectors(vectors: list[list[float]], document_ids: list[str]) -> None:
    if not vectors:
        return
    normalized_dot_top_k(vectors[0], vectors, document_ids, len(vectors))
