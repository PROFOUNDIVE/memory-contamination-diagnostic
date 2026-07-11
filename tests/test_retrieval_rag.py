from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.retrieval import DenseIndex
from memcontam.memory.stores import MemoryEntry


def _entry(entry_id: str, content: str, *, source: str = "fixture") -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=content,
        memory_type="strategy",
        clean_or_contaminated="clean",
        metadata={"source": source},
    )


def test_dense_index_returns_exact_stable_top_k(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    entries = [
        _entry("doc-b", "alpha beta"),
        _entry("doc-a", "alpha beta"),
        _entry("doc-c", "unrelated"),
    ]
    index = DenseIndex(entries, provider=provider, cache_dir=tmp_path)

    records = index.retrieve("alpha beta", k=10)

    assert [record.document_id for record in records] == ["doc-a", "doc-b", "doc-c"]
    assert [record.rank for record in records] == [1, 2, 3]
    assert records[0].score == records[1].score
    assert records[0].text == "alpha beta"
    assert records[0].title_or_type == "strategy"
    assert records[0].source == "fixture"
    assert records[0].corpus_hash.startswith("sha256:")
    assert records[0].embedding_model_id == provider.metadata["model_id"]
    assert records[0].embedding_revision == provider.metadata["revision"]
    assert records[0].embedding_library_version == provider.metadata["embedding_library_version"]
    assert index.retrieve("alpha beta", k=0) == []


def test_dense_index_invalidates_stale_cache(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    entries = [_entry("doc-1", "alpha")]
    DenseIndex(entries, provider=provider, cache_dir=tmp_path)
    manifest_path = tmp_path / "dense_index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_hash"] = "sha256:stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stale dense index cache"):
        DenseIndex(entries, provider=provider, cache_dir=tmp_path)


def test_dense_index_rejects_nan_vectors(tmp_path: Path) -> None:
    class BadProvider(FakeEmbeddingProvider):
        def encode_document(self, text: str) -> list[float]:
            return [float("nan"), 1.0]

    with pytest.raises(ValueError, match="contains NaN"):
        DenseIndex([_entry("doc-1", "alpha")], provider=BadProvider(2), cache_dir=tmp_path)
