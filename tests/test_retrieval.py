from __future__ import annotations

import pytest

from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.retrieval import DenseIndex
from memcontam.memory.stores import MemoryEntry


def _entry() -> MemoryEntry:
    return MemoryEntry(entry_id="doc-1", content="strategy", memory_type="strategy")


def test_dense_index_requires_an_explicit_embedding_provider(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit embedding provider"):
        DenseIndex([_entry()], cache_dir=tmp_path)


def test_dense_index_rejects_provider_without_normalized_embeddings(tmp_path) -> None:
    class UnnormalizedProvider(FakeEmbeddingProvider):
        @property
        def metadata(self) -> dict[str, object]:
            return {**super().metadata, "normalize_embeddings": False}

    with pytest.raises(ValueError, match="normalized embeddings"):
        DenseIndex([_entry()], provider=UnnormalizedProvider(), cache_dir=tmp_path)
