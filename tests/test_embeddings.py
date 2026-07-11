from __future__ import annotations

import importlib.util
import math

import pytest

from memcontam.memory import embeddings
from memcontam.memory.embeddings import (
    FakeEmbeddingProvider,
    SentenceTransformerProvider,
    normalized_dot_top_k,
)


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_fake_embedding_provider_is_normalized_and_deterministic() -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)

    first = provider.encode_query("same query")
    second = provider.encode_query("same query")
    document = provider.encode_document("same query")

    assert first == second
    assert document == first
    assert _norm(first) == pytest.approx(1.0)
    assert provider.metadata["model_id"] == "fake-deterministic-embedding"
    assert provider.metadata["revision"] == "local"
    assert provider.metadata["embedding_library_version"] == "fake"
    assert provider.metadata["vector_dimension"] == 8


def test_normalized_dot_top_k_ranks_by_score_then_id() -> None:
    results = normalized_dot_top_k(
        [1.0, 0.0],
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ["b", "a", "c"],
        k=2,
    )

    assert results == [("a", 1.0), ("b", 1.0)]


@pytest.mark.parametrize(
    ("query", "documents", "ids", "message"),
    [
        ([], [[1.0]], ["doc"], "query vector must not be empty"),
        ([math.nan], [[1.0]], ["doc"], "query vector contains NaN"),
        ([1.0, 0.0], [[1.0]], ["doc"], "dimension mismatch"),
        ([1.0], [[1.0], [1.0]], ["doc"], "document_vectors and document_ids"),
    ],
)
def test_normalized_dot_top_k_rejects_invalid_vectors(query, documents, ids, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalized_dot_top_k(query, documents, ids, k=1)


def test_real_provider_offline_missing_cache_is_explicit(tmp_path, monkeypatch) -> None:
    class MissingCacheSentenceTransformer:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs
            raise OSError("cache miss")

    monkeypatch.setattr(embeddings, "SentenceTransformer", MissingCacheSentenceTransformer)

    with pytest.raises(RuntimeError) as excinfo:
        SentenceTransformerProvider(cache_folder=tmp_path, local_files_only=True)

    message = str(excinfo.value)
    assert SentenceTransformerProvider.MODEL_ID in message
    assert SentenceTransformerProvider.REVISION in message
    assert str(tmp_path) in message
    assert "offline" in message


def test_cached_sentence_transformer_provider_smoke(tmp_path) -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers is not installed")

    try:
        provider = SentenceTransformerProvider(cache_folder=tmp_path, local_files_only=True)
    except RuntimeError as exc:
        if "offline" in str(exc):
            pytest.skip(str(exc))
        raise

    vector = provider.encode_query("cached model smoke")
    assert _norm(vector) == pytest.approx(1.0)
    assert provider.metadata["vector_dimension"] == len(vector)
