from __future__ import annotations

import importlib.util
import math

import pytest

from memcontam.memory import embeddings
from memcontam.memory.embeddings import (
    BgeM3EmbeddingProvider,
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


def test_bge_m3_provider_pins_identity_dimension_and_normalization(monkeypatch) -> None:
    class PinnedSentenceTransformer:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs
            self.encode_kwargs: dict[str, object] | None = None

        def encode(self, _texts, **kwargs):  # noqa: ANN001, ANN003
            self.encode_kwargs = kwargs
            return [[1.0] * BgeM3EmbeddingProvider.VECTOR_DIMENSION]

    monkeypatch.setattr(embeddings, "SentenceTransformer", PinnedSentenceTransformer)

    provider = BgeM3EmbeddingProvider(cache_folder="cache")

    assert provider.model.kwargs["model_name_or_path"] == "BAAI/bge-m3"
    assert provider.model.kwargs["revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert provider.model.encode_kwargs == {
        "batch_size": 32,
        "normalize_embeddings": True,
        "convert_to_numpy": False,
        "show_progress_bar": False,
    }
    assert provider.metadata["vector_dimension"] == 1024
    assert provider.metadata["normalize_embeddings"] is True


def test_bge_m3_provider_rejects_wrong_dimension(monkeypatch) -> None:
    class WrongDimensionSentenceTransformer:
        def __init__(self, **_kwargs):  # noqa: ANN003
            pass

        def encode(self, _texts, **_kwargs):  # noqa: ANN001, ANN003
            return [[1.0, 0.0]]

    monkeypatch.setattr(embeddings, "SentenceTransformer", WrongDimensionSentenceTransformer)

    with pytest.raises(ValueError, match="expected dimension 1024"):
        BgeM3EmbeddingProvider()
