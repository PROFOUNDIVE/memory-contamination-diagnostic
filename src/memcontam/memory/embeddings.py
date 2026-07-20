from __future__ import annotations

import hashlib
import importlib
import math
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Any
from typing import Protocol

SentenceTransformer: Any | None = None


class EmbeddingProvider(Protocol):
    @property
    def metadata(self) -> dict[str, object]: ...

    def encode_query(self, text: str) -> list[float]: ...

    def encode_document(self, text: str) -> list[float]: ...


class BgeM3EmbeddingProvider:
    MODEL_ID = "BAAI/bge-m3"
    REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
    VECTOR_DIMENSION = 1024
    NORMALIZE_EMBEDDINGS = True

    def __init__(
        self,
        cache_folder: str | Path | None = None,
        *,
        local_files_only: bool = True,
        batch_size: int = 32,
    ) -> None:
        sentence_transformer = _sentence_transformer_class()
        self.cache_folder = None if cache_folder is None else str(cache_folder)
        self.batch_size = batch_size
        try:
            self.model = sentence_transformer(
                model_name_or_path=self.MODEL_ID,
                revision=self.REVISION,
                cache_folder=self.cache_folder,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            mode = "offline" if local_files_only else "local"
            raise RuntimeError(
                f"failed to load {self.MODEL_ID} at revision {self.REVISION} from cache "
                f"{self.cache_folder or '<default>'} in {mode} mode"
            ) from exc
        dimension = len(self.encode_query("dimension probe"))
        if dimension != self.VECTOR_DIMENSION:
            raise ValueError(
                f"{self.MODEL_ID} expected dimension {self.VECTOR_DIMENSION}, got {dimension}"
            )
        self._metadata = {
            "model_id": self.MODEL_ID,
            "revision": self.REVISION,
            "embedding_library_version": _package_version("sentence-transformers"),
            "vector_dimension": dimension,
            "normalize_embeddings": self.NORMALIZE_EMBEDDINGS,
        }

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def encode_query(self, text: str) -> list[float]:
        return self._encode_one(text)

    def encode_document(self, text: str) -> list[float]:
        return self._encode_one(text)

    def _encode_one(self, text: str) -> list[float]:
        encoded = self.model.encode(
            [text],
            batch_size=self.batch_size,
            normalize_embeddings=self.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        first = encoded[0]
        if hasattr(first, "tolist"):
            first = first.tolist()
        return _normalize_vector([float(value) for value in first], "embedding")


SentenceTransformerProvider = BgeM3EmbeddingProvider


class FakeEmbeddingProvider:
    def __init__(self, vector_dimension: int = 16) -> None:
        if vector_dimension <= 0:
            raise ValueError("vector_dimension must be positive")
        self._dimension = vector_dimension
        self._metadata = {
            "model_id": "fake-deterministic-embedding",
            "revision": "local",
            "embedding_library_version": "fake",
            "vector_dimension": vector_dimension,
            "normalize_embeddings": True,
        }

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def encode_query(self, text: str) -> list[float]:
        return self._encode(text)

    def encode_document(self, text: str) -> list[float]:
        return self._encode(text)

    def _encode(self, text: str) -> list[float]:
        values = []
        counter = 0
        while len(values) < self._dimension:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            values.extend((byte / 255.0) * 2.0 - 1.0 for byte in digest)
            counter += 1
        return _normalize_vector(values[: self._dimension], "fake embedding")


def normalized_dot_top_k(
    query_vector: list[float],
    document_vectors: list[list[float]],
    document_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    if k <= 0:
        return []
    if len(document_vectors) != len(document_ids):
        raise ValueError("document_vectors and document_ids must have the same length")
    query = _normalize_vector(query_vector, "query vector")
    scored = []
    for document_id, vector in zip(document_ids, document_vectors):
        if len(vector) != len(query):
            raise ValueError(
                f"dimension mismatch for document {document_id}: "
                f"query has {len(query)}, document has {len(vector)}"
            )
        document = _normalize_vector(vector, f"document vector {document_id}")
        scored.append((document_id, sum(a * b for a, b in zip(query, document))))
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:k]


def _normalize_vector(vector: list[float], label: str) -> list[float]:
    if not vector:
        raise ValueError(f"{label} must not be empty")
    for value in vector:
        if math.isnan(value):
            raise ValueError(f"{label} contains NaN")
        if not math.isfinite(value):
            raise ValueError(f"{label} contains non-finite value")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError(f"{label} must not be zero")
    return [value / norm for value in vector]


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _sentence_transformer_class() -> Any:
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
    return SentenceTransformer
