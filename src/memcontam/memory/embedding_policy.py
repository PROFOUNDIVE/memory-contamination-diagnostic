from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from memcontam.memory.embeddings import (
    BgeM3EmbeddingProvider,
    EmbeddingProvider,
    FakeEmbeddingProvider,
)


class EmbeddingExecutionMode(str, Enum):
    TEST_DOUBLE = "test_double"
    PINNED_SEMANTIC = "pinned_semantic"


@dataclass(frozen=True)
class EmbeddingContract:
    mode: EmbeddingExecutionMode
    model_id: str | None = None
    revision: str | None = None
    vector_dimension: int | None = None
    normalize_embeddings: bool | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EmbeddingContract":
        embedding = config.get("embedding", {})
        if not isinstance(embedding, Mapping):
            raise ValueError("embedding must be a mapping")
        raw_mode = embedding.get("mode")
        if raw_mode is None:
            raise ValueError("embedding.mode is required")
        try:
            mode = EmbeddingExecutionMode(raw_mode)
        except ValueError as exc:
            raise ValueError("embedding.mode must be test_double or pinned_semantic") from exc
        _validate_execution_mode(config, mode)
        if mode is EmbeddingExecutionMode.TEST_DOUBLE:
            return cls(mode)

        expected = {
            "model_id": BgeM3EmbeddingProvider.MODEL_ID,
            "revision": BgeM3EmbeddingProvider.REVISION,
            "vector_dimension": BgeM3EmbeddingProvider.VECTOR_DIMENSION,
            "normalize_embeddings": BgeM3EmbeddingProvider.NORMALIZE_EMBEDDINGS,
        }
        aliases = {
            "vector_dimension": ("vector_dimension", "dimension"),
            "normalize_embeddings": ("normalize_embeddings", "normalization"),
        }
        for field, expected_value in expected.items():
            names = aliases.get(field, (field,))
            configured = next(
                (embedding[name] for name in names if name in embedding), expected_value
            )
            if configured != expected_value:
                raise ValueError(f"pinned_semantic embedding.{field} must be {expected_value!r}")
        return cls(mode, **expected)


def validate_embedding_execution_policy(
    config: Mapping[str, Any], *, require_mode: bool
) -> EmbeddingContract | None:
    embedding = config.get("embedding", {})
    if not isinstance(embedding, Mapping):
        raise ValueError("embedding must be a mapping")
    requires_embeddings = bool(
        {"retrieval_rag", "bot_style", "dynamic_cheatsheet_rs_optional"}
        & set(config.get("baselines", []))
    )
    if "mode" not in embedding:
        if require_mode and requires_embeddings:
            raise ValueError("embedding.mode is required")
        return None
    contract = EmbeddingContract.from_config(config)
    if "retrieval_rag" in config.get("baselines", []):
        manifest_path = config.get("memory", {}).get("corpus_manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            raise ValueError(
                "memory.corpus_manifest_path is required when retrieval_rag is configured"
            )
    return contract


def build_embedding_provider_for_run(config: Mapping[str, Any]) -> EmbeddingProvider:
    contract = EmbeddingContract.from_config(config)
    if contract.mode is EmbeddingExecutionMode.TEST_DOUBLE:
        provider: EmbeddingProvider = FakeEmbeddingProvider()
    else:
        embedding = config["embedding"]
        provider = BgeM3EmbeddingProvider(
            cache_folder=embedding.get("cache_path"),
            local_files_only=True,
        )
    validate_embedding_provider(provider, contract)
    return provider


def validate_embedding_provider(provider: EmbeddingProvider, contract: EmbeddingContract) -> None:
    if contract.mode is EmbeddingExecutionMode.TEST_DOUBLE:
        if not isinstance(provider, FakeEmbeddingProvider):
            raise ValueError("test_double embedding mode requires FakeEmbeddingProvider")
        return
    if not isinstance(provider, BgeM3EmbeddingProvider):
        raise ValueError("pinned_semantic embedding mode requires BgeM3EmbeddingProvider")
    for field, expected in {
        "model_id": contract.model_id,
        "revision": contract.revision,
        "vector_dimension": contract.vector_dimension,
        "normalize_embeddings": contract.normalize_embeddings,
    }.items():
        if provider.metadata.get(field) != expected:
            raise ValueError(
                f"pinned_semantic provider {field} does not match the embedding contract"
            )


def embedding_provider_identity(provider: EmbeddingProvider) -> str:
    metadata = provider.metadata
    return f"{metadata['model_id']}@{metadata['revision']}"


def _validate_execution_mode(config: Mapping[str, Any], mode: EmbeddingExecutionMode) -> None:
    if mode is not EmbeddingExecutionMode.TEST_DOUBLE:
        return
    run = config.get("run", {})
    if not isinstance(run, Mapping):
        raise ValueError("run must be a mapping")
    live_smoke_enabled = bool(config.get("live_smoke", {}).get("enabled", False))
    stage = run.get("stage", "pilot" if live_smoke_enabled else "replay")
    execution_class = run.get(
        "execution_class", "live" if live_smoke_enabled else "offline_contract_replay"
    )
    if (
        stage != "replay"
        or execution_class != "offline_contract_replay"
        or run.get("scientific_result", False) is not False
    ):
        raise ValueError(
            "embedding.mode=test_double is allowed only for non-scientific offline replay"
        )
