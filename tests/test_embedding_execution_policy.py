from __future__ import annotations

import importlib
import importlib.util

import pytest

from memcontam.memory.embeddings import BgeM3EmbeddingProvider, FakeEmbeddingProvider


def _policy_module():
    spec = importlib.util.find_spec("memcontam.memory.embedding_policy")
    assert spec is not None, "embedding execution policy module is required"
    return importlib.import_module("memcontam.memory.embedding_policy")


def _run_config(*, mode: str | None, stage: str = "replay") -> dict[str, object]:
    embedding = {} if mode is None else {"mode": mode}
    return {
        "run": {
            "stage": stage,
            "execution_class": "offline_contract_replay" if stage == "replay" else "live",
            "scientific_result": False,
        },
        "baselines": ["bot_style"],
        "embedding": embedding,
    }


def _v2_rag_config() -> dict[str, object]:
    return {
        "run": {
            "stage": "replay",
            "execution_class": "offline_contract_replay",
            "scientific_result": False,
            "retry_policy_version": "baseline_fidelity_v2",
            "baseline_execution_contract_version": "baseline_fidelity_v2",
            "failure_taxonomy_version": "baseline_fidelity_v2",
            "fidelity_gate_layer": "structural",
        },
        "logging": {
            "memory_policy_version": "baseline_fidelity_v2",
            "prompt_version": "baseline_fidelity_v2",
        },
        "baselines": ["retrieval_rag"],
        "memory": {},
        "embedding": {},
    }


def test_embedding_mode_is_required_for_v2_retrieval_execution() -> None:
    _policy_module()
    from memcontam.config.resolution import validate_fidelity_contract

    with pytest.raises(ValueError, match="embedding.mode is required"):
        validate_fidelity_contract(_v2_rag_config())


def test_test_double_embedding_builds_only_for_offline_replay() -> None:
    policy = _policy_module()

    provider = policy.build_embedding_provider_for_run(_run_config(mode="test_double"))

    assert isinstance(provider, FakeEmbeddingProvider)
    with pytest.raises(ValueError, match="test_double"):
        policy.EmbeddingContract.from_config(_run_config(mode="test_double", stage="pilot"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "wrong-model"),
        ("revision", "wrong-revision"),
        ("vector_dimension", 8),
        ("normalize_embeddings", False),
    ],
)
def test_pinned_semantic_embedding_rejects_wrong_bge_metadata(field: str, value: object) -> None:
    policy = _policy_module()
    contract = policy.EmbeddingContract.from_config(_run_config(mode="pinned_semantic"))
    provider = object.__new__(BgeM3EmbeddingProvider)
    metadata = {
        "model_id": BgeM3EmbeddingProvider.MODEL_ID,
        "revision": BgeM3EmbeddingProvider.REVISION,
        "embedding_library_version": "fixture",
        "vector_dimension": BgeM3EmbeddingProvider.VECTOR_DIMENSION,
        "normalize_embeddings": BgeM3EmbeddingProvider.NORMALIZE_EMBEDDINGS,
    }
    metadata[field] = value
    provider._metadata = metadata

    with pytest.raises(ValueError, match=field):
        policy.validate_embedding_provider(provider, contract)


def test_pinned_semantic_embedding_rejects_wrong_config_metadata() -> None:
    policy = _policy_module()
    config = _run_config(mode="pinned_semantic")
    config["embedding"] = {"mode": "pinned_semantic", "model_id": "wrong-model"}

    with pytest.raises(ValueError, match="model_id"):
        policy.EmbeddingContract.from_config(config)


def test_v2_rag_requires_a_corpus_manifest_path() -> None:
    _policy_module()
    from memcontam.config.resolution import validate_fidelity_contract

    config = _v2_rag_config()
    config["embedding"] = {"mode": "test_double"}
    with pytest.raises(ValueError, match="memory.corpus_manifest_path"):
        validate_fidelity_contract(config)
