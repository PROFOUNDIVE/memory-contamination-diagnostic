from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.readiness import phase13_new_mcq_rag
from memcontam.readiness import phase13_new_mcq_rag_artifacts
from memcontam.readiness import phase13_new_mcq_rag_frozen
from memcontam.readiness.phase13_new_mcq_bge import verify_runtime_binding


PACKAGE_ROOT = Path("data/phase13/rag/new_mcq")
EVALUATION_ROOT = Path("data/phase13/core/materialized")


class _BgeIdentityProvider:
    metadata = {
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "vector_dimension": 1024,
        "normalize_embeddings": True,
    }

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, *([0.0] * 1023)]

    def encode_document(self, text: str) -> list[float]:
        del text
        raise AssertionError("frozen vectors must not be recomputed")


class _RuntimeProvider(BgeM3EmbeddingProvider):
    def __init__(self, cache_folder: Path | None = None) -> None:
        self.cache_folder = None if cache_folder is None else str(cache_folder)
        self.batch_size = 32
        self.model = SimpleNamespace(
            tokenizer=SimpleNamespace(truncation_side="right", padding_side="right"),
            max_seq_length=8192,
            device="cpu",
        )
        self._metadata = {
            "model_id": self.MODEL_ID,
            "revision": self.REVISION,
            "vector_dimension": self.VECTOR_DIMENSION,
            "normalize_embeddings": self.NORMALIZE_EMBEDDINGS,
        }

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, *([0.0] * 1023)]


def test_frozen_clean_index_reconstructs_without_intervention_state() -> None:
    frozen = phase13_new_mcq_rag_frozen._load_frozen_clean_state_for_test(
        PACKAGE_ROOT,
        "mmlu_pro_engineering",
        _BgeIdentityProvider(),
    )

    assert frozen.state.branch == "clean"
    assert frozen.state.corpus is not None
    assert frozen.state.index is not None
    assert len(frozen.state.corpus.active_documents) == 24
    assert frozen.state.index.artifact_hash == frozen.index_artifact_hash
    assert frozen.reconstruction_identity


def test_frozen_non_clean_index_reconstructs_with_one_h2_intervention() -> None:
    frozen = phase13_new_mcq_rag_frozen._load_frozen_rag_state_for_test(
        PACKAGE_ROOT,
        "mmlu_pro_engineering",
        "contam",
        _BgeIdentityProvider(),
    )

    assert frozen.state.branch == "contam"
    assert frozen.state.corpus is not None
    assert frozen.state.index is not None
    assert len(frozen.state.corpus.active_documents) == 25
    assert frozen.state.index.documents[-1].document_id.endswith("::contam")
    assert frozen.state.index.artifact_hash == frozen.index_artifact_hash


def test_frozen_clean_index_rejects_test_embedder_without_explicit_override() -> None:
    with pytest.raises(
        phase13_new_mcq_rag_frozen.FrozenArtifactError,
        match="NEW_MCQ_RAG_RUNTIME_IDENTITY_INVALID",
    ):
        phase13_new_mcq_rag.load_new_mcq_clean_rag_state(
            PACKAGE_ROOT,
            EVALUATION_ROOT,
            "mmlu_pro_engineering",
            _BgeIdentityProvider(),
        )


def test_frozen_clean_index_blocks_unverified_snapshot_without_test_override() -> None:
    with pytest.raises(
        phase13_new_mcq_rag_frozen.FrozenArtifactError,
        match="NEW_MCQ_RAG_RUNTIME_SNAPSHOT_UNVERIFIED",
    ):
        phase13_new_mcq_rag.load_new_mcq_clean_rag_state(
            PACKAGE_ROOT,
            EVALUATION_ROOT,
            "mmlu_pro_engineering",
            _BgeIdentityProvider(),
            allow_test_embedder=True,
        )


def test_runtime_materializer_binds_measured_snapshot_and_query(tmp_path: Path) -> None:
    snapshot = (
        tmp_path
        / "models--BAAI--bge-m3"
        / "snapshots"
        / "5617a9f61b028005a4858fdac845db406aefb181"
    )
    (snapshot / "1_Pooling").mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
    provider = _RuntimeProvider(tmp_path)
    artifact = phase13_new_mcq_rag_artifacts.runtime(tmp_path, provider)
    snapshot_tree = artifact["model_snapshot_tree"]
    query_verification = artifact["production_query_snapshot_verification"]

    assert artifact["status"] == "COMPLETE"
    assert artifact["missing_objects"] == []
    assert isinstance(snapshot_tree, dict)
    assert snapshot_tree["files"] == [
        {
            "path": "1_Pooling/config.json",
            "sha256": hashlib.sha256(b"{}").hexdigest(),
            "size": 2,
        },
        {
            "path": "tokenizer.json",
            "sha256": hashlib.sha256(b"{}").hexdigest(),
            "size": 2,
        },
    ]
    assert isinstance(query_verification, dict)
    assert query_verification["vector_dimension"] == 1024
    (tmp_path / "embedding_runtime_v1.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )
    verify_runtime_binding(tmp_path, provider)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("probe_id", "unregistered-probe"),
        ("vector_sha256", "not-a-sha256"),
    ],
)
def test_runtime_validation_rejects_malformed_query_attestation(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    runtime = json.loads((PACKAGE_ROOT / "embedding_runtime_v1.json").read_bytes())
    runtime["production_query_snapshot_verification"][field] = value
    (tmp_path / "embedding_runtime_v1.json").write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(ValueError, match="NEW_MCQ_RAG_EMBEDDING_RUNTIME_INVALID"):
        phase13_new_mcq_rag_frozen.validate_runtime_artifact(tmp_path)


def test_runtime_validation_rejects_malformed_snapshot_entry(tmp_path: Path) -> None:
    runtime = json.loads((PACKAGE_ROOT / "embedding_runtime_v1.json").read_bytes())
    runtime["model_snapshot_tree"]["files"][0]["path"] = "../outside"
    (tmp_path / "embedding_runtime_v1.json").write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(ValueError, match="NEW_MCQ_RAG_EMBEDDING_RUNTIME_INVALID"):
        phase13_new_mcq_rag_frozen.validate_runtime_artifact(tmp_path)
