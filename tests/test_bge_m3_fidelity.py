from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from memcontam.memory import embeddings
from memcontam.memory.embedding_policy import EmbeddingContract, validate_embedding_provider
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, FakeEmbeddingProvider
from memcontam.readiness import RetrievalSmokeError, resolve_bge_cache_path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "baseline_fidelity_v2_bge_smoke.yaml"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_bge_m3_fidelity.py"


def test_f1c_config_requires_the_pinned_real_retriever() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["run"]["fidelity_gate_layer"] == "real_retriever"
    assert config["embedding"] == {
        "mode": "pinned_semantic",
        "model_id": BgeM3EmbeddingProvider.MODEL_ID,
        "revision": BgeM3EmbeddingProvider.REVISION,
        "vector_dimension": BgeM3EmbeddingProvider.VECTOR_DIMENSION,
        "normalize_embeddings": BgeM3EmbeddingProvider.NORMALIZE_EMBEDDINGS,
    }
    contract = EmbeddingContract.from_config(config)
    with pytest.raises(ValueError, match="requires BgeM3EmbeddingProvider"):
        validate_embedding_provider(FakeEmbeddingProvider(), contract)


def test_f1c_rejects_a_partial_cached_snapshot_before_model_load(tmp_path: Path) -> None:
    snapshot = (
        tmp_path
        / "models--BAAI--bge-m3"
        / "snapshots"
        / BgeM3EmbeddingProvider.REVISION
    )
    snapshot.mkdir(parents=True)
    for filename in ("config.json", "colbert_linear.pt", "sparse_linear.pt", "model.safetensors"):
        (snapshot / filename).write_text("partial", encoding="utf-8")

    with pytest.raises(RetrievalSmokeError, match="MISSING_CACHED_BGE_M3"):
        resolve_bge_cache_path({"MEMCONTAM_BGE_CACHE_DIR": str(tmp_path)})


def test_f1c_provider_loads_the_pinned_local_snapshot_path(tmp_path: Path, monkeypatch) -> None:
    snapshot = (
        tmp_path
        / "models--BAAI--bge-m3"
        / "snapshots"
        / BgeM3EmbeddingProvider.REVISION
    )
    snapshot.mkdir(parents=True)

    class LocalSentenceTransformer:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

        def encode(self, _texts, **_kwargs):  # noqa: ANN001, ANN003
            return [[1.0] * BgeM3EmbeddingProvider.VECTOR_DIMENSION]

    monkeypatch.setattr(embeddings, "SentenceTransformer", LocalSentenceTransformer)

    provider = BgeM3EmbeddingProvider(cache_folder=tmp_path, local_files_only=True)

    assert provider.model.kwargs["model_name_or_path"] == str(snapshot)
    assert provider.model.kwargs["local_files_only"] is True


def test_f1c_verifier_reports_a_cache_blocker_or_a_verified_gate() -> None:
    completed = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["overall"] in {"pass", "blocked"}
    assert report["reason_code"] is None if report["overall"] == "pass" else "missing_cached_bge_m3"
    assert {"python", "sentence_transformers", "device", "dtype", "local_files_only"} <= set(
        report["runtime"]
    )
    if report["overall"] == "blocked":
        assert completed.returncode == 1
        assert report["blocker"] == "missing_cached_bge_m3"
        assert BgeM3EmbeddingProvider.MODEL_ID in report["detail"]
    else:
        assert completed.returncode == 0
        assert report["provider_identity"] == (
            f"{BgeM3EmbeddingProvider.MODEL_ID}@{BgeM3EmbeddingProvider.REVISION}"
        )
        assert report["rag_retrieval_count"] == 3
        assert report["bot_nonempty_buffer"] is True
        assert report["calls"] > 0
