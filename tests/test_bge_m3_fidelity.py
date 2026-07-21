from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from memcontam.memory.embedding_policy import EmbeddingContract, validate_embedding_provider
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, FakeEmbeddingProvider


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
