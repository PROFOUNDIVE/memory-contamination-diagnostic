from __future__ import annotations

import json
import math
import re
import socket
from pathlib import Path

import pytest

from memcontam.readiness import PRIMARY_THRESHOLD, RetrievalSmokeError, run_retrieval_smoke
from memcontam.readiness.retrieval_smoke import _EmbeddingProvider


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase12" / "pilot_a_game24_minimal.yaml"
REGISTRY = ROOT / "data" / "phase12" / "readiness" / "game24_micro_retrieval.jsonl"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


class FakeBgeM3:
    metadata = {
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "vector_dimension": 1024,
        "normalize_embeddings": True,
        "embedding_library_version": "fake",
    }

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text)

    def encode_document(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * 1024
        if text.startswith(("BOT_HIGH", "BOT_REFERENCE_HIGH", "Construct a valid Game24")):
            vector[900] = 1.0
        elif text.startswith(("BOT_LOW", "Alphabetically sort unrelated")):
            vector[901] = 1.0
        elif text.startswith("BOT_REFERENCE_LOW"):
            vector[902] = 1.0
        else:
            match = re.search(r"q(\d{2})", text)
            vector[int(match.group(1)) if match is not None else 999] = 1.0
            if text.startswith("NEAR"):
                vector[950] = 0.2
            elif text.startswith("HARD"):
                vector[950] = 0.7
            elif text.startswith("CONFLICT"):
                vector[951] = 0.8
            elif text.startswith("UNRELATED"):
                vector[952] = 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]


def _rows() -> list[dict[str, object]]:
    return [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines()]


def _run(
    tmp_path: Path, provider: _EmbeddingProvider, rows: list[dict[str, object]] | None = None
) -> dict:
    registry = tmp_path / "registry.jsonl"
    registry.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in (rows or _rows())) + "\n",
        encoding="utf-8",
    )
    return run_retrieval_smoke(
        config_path=CONFIG,
        registry_path=registry,
        thresholds=THRESHOLDS,
        primary_threshold=PRIMARY_THRESHOLD,
        output_path=tmp_path / "memo.json",
        provider=provider,
    )


def test_micro_retrieval_memo_is_complete_and_hash_frozen(tmp_path: Path) -> None:
    report = _run(tmp_path, FakeBgeM3())

    assert report["overall"] == "pass"
    assert report["query_count"] == 30
    assert report["primary_threshold"] == 0.7
    assert set(report["threshold_results"]) == {"0.5", "0.6", "0.7", "0.8"}
    assert report["metrics"]["recall_at_1"] == 1.0
    assert report["metrics"]["recall_at_3"] == 1.0
    assert report["metrics"]["bot_admission_growth_rate"] == 0.5
    assert report["registry_sha256"]
    assert report["canonical_memo_sha256"]
    assert report["runtime"]["provider_identity"] == "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181"
    assert report["latency_ms"] >= 0
    assert report["peak_memory_bytes"] > 0


def test_micro_retrieval_embeds_semantic_bot_calibration_controls(tmp_path: Path) -> None:
    class RecordingBgeM3(FakeBgeM3):
        def __init__(self) -> None:
            self.queries: list[str] = []

        def encode_query(self, text: str) -> list[float]:
            self.queries.append(text)
            return super().encode_query(text)

    provider = RecordingBgeM3()
    _run(tmp_path, provider)

    assert "Alphabetically sort unrelated botanical specimen names." in provider.queries


@pytest.mark.parametrize(
    ("mutate", "provider", "code"),
    [
        (lambda rows: rows.__setitem__(-1, dict(rows[0])), FakeBgeM3(), "DUPLICATE_QUERY_ID"),
        (
            lambda rows: rows[0].__setitem__("answer_call_source_spans", []),
            FakeBgeM3(),
            "MISSING_ANSWER_CALL_SOURCE_SPANS",
        ),
        (
            lambda rows: rows[0].__setitem__("index_branch", "filter"),
            FakeBgeM3(),
            "BRANCH_INDEX_MISMATCH",
        ),
    ],
)
def test_micro_retrieval_rejects_invalid_registry_rows(tmp_path, mutate, provider, code) -> None:
    rows = _rows()
    mutate(rows)

    with pytest.raises(RetrievalSmokeError, match=code):
        _run(tmp_path, provider, rows)


def test_micro_retrieval_rejects_vector_and_network_degeneracy(tmp_path: Path) -> None:
    class ZeroVector(FakeBgeM3):
        def encode_query(self, text: str) -> list[float]:
            del text
            return [0.0] * 1024

    class NetworkAttempt(FakeBgeM3):
        def encode_query(self, text: str) -> list[float]:
            del text
            socket.create_connection(("127.0.0.1", 1))
            return []

    with pytest.raises(RetrievalSmokeError, match="ZERO_VECTOR"):
        _run(tmp_path, ZeroVector())
    with pytest.raises(RetrievalSmokeError, match="NETWORK_ATTEMPT"):
        _run(tmp_path, NetworkAttempt())


def test_micro_retrieval_rejects_threshold_retuning(tmp_path: Path) -> None:
    with pytest.raises(RetrievalSmokeError, match="PRIMARY_THRESHOLD_MISMATCH"):
        run_retrieval_smoke(
            config_path=CONFIG,
            registry_path=REGISTRY,
            thresholds=THRESHOLDS,
            primary_threshold=0.6,
            output_path=tmp_path / "memo.json",
            provider=FakeBgeM3(),
        )
