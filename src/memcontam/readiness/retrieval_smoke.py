from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import tempfile
import time
import tracemalloc
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import yaml  # type: ignore[import-untyped]

from memcontam.memory.embeddings import BgeM3EmbeddingProvider, normalized_dot_top_k


EXPECTED_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
PRIMARY_THRESHOLD = 0.7
QUERY_COUNT = 30
_REQUIRED_DOCUMENT_KINDS = {
    "relevant",
    "unrelated_negative",
    "hard_lexical_negative",
    "near_duplicate",
    "semantic_conflict",
}
_REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "colbert_linear.pt",
    "sparse_linear.pt",
    "1_Pooling/config.json",
)
_BOT_MATCH_CALIBRATION = (
    "Construct a valid Game24 arithmetic expression using rational intermediate values."
)
_BOT_NOVEL_CALIBRATION = "Alphabetically sort unrelated botanical specimen names."


class RetrievalSmokeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _NetworkAttempt(RuntimeError):
    pass


class _EmbeddingProvider(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]: ...

    def encode_query(self, text: str) -> list[float]: ...

    def encode_document(self, text: str) -> list[float]: ...


class _NetworkGuard:
    attempted = False


@contextmanager
def deny_network() -> Iterator[_NetworkGuard]:
    guard = _NetworkGuard()
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def deny(*_args: Any, **_kwargs: Any) -> None:
        guard.attempted = True
        raise _NetworkAttempt("NETWORK_ATTEMPT")

    socket.socket.connect = deny
    socket.create_connection = deny
    try:
        yield guard
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


def resolve_bge_cache_path(environment: Mapping[str, str] | None = None) -> Path:
    cache_value = (environment or os.environ).get("MEMCONTAM_BGE_CACHE_DIR")
    if not cache_value:
        raise RetrievalSmokeError("MISSING_CACHED_BGE_M3")
    cache_root = Path(cache_value).expanduser()
    if not cache_root.is_absolute() or not cache_root.is_dir():
        raise RetrievalSmokeError("MISSING_CACHED_BGE_M3")
    _resolve_bge_snapshot(cache_root)
    return cache_root.resolve()


def _resolve_bge_snapshot(cache_root: Path) -> Path:
    snapshot = (
        cache_root
        / f"models--{BgeM3EmbeddingProvider.MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / BgeM3EmbeddingProvider.REVISION
    )
    if not snapshot.is_dir() or any(not (snapshot / name).is_file() for name in _REQUIRED_SNAPSHOT_FILES):
        raise RetrievalSmokeError("MISSING_CACHED_BGE_M3")
    if not any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise RetrievalSmokeError("MISSING_CACHED_BGE_M3")
    return snapshot


def runtime_metadata(provider: _EmbeddingProvider | None = None) -> dict[str, object]:
    model = getattr(provider, "model", None)
    device = getattr(model, "device", "unknown")
    dtype = getattr(model, "dtype", None)
    if dtype is None:
        first_module = getattr(model, "_first_module", None)
        module = first_module() if callable(first_module) else None
        dtype = getattr(getattr(module, "auto_model", None), "dtype", "unknown")
    return {
        "python": platform.python_version(),
        "sentence_transformers": _package_version("sentence-transformers"),
        "torch": _package_version("torch"),
        "device": str(device),
        "dtype": str(dtype),
        "local_files_only": True,
    }


def validate_bge_provider(provider: _EmbeddingProvider) -> dict[str, object]:
    metadata = provider.metadata
    expected = {
        "model_id": BgeM3EmbeddingProvider.MODEL_ID,
        "revision": BgeM3EmbeddingProvider.REVISION,
        "vector_dimension": BgeM3EmbeddingProvider.VECTOR_DIMENSION,
        "normalize_embeddings": True,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RetrievalSmokeError("IDENTITY_MISMATCH")
    _validate_vector(provider.encode_query("BGE-M3 query probe"))
    _validate_vector(provider.encode_document("BGE-M3 document probe"))
    return {
        **runtime_metadata(provider),
        "provider_identity": f"{expected['model_id']}@{expected['revision']}",
    }


def run_retrieval_smoke(
    *,
    config_path: Path,
    registry_path: Path,
    thresholds: Sequence[float],
    primary_threshold: float,
    output_path: Path,
    provider: _EmbeddingProvider | None = None,
) -> dict[str, object]:
    if tuple(thresholds) != EXPECTED_THRESHOLDS:
        raise RetrievalSmokeError("THRESHOLD_GRID_MISMATCH")
    if primary_threshold != PRIMARY_THRESHOLD:
        raise RetrievalSmokeError("PRIMARY_THRESHOLD_MISMATCH")
    _validate_pilot_a_config(config_path)
    rows = _load_registry(registry_path)
    guard: _NetworkGuard | None = None
    try:
        with deny_network() as guard:
            active_provider = provider
            if active_provider is None:
                cache_root = resolve_bge_cache_path()
                active_provider = BgeM3EmbeddingProvider(cache_folder=cache_root, local_files_only=True)
            report = _evaluate(rows, active_provider, thresholds, primary_threshold, registry_path)
    except Exception as error:
        if guard is not None and guard.attempted:
            raise RetrievalSmokeError("NETWORK_ATTEMPT") from error
        if isinstance(error, RetrievalSmokeError):
            raise
        raise RetrievalSmokeError("RETRIEVAL_SMOKE_FAILED") from error
    if guard is not None and guard.attempted:
        raise RetrievalSmokeError("NETWORK_ATTEMPT")
    _write_json_atomic(output_path, report)
    return report


def _validate_pilot_a_config(path: Path) -> None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RetrievalSmokeError("INVALID_PILOT_A_CONFIG") from error
    if not isinstance(payload, dict) or payload.get("config_kind") != "phase12_pilot_a_preflight_v1":
        raise RetrievalSmokeError("INVALID_PILOT_A_CONFIG")
    if payload.get("task_family") != "game24" or payload.get("evidence_layers") != [
        "build",
        "calibration",
    ]:
        raise RetrievalSmokeError("INVALID_PILOT_A_CONFIG")


def _load_registry(path: Path) -> list[dict[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW") from error
    if len(rows) != QUERY_COUNT:
        raise RetrievalSmokeError("QUERY_COUNT_MISMATCH")
    parsed: list[dict[str, object]] = []
    query_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
        if query_id in query_ids:
            raise RetrievalSmokeError("DUPLICATE_QUERY_ID")
        query_ids.add(query_id)
        _validate_row(row)
        parsed.append(row)
    return parsed


def _validate_row(row: Mapping[str, object]) -> None:
    query_id = str(row["query_id"])
    if "main" in query_id.lower() or "extension" in query_id.lower():
        raise RetrievalSmokeError("MAIN_EXTENSION_REGISTRY_FORBIDDEN")
    if row.get("evidence_layer") not in {"build", "calibration"} or row.get("task_family") != "game24":
        raise RetrievalSmokeError("MAIN_EXTENSION_REGISTRY_FORBIDDEN")
    if row.get("branch") != row.get("index_branch"):
        raise RetrievalSmokeError("BRANCH_INDEX_MISMATCH")
    if not isinstance(row.get("query"), str) or not str(row["query"]).strip():
        raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
    documents = row.get("documents")
    if not isinstance(documents, list) or len(documents) != len(_REQUIRED_DOCUMENT_KINDS):
        raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
    document_ids: set[str] = set()
    kinds: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
        document_id = document.get("document_id")
        text = document.get("text")
        kind = document.get("kind")
        if not isinstance(document_id, str) or not document_id or document_id in document_ids:
            raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
        if not isinstance(text, str) or not text.strip() or not isinstance(kind, str):
            raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
        document_ids.add(document_id)
        kinds.add(kind)
    if kinds != _REQUIRED_DOCUMENT_KINDS:
        raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
    relevant_ids = row.get("relevant_document_ids")
    spans = row.get("answer_call_source_spans")
    if not isinstance(relevant_ids, list) or not relevant_ids or not all(
        isinstance(item, str) and item in document_ids for item in relevant_ids
    ):
        raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")
    if not isinstance(spans, list) or not spans:
        raise RetrievalSmokeError("MISSING_ANSWER_CALL_SOURCE_SPANS")
    if not all(isinstance(item, str) and item in relevant_ids for item in spans):
        raise RetrievalSmokeError("ANSWER_CALL_SOURCE_SPAN_MISMATCH")
    if not all(isinstance(row.get(key), str) and str(row[key]).strip() for key in ("bot_candidate", "bot_reference")):
        raise RetrievalSmokeError("MALFORMED_REGISTRY_ROW")


def _evaluate(
    rows: list[dict[str, object]],
    provider: _EmbeddingProvider,
    thresholds: Sequence[float],
    primary_threshold: float,
    registry_path: Path,
) -> dict[str, object]:
    runtime = validate_bge_provider(provider)
    started = time.perf_counter()
    tracemalloc.start()
    top_one_scores: list[float] = []
    all_scores: list[float] = []
    separations: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    recall_at_one = 0
    recall_at_three = 0
    bot_similarities: list[float] = []
    try:
        for row in rows:
            documents = row["documents"]
            assert isinstance(documents, list)
            document_ids = [str(document["document_id"]) for document in documents]
            vectors = [provider.encode_document(str(document["text"])) for document in documents]
            query_vector = provider.encode_query(str(row["query"]))
            _validate_vector(query_vector)
            for vector in vectors:
                _validate_vector(vector)
            ranked = normalized_dot_top_k(query_vector, vectors, document_ids, len(document_ids))
            if not ranked:
                raise RetrievalSmokeError("EMPTY_RETRIEVAL")
            if len({round(score, 12) for _, score in ranked}) == 1:
                raise RetrievalSmokeError("ALL_TIED_RANKING")
            top_three = ranked[:3]
            retrieved_ids = {document_id for document_id, _ in top_three}
            raw_relevant_ids = row["relevant_document_ids"]
            raw_source_spans = row["answer_call_source_spans"]
            assert isinstance(raw_relevant_ids, list) and isinstance(raw_source_spans, list)
            relevant_ids = {str(item) for item in raw_relevant_ids}
            source_spans = {str(item) for item in raw_source_spans}
            if not source_spans <= retrieved_ids:
                raise RetrievalSmokeError("ANSWER_CALL_SOURCE_SPAN_MISMATCH")
            top_one_scores.append(ranked[0][1])
            all_scores.extend(score for _, score in ranked)
            first_rank = next(
                (rank for rank, (document_id, _) in enumerate(top_three, start=1) if document_id in relevant_ids),
                None,
            )
            if first_rank == 1:
                recall_at_one += 1
            if first_rank is not None:
                recall_at_three += 1
                reciprocal_ranks.append(1.0 / first_rank)
                ndcgs.append(1.0 / math.log2(first_rank + 1))
            else:
                reciprocal_ranks.append(0.0)
                ndcgs.append(0.0)
            score_by_id = dict(ranked)
            best_relevant = max(score_by_id[document_id] for document_id in relevant_ids)
            best_negative = max(
                score for document_id, score in ranked if document_id not in relevant_ids
            )
            separations.append(best_relevant - best_negative)
            candidate_vector = provider.encode_query(_bot_calibration_text(str(row["bot_candidate"])))
            reference_vector = provider.encode_document(_bot_calibration_text(str(row["bot_reference"])))
            _validate_vector(candidate_vector)
            _validate_vector(reference_vector)
            bot_similarities.append(sum(a * b for a, b in zip(candidate_vector, reference_vector)))
    finally:
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    threshold_results = {
        f"{threshold:.1f}": _threshold_result(bot_similarities, threshold) for threshold in thresholds
    }
    primary = threshold_results[f"{primary_threshold:.1f}"]
    if primary["admitted"] in {0, len(rows)}:
        raise RetrievalSmokeError("DEGENERATE_ADMISSION_REGIME")
    metrics: dict[str, object] = {
        "recall_at_1": recall_at_one / len(rows),
        "recall_at_3": recall_at_three / len(rows),
        "mrr": sum(reciprocal_ranks) / len(rows),
        "ndcg_at_3": sum(ndcgs) / len(rows),
        "similarity_distributions": {
            "retrieval": _distribution(all_scores),
            "top_1": _distribution(top_one_scores),
            "bot_novelty": _distribution(bot_similarities),
        },
        "rank_separation": _distribution(separations),
        "retrieval_hit_rate": recall_at_three / len(rows),
        "bot_admission_rate": primary["admission_rate"],
        "bot_admission_growth_rate": primary["growth_rate"],
    }
    memo = {
        "provider_identity": runtime["provider_identity"],
        "query_count": len(rows),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "threshold_results": threshold_results,
        "primary_threshold": primary_threshold,
        "metrics": metrics,
    }
    return {
        "overall": "pass",
        **memo,
        "canonical_memo_sha256": _canonical_sha256(memo),
        "runtime": runtime,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "peak_memory_bytes": peak_memory,
    }


def _validate_vector(vector: list[float]) -> None:
    if len(vector) != BgeM3EmbeddingProvider.VECTOR_DIMENSION:
        raise RetrievalSmokeError("WRONG_VECTOR_DIMENSION")
    if any(not math.isfinite(value) for value in vector):
        raise RetrievalSmokeError("INVALID_VECTOR")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise RetrievalSmokeError("ZERO_VECTOR")
    if len(set(vector)) == 1:
        raise RetrievalSmokeError("CONSTANT_VECTOR")
    if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RetrievalSmokeError("NON_NORMALIZED_VECTOR")


def _threshold_result(similarities: list[float], threshold: float) -> dict[str, float | int]:
    admitted = sum(score < threshold for score in similarities)
    return {
        "threshold": threshold,
        "admitted": admitted,
        "rejected": len(similarities) - admitted,
        "admission_rate": admitted / len(similarities),
        "growth_rate": admitted / len(similarities),
    }


def _bot_calibration_text(value: str) -> str:
    if value.startswith(("BOT_HIGH", "BOT_REFERENCE_HIGH")):
        return _BOT_MATCH_CALIBRATION
    if value.startswith("BOT_LOW"):
        return _BOT_NOVEL_CALIBRATION
    return value


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (ordered[middle - 1] + ordered[middle]) / 2 if len(ordered) % 2 == 0 else ordered[middle]
    return {"min": ordered[0], "max": ordered[-1], "mean": sum(ordered) / len(ordered), "median": median}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_path).replace(path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"
