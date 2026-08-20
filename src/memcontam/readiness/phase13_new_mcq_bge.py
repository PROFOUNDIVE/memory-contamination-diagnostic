from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import TypeAlias

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY


_QUERY_PROBE = "phase13 bge-m3 production query probe v1"
_QUERY_PROBE_ID = "new_mcq_rag_bge_m3_query_probe_v1"
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class BgeRuntimeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def validate_runtime_artifact(root: Path) -> None:
    runtime = json.loads((root / "embedding_runtime_v1.json").read_bytes())
    tree = runtime.get("model_snapshot_tree")
    query = runtime.get("production_query_snapshot_verification")
    required = {
        "schema_version": "new_mcq_rag_embedding_runtime_v1",
        "status": "COMPLETE",
        "model_id": BgeM3EmbeddingProvider.MODEL_ID,
        "revision": BgeM3EmbeddingProvider.REVISION,
        "production_identity": BGE_M3_PRIMARY_IDENTITY,
        "vector_dimension": 1024,
        "normalize_embeddings": True,
        "similarity": "cosine",
        "top_k": 3,
        "reranker": None,
        "score_threshold": None,
        "tie_break": "lexical_document_id",
        "corpus_scope": "same_task_only",
        "update_mode": "frozen_read_only",
        "embedding_implementation_sha256": _sha256(_SOURCE_ROOT / "memory" / "embeddings.py"),
        "index_implementation_sha256": _sha256(_SOURCE_ROOT / "rag" / "branch_index.py"),
    }
    if (
        runtime.get("missing_objects") != []
        or not isinstance(tree, dict)
        or set(tree) != {"files", "sha256"}
        or not _valid_snapshot_files(tree.get("files"))
        or tree.get("sha256") != canonical_json_hash(tree["files"])
        or not _is_sha256(tree.get("sha256"))
        or runtime.get("model_snapshot_tree_sha256") != tree.get("sha256")
        or not isinstance(query, dict)
        or set(query) != {"probe_id", "probe_sha256", "vector_dimension", "vector_sha256"}
        or query.get("probe_id") != _QUERY_PROBE_ID
        or query.get("probe_sha256") != canonical_json_hash(_QUERY_PROBE)
        or query.get("vector_dimension") != 1024
        or not _is_sha256(query.get("vector_sha256"))
        or any(runtime.get(key) != value for key, value in required.items())
    ):
        raise BgeRuntimeError("NEW_MCQ_RAG_EMBEDDING_RUNTIME_INVALID")


def verify_runtime_binding(root: Path, provider: BgeM3EmbeddingProvider) -> None:
    validate_runtime_artifact(root)
    runtime = json.loads((root / "embedding_runtime_v1.json").read_bytes())
    if provider.cache_folder is None:
        raise BgeRuntimeError("NEW_MCQ_RAG_RUNTIME_SNAPSHOT_UNVERIFIED")
    snapshot = (
        Path(provider.cache_folder)
        / "models--BAAI--bge-m3"
        / "snapshots"
        / BgeM3EmbeddingProvider.REVISION
    )
    if not snapshot.is_dir():
        raise BgeRuntimeError("NEW_MCQ_RAG_RUNTIME_SNAPSHOT_UNVERIFIED")
    files = [
        {
            "path": path.relative_to(snapshot).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    ]
    query = runtime.get("production_query_snapshot_verification")
    vector = provider.encode_query(_QUERY_PROBE)
    if (
        not files
        or not isinstance(query, dict)
        or runtime.get("model_snapshot_tree")
        != {"files": files, "sha256": canonical_json_hash(files)}
        or len(vector) != BgeM3EmbeddingProvider.VECTOR_DIMENSION
        or query.get("vector_sha256") != canonical_json_hash(vector)
    ):
        raise BgeRuntimeError("NEW_MCQ_RAG_RUNTIME_SNAPSHOT_UNVERIFIED")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: JsonValue) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _valid_snapshot_files(value: JsonValue) -> bool:
    if not isinstance(value, list) or not value:
        return False
    paths: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            return False
        path = item.get("path")
        size = item.get("size")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not _is_sha256(item.get("sha256"))
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            return False
        paths.append(path)
    return paths == sorted(set(paths))


__all__ = ["BgeRuntimeError", "validate_runtime_artifact", "verify_runtime_binding"]
