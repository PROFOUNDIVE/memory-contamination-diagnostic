from __future__ import annotations

from pathlib import Path
import hashlib
import json

from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.readiness.phase13_readiness0_live_models import F1CRegistry, F1CRuntimeMetadata
from memcontam.readiness.retrieval_smoke import (
    RetrievalSmokeError,
    deny_network,
    resolve_bge_cache_path,
    validate_bge_provider,
)


class F1CRuntimeError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def verify_f1c_runtime(registry: F1CRegistry, cache_root: Path) -> F1CRuntimeMetadata:
    guard = None
    try:
        with deny_network() as guard:
            resolved = resolve_bge_cache_path(
                {registry.cache_environment_variable: str(cache_root)}
            )
            provider = BgeM3EmbeddingProvider(cache_folder=resolved, local_files_only=True)
            report = validate_bge_provider(provider)
    except (RetrievalSmokeError, RuntimeError, ValueError) as error:
        if guard is not None and guard.attempted:
            raise F1CRuntimeError("READINESS0_F1C_NETWORK_ATTEMPT") from error
        raise F1CRuntimeError("READINESS0_F1C_RUNTIME_FAILED") from error
    return F1CRuntimeMetadata.model_validate(
        {
            **report,
            "vector_dimension": registry.vector_dimension,
            "normalize_embeddings": registry.normalize_embeddings,
            "network_attempts": 0,
            "runtime_hash": _runtime_hash(
                {
                    **report,
                    "vector_dimension": registry.vector_dimension,
                    "normalize_embeddings": registry.normalize_embeddings,
                    "network_attempts": 0,
                }
            ),
        }
    )


def _runtime_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


__all__ = ["F1CRuntimeError", "verify_f1c_runtime"]
