from __future__ import annotations

import json
from hashlib import sha256
from typing import TypedDict, assert_never

from memcontam.experiment.phase12.filter_challenge.executor_types import (
    BoTExecutionRequest,
    FullHistoryExecutionRequest,
    NativeExecutionRequest,
    PairExecutorError,
    RagFrozenExecutionRequest,
    ReflexionExecutionRequest,
    SourceSnapshot,
)
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint, deserialize_checkpoint


class _RagSnapshotPayload(TypedDict):
    branch: str
    corpus_content_hash: str
    corpus_documents: list[dict[str, str]]
    active_document_ids: list[str]
    corpus_serialization_id: str
    corpus_version: str
    index_artifact_hash: str
    index_document_ids: list[str]
    index_serialization_id: str
    index_version: str


def source_snapshot(execution: NativeExecutionRequest) -> SourceSnapshot:
    match execution:
        case FullHistoryExecutionRequest(native_request=request):
            return _checkpoint_snapshot(request.checkpoint)
        case RagFrozenExecutionRequest(
            source_checkpoint_id=checkpoint_id, native_request=request
        ):
            corpus = request.source_state.corpus
            index = request.source_state.index
            if corpus is None or index is None:
                raise PairExecutorError("MISSING_RAG_SOURCE")
            payload = _RagSnapshotPayload(
                branch=request.source_state.branch,
                corpus_content_hash=corpus.content_hash,
                corpus_documents=[document.payload() for document in corpus.documents],
                active_document_ids=list(corpus.active_document_ids),
                corpus_serialization_id=corpus.serialization_id,
                corpus_version=corpus.corpus_version,
                index_artifact_hash=index.artifact_hash,
                index_document_ids=[document.document_id for document in index.documents],
                index_serialization_id=index.serialization_id,
                index_version=index.index_version,
            )
            canonical_bytes = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()
            digest = sha256(canonical_bytes).hexdigest()
            return SourceSnapshot(
                checkpoint_id,
                canonical_bytes,
                digest,
                canonical_bytes,
                digest,
            )
        case BoTExecutionRequest(control=control):
            return _checkpoint_snapshot(control.checkpoint)
        case ReflexionExecutionRequest(source_checkpoint=checkpoint):
            return _checkpoint_snapshot(checkpoint)
        case unreachable:
            assert_never(unreachable)


def _checkpoint_snapshot(checkpoint: Phase12Checkpoint) -> SourceSnapshot:
    deserialize_checkpoint(checkpoint)
    digest = sha256(checkpoint.canonical_bytes).hexdigest()
    if digest != checkpoint.canonical_sha256:
        raise PairExecutorError("SOURCE_HASH_MISMATCH")
    return SourceSnapshot(
        checkpoint.identity.checkpoint_id,
        checkpoint.canonical_bytes,
        checkpoint.canonical_sha256,
        checkpoint.canonical_bytes,
        checkpoint.canonical_sha256,
    )
