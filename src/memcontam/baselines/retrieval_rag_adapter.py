from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import (
    BaselineExecutionOutcome,
    CorpusIdentity,
    ErrorType,
    FailureDisposition,
    ScientificIneligibilityReason,
)
from memcontam.baselines.retrieval_rag import NEUTRAL_SYSTEM_INSTRUCTION, RetrievalDocumentPayload
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.provenance import PromptSourcePart, build_prompt_with_sources
from memcontam.logging.schema import RetrievalRecord, VerifierResult
from memcontam.memory.embeddings import EmbeddingProvider
from memcontam.memory.retrieval import DenseIndex
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


class RetrievalRagAdapter:
    def execute(
        self,
        task: TaskInstance,
        memory: MemoryState,
        *,
        client: LLMClient,
        model: str,
        embedding_provider: EmbeddingProvider | None,
        corpus_identity: CorpusIdentity | None = None,
        config: dict[str, Any] | None = None,
        cache_dir: str | Path | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None = None,
    ) -> BaselineExecutionOutcome:
        config = dict(config or {})
        memory_before = tuple(entry.model_dump() for entry in memory.entries)
        recorder = MethodCallRecorder(
            client,
            event_callback=config.get("_logging_event_callback"),
            trial_context={
                **config.get("_logging_trial_context", {}),
                "trial_id": _trial_id(task, config, model),
            },
        )
        provider_failure = _provider_failure(
            embedding_provider,
            corpus_identity,
            task,
            require_corpus_identity=bool(config.get("_require_corpus_identity", False)),
        )
        if provider_failure is not None:
            return _failed_outcome(recorder, memory_before, memory, *provider_failure)
        if not memory.entries and config.get("_require_corpus_identity", False):
            return _failed_outcome(
                recorder,
                memory_before,
                memory,
                "CorpusContractError",
                "rag_manifest_invalid",
                "manifest_invalid",
            )
        assert embedding_provider is not None

        query = canonical_task_json(task)
        try:
            if cache_dir is None:
                with tempfile.TemporaryDirectory() as temporary_cache_dir:
                    index = _build_index(
                        memory.entries, embedding_provider, temporary_cache_dir, corpus_identity
                    )
                    records = _retrieve(index, query)
            else:
                index = _build_index(memory.entries, embedding_provider, cache_dir, corpus_identity)
                records = _retrieve(index, query)
        except _RagFailure as failure:
            return _failed_outcome(recorder, memory_before, memory, *failure.triple)
        except Exception:
            return _failed_outcome(
                recorder,
                memory_before,
                memory,
                "EmbeddingContractError",
                "rag_embedding_failed",
                "embedding_failed",
            )

        messages, source_spans = _messages(task, records, memory.entries)
        try:
            response = recorder.chat(
                messages,
                model=model,
                config={
                    **config,
                    "sample_id": config.get("sample_id", task.sample_id),
                    "method_stage": "rag_generate",
                    "source_spans": source_spans,
                },
            )
        except Exception:
            return _failed_outcome(
                recorder,
                memory_before,
                memory,
                "ProviderCallFailure",
                "provider_call_failed",
                "provider_call_failed",
                records=records,
            )
        if response is None:
            return _failed_outcome(
                recorder,
                memory_before,
                memory,
                "ProviderCallFailure",
                "provider_call_failed",
                "provider_call_failed",
                records=records,
            )

        method_calls = recorder.get_records()
        if method_calls:
            method_calls[-1].retrieved_records = records
        answer_call_id = _answer_call_id(recorder)
        try:
            parsed_answer = parse_final_answer(response.content)
        except ValueError:
            parsed_answer = ""
        if not parsed_answer:
            return _failed_outcome(
                recorder,
                memory_before,
                memory,
                "BaselineOutputError",
                "rag_invalid_final_answer",
                "invalid_final_answer",
                final_response=response.content,
                answer_call_id=answer_call_id,
                records=records,
            )
        try:
            verifier_result = _verify(verifier, parsed_answer, task)
        except Exception:
            return _failed_outcome(
                recorder,
                memory_before,
                memory,
                "VerifierContractError",
                "verifier_contract_failed",
                "verifier_contract_failed",
                final_response=response.content,
                parsed_answer=parsed_answer,
                answer_call_id=answer_call_id,
                records=records,
            )
        return BaselineExecutionOutcome(
            status="succeeded",
            final_response=response.content,
            parsed_answer=parsed_answer,
            verifier_result=verifier_result,
            answer_call_id=answer_call_id,
            method_calls=tuple(method_calls),
            memory_before=memory_before,
            memory_after=tuple(entry.model_dump() for entry in memory.entries),
            retrieved_memory=tuple(
                _entries_by_id(memory.entries)[record.document_id].model_dump()
                for record in records
            ),
            retrieved_scores=tuple(record.score for record in records),
            metadata=_metadata(index, corpus_identity),
        )


class _RagFailure(Exception):
    def __init__(
        self,
        error_type: ErrorType,
        failure_disposition: FailureDisposition,
        scientific_ineligibility_reason: ScientificIneligibilityReason,
    ) -> None:
        self.triple = (error_type, failure_disposition, scientific_ineligibility_reason)


def _retrieve(index: DenseIndex, query: str) -> list[RetrievalRecord]:
    try:
        return index.retrieve(query, 3)
    except ValueError as exc:
        if "dimension mismatch" in str(exc):
            raise _RagFailure(
                "EmbeddingContractError",
                "rag_embedding_dimension_mismatch",
                "embedding_dimension_mismatch",
            ) from exc
        raise _RagFailure(
            "RetrievalContractError", "rag_retrieval_failed", "retrieval_failed"
        ) from exc
    except Exception as exc:
        raise _RagFailure(
            "RetrievalContractError", "rag_retrieval_failed", "retrieval_failed"
        ) from exc


def _build_index(
    entries: list[MemoryEntry],
    provider: EmbeddingProvider,
    cache_dir: str | Path,
    corpus_identity: CorpusIdentity | None,
) -> DenseIndex:
    try:
        return DenseIndex(
            entries,
            provider=provider,
            cache_dir=cache_dir,
            corpus_identity=corpus_identity,
        )
    except ValueError as exc:
        message = str(exc)
        if "stale dense index cache" in message:
            raise _RagFailure(
                "CorpusContractError", "rag_manifest_invalid", "manifest_invalid"
            ) from exc
        if "dimension mismatch" in message:
            raise _RagFailure(
                "EmbeddingContractError",
                "rag_embedding_dimension_mismatch",
                "embedding_dimension_mismatch",
            ) from exc
        raise _RagFailure(
            "EmbeddingContractError", "rag_embedding_failed", "embedding_failed"
        ) from exc
    except Exception as exc:
        raise _RagFailure(
            "EmbeddingContractError", "rag_embedding_failed", "embedding_failed"
        ) from exc


def _provider_failure(
    provider: EmbeddingProvider | None,
    corpus_identity: CorpusIdentity | None,
    task: TaskInstance,
    *,
    require_corpus_identity: bool,
) -> tuple[ErrorType, FailureDisposition, ScientificIneligibilityReason] | None:
    if corpus_identity is not None and type(corpus_identity) is not CorpusIdentity:
        return "CorpusContractError", "rag_manifest_invalid", "manifest_invalid"
    if require_corpus_identity and corpus_identity is None:
        return "CorpusContractError", "rag_manifest_invalid", "manifest_invalid"
    if provider is None:
        return (
            "EmbeddingContractError",
            "rag_embedding_provider_unpinned",
            "embedding_provider_unpinned",
        )
    try:
        metadata = provider.metadata
        model_id = str(metadata["model_id"])
        revision = str(metadata["revision"])
        metadata["embedding_library_version"]
        metadata["vector_dimension"]
        if metadata.get("normalize_embeddings", True) is not True:
            raise ValueError("embedding provider must normalize vectors")
    except (KeyError, TypeError, ValueError):
        return (
            "EmbeddingContractError",
            "rag_embedding_provider_unpinned",
            "embedding_provider_unpinned",
        )
    except Exception:
        return "EmbeddingContractError", "rag_embedding_failed", "embedding_failed"
    if corpus_identity is None:
        return None
    if corpus_identity.task_family != task.task_name:
        return "CorpusContractError", "rag_manifest_invalid", "manifest_invalid"
    if corpus_identity.embedding_provider_identity != f"{model_id}@{revision}":
        return (
            "EmbeddingContractError",
            "rag_embedding_provider_unpinned",
            "embedding_provider_unpinned",
        )
    return None


def _messages(
    task: TaskInstance, records: list[RetrievalRecord], entries: list[MemoryEntry]
) -> tuple[list[dict[str, str]], list[Any]]:
    entries_by_id = _entries_by_id(entries)
    documents = [RetrievalDocumentPayload(record.text) for record in records]
    parts: list[str | PromptSourcePart] = ["Retrieved documents:\n"]
    for index, document in enumerate(documents):
        if index:
            parts.append("\n\n")
        parts.append(PromptSourcePart(document.text, entries_by_id[records[index].document_id]))
    parts.extend(["\n\nCurrent task:\n", canonical_task_json(task)])
    content, spans = build_prompt_with_sources(parts, message_index=1)
    return [
        {"role": "system", "content": NEUTRAL_SYSTEM_INSTRUCTION},
        {"role": "user", "content": content},
    ], spans


def _metadata(index: DenseIndex, corpus_identity: CorpusIdentity | None) -> dict[str, Any]:
    metadata = {
        "corpus_hash": str(index.manifest["corpus_hash"]),
        "embedding_model_id": str(index.manifest["embedding_model_id"]),
        "embedding_revision": str(index.manifest["embedding_revision"]),
        "embedding_library_version": str(index.manifest["embedding_library_version"]),
        "top_k": 3,
        "effective_k": len(index.entries) if len(index.entries) < 3 else 3,
        "similarity": "normalized_dot_product",
        "normalization": bool(index.manifest["normalize_embeddings"]),
        "retrieval_unit": "document",
        "query_serialization_version": "canonical_task_json_v1",
    }
    if corpus_identity is not None:
        metadata["corpus_identity"] = asdict(corpus_identity)
    return metadata


def _failed_outcome(
    recorder: MethodCallRecorder,
    memory_before: tuple[dict[str, Any], ...],
    memory: MemoryState,
    error_type: ErrorType,
    failure_disposition: FailureDisposition,
    scientific_ineligibility_reason: ScientificIneligibilityReason,
    *,
    final_response: str | None = None,
    parsed_answer: str | None = None,
    answer_call_id: str | None = None,
    records: list[RetrievalRecord] | None = None,
) -> BaselineExecutionOutcome:
    return BaselineExecutionOutcome(
        status="failed",
        final_response=final_response,
        parsed_answer=parsed_answer,
        answer_call_id=answer_call_id,
        method_calls=tuple(recorder.get_records()),
        memory_before=memory_before,
        memory_after=tuple(entry.model_dump() for entry in memory.entries),
        retrieved_memory=tuple(
            _entries_by_id(memory.entries)[record.document_id].model_dump()
            for record in records or []
        ),
        retrieved_scores=tuple(record.score for record in records or []),
        error_type=error_type,
        failure_disposition=failure_disposition,
        scientific_ineligibility_reason=scientific_ineligibility_reason,
    )


def _verify(
    verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None,
    parsed_answer: str,
    task: TaskInstance,
) -> bool:
    if verifier is None:
        return True
    result = verifier(parsed_answer, task)
    if isinstance(result, VerifierResult):
        return result.is_correct
    if isinstance(result, bool):
        return result
    raise TypeError("RAG verifier must return VerifierResult or bool")


def _trial_id(task: TaskInstance, config: dict[str, Any], model: str) -> str:
    return ":".join(
        [
            str(config.get("run_id", "unknown")),
            task.task_name,
            task.sample_id,
            str(config.get("baseline", "retrieval_rag")),
            str(config.get("arm", "clean")),
            str(config.get("model", model)),
        ]
    )


def _answer_call_id(recorder: MethodCallRecorder) -> str | None:
    records = recorder.get_records()
    return records[0].call_id if records else None


def _entries_by_id(entries: list[MemoryEntry]) -> dict[str, MemoryEntry]:
    return {entry.entry_id: entry for entry in entries}
