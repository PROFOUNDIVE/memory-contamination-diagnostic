from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterable

from memcontam.baselines.contracts import CorpusIdentity
from memcontam.clients.base import LLMClient
from memcontam.logging.schema import VerifierResult
from memcontam.memory.embeddings import EmbeddingProvider
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


NEUTRAL_SYSTEM_INSTRUCTION = "Use the retrieved text only as neutral context for the current task."


@dataclass(frozen=True)
class RetrievalDocumentPayload:
    text: str


def render_retrieved_documents(documents: Iterable[RetrievalDocumentPayload]) -> str:
    return "\n\n".join(document.text for document in documents)


class RetrievalRagPolicy:
    def run(
        self,
        task: TaskInstance,
        memory: MemoryState,
        *,
        client: LLMClient,
        model: str,
        config: dict[str, Any] | None = None,
        top_k: int | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        corpus_identity: CorpusIdentity | None = None,
        cache_dir: str | Path | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None = None,
    ) -> dict[str, Any]:
        adapter = import_module("memcontam.baselines.retrieval_rag_adapter").RetrievalRagAdapter()
        outcome = adapter.execute(
            task,
            memory,
            client=client,
            model=model,
            config={**(config or {}), "top_k": top_k},
            embedding_provider=embedding_provider,
            corpus_identity=corpus_identity,
            cache_dir=cache_dir,
            verifier=verifier,
        )
        records = list(outcome.method_calls[0].retrieved_records) if outcome.method_calls else []
        return {
            "status": outcome.status,
            "final_response": outcome.final_response,
            "parsed_answer": outcome.parsed_answer,
            "verifier_result": outcome.verifier_result,
            "method_calls": list(outcome.method_calls),
            "memory_before": list(outcome.memory_before),
            "memory_after": list(outcome.memory_after),
            "memory_write_event": outcome.memory_write_event,
            "metadata": outcome.metadata,
            "retrieved_records": records,
            "retrieved_scores": list(outcome.retrieved_scores),
            "answer_call_id": outcome.answer_call_id,
            "error_type": outcome.error_type,
            "failure_disposition": outcome.failure_disposition,
            "scientific_ineligibility_reason": outcome.scientific_ineligibility_reason,
        }


def __getattr__(name: str) -> Any:
    if name == "RetrievalRagAdapter":
        return import_module("memcontam.baselines.retrieval_rag_adapter").RetrievalRagAdapter
    raise AttributeError(name)
