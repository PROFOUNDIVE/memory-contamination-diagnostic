from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.memory.embeddings import EmbeddingProvider
from memcontam.memory.retrieval import DenseIndex, render_retrieved_record, retrieve_records
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


def run_faithful_rag(
    task: TaskInstance,
    memory: MemoryState,
    *,
    client: LLMClient,
    model: str,
    config: dict[str, Any] | None = None,
    top_k: int | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = dict(config or {})
    k = int(top_k if top_k is not None else config.get("top_k", config.get("rag_top_k", 3)))
    memory_before = [entry.model_dump() for entry in memory.entries]

    def _run(index_cache_dir: str | Path) -> dict[str, Any]:
        index = DenseIndex(memory.entries, provider=embedding_provider, cache_dir=index_cache_dir)
        retrieved_records = index.retrieve(str(task.input), k)
        messages = [_generation_message(task, retrieved_records)]
        recorder = MethodCallRecorder(client)
        response = recorder.chat(
            messages,
            model=model,
            config={**config, "sample_id": config.get("sample_id", task.sample_id), "method_stage": "rag_generate"},
        )
        method_calls = recorder.get_records()
        if method_calls:
            method_calls[-1].retrieved_records = retrieved_records
        metadata = {
            "corpus_hash": str(index.manifest["corpus_hash"]),
            "embedding_model_id": str(index.manifest["embedding_model_id"]),
            "embedding_revision": str(index.manifest["embedding_revision"]),
            "embedding_library_version": str(index.manifest["embedding_library_version"]),
            "top_k": k,
        }
        return {
            "final_response": response.content,
            "parsed_answer": _parse_answer(response.content),
            "retrieved_records": retrieved_records,
            "method_calls": method_calls,
            "memory_before": memory_before,
            "memory_after": [entry.model_dump() for entry in memory.entries],
            "metadata": metadata,
            "memory_write_event": None,
        }

    if cache_dir is not None:
        return _run(cache_dir)
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        return _run(temp_cache_dir)


def _generation_message(task: TaskInstance, records: list[Any]) -> dict[str, str]:
    context = "\n".join(
        f"rank={record.rank} document_id={record.document_id} score={record.score:.6f}\n{record.text}"
        for record in records
    )
    return {"role": "user", "content": f"Retrieved memory:\n{context}\n\nSolve: {task.input}"}


def _parse_answer(response: str) -> str:
    response = response.strip()
    if ":" in response:
        return response.split(":", 1)[1].strip()
    return response


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
        cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return run_faithful_rag(
            task,
            memory,
            client=client,
            model=model,
            config=config,
            top_k=top_k,
            embedding_provider=embedding_provider,
            cache_dir=cache_dir,
        )

    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        retrieved = retrieve_records(str(task.input), memory.entries)
        context = "\n".join(render_retrieved_record(record) for record in retrieved)
        return [{"role": "user", "content": f"Retrieved memory:\n{context}\n\nSolve: {task.input}"}]
