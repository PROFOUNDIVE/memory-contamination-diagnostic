from __future__ import annotations

from pathlib import Path

import pytest

from memcontam.baselines import retrieval_rag
from memcontam.baselines.retrieval_rag import RetrievalRagPolicy
from memcontam.clients.base import LLMResponse
from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.retrieval import DenseIndex
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


class _FakeClient:
    def __init__(self, content: str = "final: 24") -> None:
        self.content = content
        self.calls: list[tuple[list[dict[str, str]], str, dict]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        self.calls.append((messages, model, config))
        return LLMResponse(
            content=self.content,
            raw={"replay": True},
            token_usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            latency_ms=11,
        )


def _entry(entry_id: str, content: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=content,
        memory_type="strategy",
        clean_or_contaminated="clean",
        metadata={"source": "fixture"},
    )


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="game24_1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    )


def test_faithful_rag_prompt_matches_logged_top_k(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    entries = [
        _entry("doc-a", "Use multiplication before addition."),
        _entry("doc-b", "Try factor pairs that reach twenty four."),
        _entry("doc-c", "Sort words alphabetically."),
    ]
    memory = MemoryState(entries=entries)
    expected_records = DenseIndex(entries, provider=provider, cache_dir=tmp_path / "expected").retrieve(
        str(_task().input), 2
    )
    client = _FakeClient()

    result = RetrievalRagPolicy().run(
        _task(),
        memory,
        client=client,
        model="replay-model",
        config={"top_k": 2, "temperature": 0.0, "sample_id": "game24_1"},
        embedding_provider=provider,
        cache_dir=tmp_path / "actual",
    )

    assert result["final_response"] == "final: 24"
    assert result["parsed_answer"] == "24"
    assert result["retrieved_records"] == expected_records
    assert result["memory_before"] == [entry.model_dump() for entry in entries]
    assert result["memory_after"] == result["memory_before"]
    assert result["memory_write_event"] is None
    assert result["metadata"] == {
        "corpus_hash": expected_records[0].corpus_hash,
        "embedding_model_id": provider.metadata["model_id"],
        "embedding_revision": provider.metadata["revision"],
        "embedding_library_version": provider.metadata["embedding_library_version"],
        "top_k": 2,
    }

    assert len(client.calls) == 1
    messages, model, config = client.calls[0]
    assert model == "replay-model"
    assert config["method_stage"] == "rag_generate"
    prompt = messages[0]["content"]
    for record in expected_records:
        assert f"rank={record.rank}" in prompt
        assert f"document_id={record.document_id}" in prompt
        assert f"score={record.score:.6f}" in prompt
        assert record.text in prompt

    assert len(result["method_calls"]) == 1
    assert result["method_calls"][0].stage == "rag_generate"
    assert result["method_calls"][0].messages == messages
    assert result["method_calls"][0].retrieved_records == expected_records


def test_faithful_rag_does_not_fallback_to_proxy_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenDenseIndex:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("dense index unavailable")

    monkeypatch.setattr(retrieval_rag, "DenseIndex", BrokenDenseIndex)
    client = _FakeClient()

    with pytest.raises(RuntimeError, match="dense index unavailable"):
        RetrievalRagPolicy().run(
            _task(),
            MemoryState(entries=[_entry("doc-a", "content")]),
            client=client,
            model="replay-model",
            config={"top_k": 1},
        )

    assert client.calls == []
