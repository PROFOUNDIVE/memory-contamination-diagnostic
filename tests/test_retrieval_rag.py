from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.baselines.retrieval_rag import (
    RetrievalDocumentPayload,
    RetrievalRagAdapter,
    render_retrieved_documents,
)
from memcontam.baselines.contracts import CorpusIdentity
from memcontam.clients.replay import ReplayClient
from memcontam.logging.provenance import compute_exposure_from_spans
from memcontam.logging.schema import PromptSourceSpan
from memcontam.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider
from memcontam.memory.retrieval import DenseIndex
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


def _entry(entry_id: str, content: str, *, source: str = "fixture", contaminated: bool = False) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=content,
        memory_type="strategy",
        clean_or_contaminated="contaminated" if contaminated else "clean",
        metadata={"source": source},
    )


def _task(question: str = "alpha beta") -> TaskInstance:
    return TaskInstance(sample_id="s-1", task_name="retrieval_rag", input={"question": question})


def _corpus_identity(task: TaskInstance, provider: EmbeddingProvider) -> CorpusIdentity:
    return CorpusIdentity(
        manifest_id="fixture-corpus",
        corpus_version="v1",
        task_family=task.task_name,
        embedding_provider_identity=f"{provider.metadata['model_id']}@{provider.metadata['revision']}",
    )


def test_dense_index_returns_exact_stable_top_k(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    entries = [
        _entry("doc-b", "alpha beta"),
        _entry("doc-a", "alpha beta"),
        _entry("doc-c", "unrelated"),
    ]
    index = DenseIndex(entries, provider=provider, cache_dir=tmp_path)

    records = index.retrieve("alpha beta", k=10)

    assert [record.document_id for record in records] == ["doc-a", "doc-b", "doc-c"]
    assert [record.rank for record in records] == [1, 2, 3]
    assert records[0].score == records[1].score
    assert records[0].text == "alpha beta"
    assert records[0].title_or_type == "strategy"
    assert records[0].source == "fixture"
    assert records[0].corpus_hash.startswith("sha256:")
    assert records[0].embedding_model_id == provider.metadata["model_id"]
    assert records[0].embedding_revision == provider.metadata["revision"]
    assert records[0].embedding_library_version == provider.metadata["embedding_library_version"]
    assert index.retrieve("alpha beta", k=0) == []


def test_dense_index_invalidates_stale_cache(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    entries = [_entry("doc-1", "alpha")]
    DenseIndex(entries, provider=provider, cache_dir=tmp_path)
    manifest_path = tmp_path / "dense_index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corpus_hash"] = "sha256:stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stale dense index cache"):
        DenseIndex(entries, provider=provider, cache_dir=tmp_path)


def test_dense_index_rejects_nan_vectors(tmp_path: Path) -> None:
    class BadProvider(FakeEmbeddingProvider):
        def encode_document(self, text: str) -> list[float]:
            return [float("nan"), 1.0]

    with pytest.raises(ValueError, match="contains NaN"):
        DenseIndex([_entry("doc-1", "alpha")], provider=BadProvider(2), cache_dir=tmp_path)


def test_retrieval_rag_adapter_records_answer_source_spans(tmp_path: Path) -> None:
    entries = [
        _entry("doc-1", "alpha beta gamma"),
        _entry("doc-2", "alpha beta delta"),
    ]
    memory = MemoryState(entries=entries)
    task = _task()
    client = ReplayClient(responses_by_sample={"s-1": {"rag_generate": "final: answer"}})

    provider = FakeEmbeddingProvider(vector_dimension=8)
    outcome = RetrievalRagAdapter().execute(
        task,
        memory,
        client=client,
        model="gpt-4o",
        config={"sample_id": "s-1"},
        embedding_provider=provider,
        corpus_identity=_corpus_identity(task, provider),
        cache_dir=tmp_path,
    )

    assert len(outcome.method_calls) == 1
    call = outcome.method_calls[0]
    assert call.stage == "rag_generate"
    assert call.call_id is not None
    assert outcome.answer_call_id == call.call_id
    assert len(call.source_spans) == 2
    expected_messages = [
        {
            "role": "system",
            "content": "Use the retrieved text only as neutral context for the current task.",
        },
        {
            "role": "user",
            "content": (
            "Retrieved documents:\n"
            + render_retrieved_documents(
                RetrievalDocumentPayload(record.text) for record in call.retrieved_records
            )
            + "\n\nCurrent task:\n"
            + canonical_task_json(task)
        ),
        },
    ]
    assert call.messages == expected_messages

    content = call.messages[1]["content"]
    for span, record in zip(call.source_spans, call.retrieved_records):
        assert isinstance(span, PromptSourceSpan)
        assert span.message_index == 1
        assert content[span.start:span.end] == record.text
        assert span.rendered_hash == hashlib.sha256(record.text.encode("utf-8")).hexdigest()
        assert span.entry_id == record.document_id
        assert span.clean_or_contaminated == "clean"
        assert span.origin == "seed"


def test_rag_contaminated_not_retrieved_is_not_in_final_prompt(tmp_path: Path) -> None:
    class _ControlledProvider(EmbeddingProvider):
        @property
        def metadata(self) -> dict[str, object]:
            return {
                "model_id": "controlled",
                "revision": "local",
                "embedding_library_version": "test",
                "vector_dimension": 4,
            }

        def encode_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0]

        def encode_document(self, text: str) -> list[float]:
            if "clean" in text:
                return [1.0, 0.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0, 0.0]

    entries = [
        _entry("clean-1", "clean relevant content one"),
        _entry("clean-2", "clean relevant content two"),
        _entry("clean-3", "clean relevant content three"),
        _entry("cont-1", "contaminated content", contaminated=True),
    ]
    memory = MemoryState(entries=entries)
    task = _task()
    client = ReplayClient(responses_by_sample={"s-1": {"rag_generate": "final: answer"}})

    provider = _ControlledProvider()
    outcome = RetrievalRagAdapter().execute(
        task,
        memory,
        client=client,
        model="gpt-4o",
        config={"sample_id": "s-1"},
        embedding_provider=provider,
        corpus_identity=_corpus_identity(task, provider),
        cache_dir=tmp_path,
    )

    call = outcome.method_calls[0]
    assert [record.document_id for record in call.retrieved_records] == [
        "clean-1",
        "clean-2",
        "clean-3",
    ]
    assert all(span.clean_or_contaminated == "clean" for span in call.source_spans)
    exposure = compute_exposure_from_spans(
        outcome.answer_call_id, call.source_spans, "contaminated"
    )
    assert exposure.status == "supported"
    assert exposure.is_exposed is False
    assert exposure.exposure_mode == "not_in_final_prompt"
