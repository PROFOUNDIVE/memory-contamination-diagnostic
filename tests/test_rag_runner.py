from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.baselines import retrieval_rag
from memcontam.baselines.contracts import CorpusIdentity
from memcontam.baselines.retrieval_rag import RetrievalRagPolicy
from memcontam.clients.base import LLMResponse
from memcontam.cli import load_config, run_config
from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.retrieval import DenseIndex
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.dispatch import canonical_task_json
from memcontam.tasks.base import TaskInstance


def test_rag_contract_exposes_single_adapter_and_canonical_query_renderer() -> None:
    adapter = getattr(retrieval_rag, "RetrievalRagAdapter", None)
    assert adapter is not None
    assert callable(adapter().execute)
    assert not hasattr(adapter(), "run")
    assert not hasattr(adapter(), "build_prompt")
    assert callable(getattr(retrieval_rag, "render_retrieved_documents", None))
    assert not hasattr(RetrievalRagPolicy(), "build_prompt")
    assert not hasattr(retrieval_rag, "run_faithful_rag")


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


def _corpus_identity(task: TaskInstance) -> CorpusIdentity:
    return CorpusIdentity(
        manifest_id="fixture-corpus",
        corpus_version="v1",
        task_family=task.task_name,
        embedding_provider_identity="fake-deterministic-embedding@local",
    )


def _fixture(name: str) -> dict:
    return json.loads(
        (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")
    )


def test_retrieval_rag_uses_canonical_top_three_text_only_prompt(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(vector_dimension=8)
    entries = [
        _entry("doc-a", "Use multiplication before addition."),
        _entry("doc-b", "Try factor pairs that reach twenty four."),
        _entry("doc-c", "Sort words alphabetically."),
        _entry("doc-d", "Keep intermediate values exact."),
    ]
    memory = MemoryState(entries=entries)
    expected_records = DenseIndex(entries, provider=provider, cache_dir=tmp_path / "expected").retrieve(
        canonical_task_json(_task()), 3
    )
    prompt_fixture = _fixture("prompts/rag_generate.json")
    replay_fixture = _fixture("replay/baseline_fidelity_v1/retrieval_rag.json")
    client = _FakeClient(replay_fixture["response"])

    outcome = retrieval_rag.RetrievalRagAdapter().execute(
        _task(),
        memory,
        client=client,
        model="replay-model",
        config={"top_k": 1, "temperature": 0.0, "sample_id": "game24_1"},
        embedding_provider=provider,
        corpus_identity=_corpus_identity(_task()),
        cache_dir=tmp_path / "actual",
        verifier=lambda answer, task: False,
    )

    assert outcome.status == "succeeded"
    assert outcome.final_response == replay_fixture["response"]
    assert outcome.parsed_answer == "24"
    assert outcome.verifier_result is False
    entries_by_id = {entry.entry_id: entry for entry in entries}
    assert outcome.retrieved_memory == tuple(
        entries_by_id[record.document_id].model_dump() for record in expected_records
    )
    assert outcome.retrieved_scores == tuple(record.score for record in expected_records)
    assert outcome.memory_before == tuple(entry.model_dump() for entry in entries)
    assert outcome.memory_after == outcome.memory_before
    assert outcome.memory_write_event is None
    assert outcome.metadata == {
        "corpus_hash": expected_records[0].corpus_hash,
        "embedding_model_id": provider.metadata["model_id"],
        "embedding_revision": provider.metadata["revision"],
        "embedding_library_version": provider.metadata["embedding_library_version"],
        "top_k": replay_fixture["top_k"],
        "effective_k": 3,
        "similarity": "normalized_dot_product",
        "normalization": True,
        "retrieval_unit": "document",
        "query_serialization_version": "canonical_task_json_v1",
        "corpus_identity": {
            "manifest_id": "fixture-corpus",
            "corpus_version": "v1",
            "task_family": "game24",
            "embedding_provider_identity": "fake-deterministic-embedding@local",
        },
    }

    assert len(client.calls) == 1
    messages, model, config = client.calls[0]
    assert model == "replay-model"
    assert config["method_stage"] == "rag_generate"
    assert messages[0] == {
        "role": "system",
        "content": prompt_fixture["system_instruction"],
    }
    prompt = messages[1]["content"]
    assert prompt == (
        prompt_fixture["documents_header"]
        + "\n\n".join(record.text for record in expected_records)
        + prompt_fixture["task_header"]
        + canonical_task_json(_task())
    )
    for record in expected_records:
        assert record.text in prompt
    for forbidden in ("rank=", "document_id=", "score=", "source=", "metadata="):
        assert forbidden not in prompt
    assert len(outcome.method_calls) == 1
    assert outcome.method_calls[0].stage == "rag_generate"
    assert outcome.method_calls[0].messages == messages
    assert outcome.method_calls[0].retrieved_records == expected_records


def test_retrieval_rag_retrieval_failure_uses_the_closed_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenDenseIndex:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def retrieve(self, query: str, k: int) -> list[object]:
            raise RuntimeError("dense index unavailable")

    adapter_module = pytest.importorskip("memcontam.baselines.retrieval_rag_adapter")
    monkeypatch.setattr(adapter_module, "DenseIndex", BrokenDenseIndex)
    client = _FakeClient()
    failure_fixture = _fixture("replay/baseline_fidelity_v1/rag_failure_taxonomy.json")

    result = RetrievalRagPolicy().run(
        _task(),
        MemoryState(entries=[_entry("doc-a", "content")]),
        client=client,
        model="replay-model",
        embedding_provider=FakeEmbeddingProvider(),
        corpus_identity=_corpus_identity(_task()),
    )

    assert result["status"] == "failed"
    assert result["error_type"] == failure_fixture["error_type"]
    assert result["failure_disposition"] == failure_fixture["failure_disposition"]
    assert result["scientific_ineligibility_reason"] == failure_fixture[
        "scientific_ineligibility_reason"
    ]
    assert client.calls == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "stale dense index cache: manifest does not match corpus/provider",
            ("CorpusContractError", "rag_manifest_invalid", "manifest_invalid"),
        ),
        (
            "dimension mismatch for document doc-a: query has 8, document has 16",
            (
                "EmbeddingContractError",
                "rag_embedding_dimension_mismatch",
                "embedding_dimension_mismatch",
            ),
        ),
    ],
)
def test_retrieval_rag_index_contract_failures_keep_their_exact_taxonomy(
    monkeypatch: pytest.MonkeyPatch, message: str, expected: tuple[str, str, str]
) -> None:
    class BrokenDenseIndex:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ValueError(message)

    adapter_module = pytest.importorskip("memcontam.baselines.retrieval_rag_adapter")
    monkeypatch.setattr(adapter_module, "DenseIndex", BrokenDenseIndex)

    outcome = retrieval_rag.RetrievalRagAdapter().execute(
        _task(),
        MemoryState(entries=[_entry("doc-a", "content")]),
        client=_FakeClient(),
        model="replay-model",
        embedding_provider=FakeEmbeddingProvider(),
        corpus_identity=_corpus_identity(_task()),
    )

    assert (
        outcome.error_type,
        outcome.failure_disposition,
        outcome.scientific_ineligibility_reason,
    ) == expected


def test_v2_runner_passes_a_non_empty_task_bound_corpus_to_rag(tmp_path: Path) -> None:
    config = load_config(Path("configs/baseline_fidelity_v2_structural_replay.yaml"))
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["embedding"]["cache_path"] = str(tmp_path / "cache")
    run_dir = run_config(config, run_id="task-bound-rag-corpus")
    trials = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rag_trial = next(trial for trial in trials if trial["baseline"] == "retrieval_rag")

    assert rag_trial["status"] == "succeeded"
    assert len(rag_trial["retrieved_memory"]) == 3
    assert rag_trial["metadata"]["corpus_identity"]["task_family"] == "game24"
