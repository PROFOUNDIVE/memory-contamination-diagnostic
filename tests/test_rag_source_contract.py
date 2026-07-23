from __future__ import annotations

from pathlib import Path

from memcontam.baselines.contracts import CorpusIdentity
from memcontam.baselines.retrieval_rag import RetrievalRagAdapter
from memcontam.clients.replay import ReplayClient
from memcontam.memory.corpus import (
    build_arm_corpus,
    build_trusted_corpus_identity,
    load_corpus,
    load_corpus_manifest,
)
from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


CORPUS_PATH = Path("data/memory/baseline_fidelity_v2_contract_corpus.jsonl")
MANIFEST_PATH = Path("data/memory/baseline_fidelity_v2_contract_corpus.manifest.json")
NEUTRAL_SYSTEM_INSTRUCTION = "Use the retrieved text only as neutral context for the current task."


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="game24_1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    )


def _identity(task: TaskInstance, provider: FakeEmbeddingProvider) -> CorpusIdentity:
    return build_trusted_corpus_identity(
        load_corpus(CORPUS_PATH),
        manifest=load_corpus_manifest(MANIFEST_PATH),
        task_family=task.task_name,
        embedding_provider_identity=f"{provider.metadata['model_id']}@{provider.metadata['revision']}",
    )


def test_rag_source_contract_requires_identity_before_retrieval(tmp_path: Path) -> None:
    task = _task()
    provider = FakeEmbeddingProvider(vector_dimension=8)
    entries, _ = build_arm_corpus(load_corpus(CORPUS_PATH), task.task_name, "clean")

    outcome = RetrievalRagAdapter().execute(
        task,
        MemoryState(entries=entries),
        client=ReplayClient(responses_by_sample={"game24_1": {"rag_generate": "final: 24"}}),
        model="replay",
        config={"_require_corpus_identity": True},
        embedding_provider=provider,
        cache_dir=tmp_path,
    )

    assert outcome.status == "failed"
    assert (
        outcome.error_type,
        outcome.failure_disposition,
        outcome.scientific_ineligibility_reason,
    ) == ("CorpusContractError", "rag_manifest_invalid", "manifest_invalid")
    assert outcome.method_calls == ()


def test_rag_source_contract_rejects_an_empty_identified_corpus(tmp_path: Path) -> None:
    task = _task()
    provider = FakeEmbeddingProvider(vector_dimension=8)

    outcome = RetrievalRagAdapter().execute(
        task,
        MemoryState(),
        client=ReplayClient(responses_by_sample={"game24_1": {"rag_generate": "final: 24"}}),
        model="replay",
        config={"_require_corpus_identity": True},
        embedding_provider=provider,
        corpus_identity=_identity(task, provider),
        cache_dir=tmp_path,
    )

    assert outcome.status == "failed"
    assert (
        outcome.error_type,
        outcome.failure_disposition,
        outcome.scientific_ineligibility_reason,
    ) == ("CorpusContractError", "rag_manifest_invalid", "manifest_invalid")
    assert outcome.method_calls == ()


def test_rag_source_contract_uses_three_text_only_documents_and_no_memory_write(
    tmp_path: Path,
) -> None:
    task = _task()
    provider = FakeEmbeddingProvider(vector_dimension=8)
    entries, _ = build_arm_corpus(load_corpus(CORPUS_PATH), task.task_name, "clean")
    memory = MemoryState(entries=entries)
    memory_before = tuple(entry.model_dump() for entry in memory.entries)

    outcome = RetrievalRagAdapter().execute(
        task,
        memory,
        client=ReplayClient(responses_by_sample={"game24_1": {"rag_generate": "final: 24"}}),
        model="replay",
        config={"_require_corpus_identity": True},
        embedding_provider=provider,
        corpus_identity=_identity(task, provider),
        cache_dir=tmp_path,
    )

    assert outcome.status == "succeeded"
    call = outcome.method_calls[0]
    assert len(call.retrieved_records) == 3
    assert call.messages[0] == {"role": "system", "content": NEUTRAL_SYSTEM_INSTRUCTION}
    prompt = call.messages[1]["content"]
    for record in call.retrieved_records:
        assert record.text in prompt
    for record in call.retrieved_records:
        for hidden in (
            record.document_id,
            f"rank={record.rank}",
            f"score={record.score}",
            record.title_or_type,
            record.clean_or_contaminated,
            record.source,
        ):
            assert hidden not in prompt
    assert outcome.memory_before == memory_before
    assert outcome.memory_after == memory_before
    assert outcome.memory_write_event is None
    assert outcome.metadata["effective_k"] == 3
    assert outcome.metadata["similarity"] == "normalized_dot_product"
    assert outcome.metadata["normalization"] is True
    assert outcome.metadata["retrieval_unit"] == "document"
    assert outcome.metadata["query_serialization_version"] == "canonical_task_json_v1"
    assert outcome.metadata["corpus_identity"] == {
        "manifest_id": "baseline-fidelity-v2-contract-corpus",
        "corpus_version": "v1",
        "task_family": "game24",
        "embedding_provider_identity": "fake-deterministic-embedding@local",
    }


def test_rag_source_contract_corpus_has_exactly_three_records_per_task() -> None:
    records = load_corpus(CORPUS_PATH)
    for task_name in ("game24", "math_equation_balancer", "word_sorting"):
        entries, _ = build_arm_corpus(records, task_name, "clean")
        assert len(entries) == 3
