from __future__ import annotations

from dataclasses import dataclass, fields
import importlib
import importlib.util
from typing import cast

from memcontam.memory import corpus, retrieval


def test_corpus_identity_has_one_shared_definition() -> None:
    assert importlib.util.find_spec("memcontam.baselines.contracts"), (
        "CorpusIdentity must be defined once in memcontam.baselines.contracts"
    )
    contracts = importlib.import_module("memcontam.baselines.contracts")

    identity = getattr(contracts, "CorpusIdentity", None)
    assert identity is not None
    assert identity.__module__ == "memcontam.baselines.contracts"
    assert [field.name for field in fields(identity)] == [
        "manifest_id",
        "corpus_version",
        "task_family",
        "embedding_provider_identity",
    ]
    assert corpus.CorpusIdentity is identity
    assert retrieval.CorpusIdentity is identity


def test_retrieval_rag_consumes_the_shared_corpus_identity() -> None:
    contracts = importlib.import_module("memcontam.baselines.contracts")
    rag = importlib.import_module("memcontam.baselines.retrieval_rag")

    assert rag.CorpusIdentity is contracts.CorpusIdentity


def test_retrieval_rag_rejects_a_lookalike_corpus_identity() -> None:
    from memcontam.baselines.contracts import CorpusIdentity
    from memcontam.baselines.retrieval_rag import RetrievalRagAdapter
    from memcontam.clients.base import LLMResponse
    from memcontam.memory.embeddings import FakeEmbeddingProvider
    from memcontam.memory.stores import MemoryState
    from memcontam.tasks.base import TaskInstance

    @dataclass(frozen=True)
    class LookalikeCorpusIdentity:
        manifest_id: str
        corpus_version: str
        task_family: str
        embedding_provider_identity: str

    class Client:
        def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
            raise AssertionError("invalid corpus identity must fail before generation")

    outcome = RetrievalRagAdapter().execute(
        TaskInstance(sample_id="sample-1", task_name="game24", input={}),
        MemoryState(),
        client=Client(),
        model="replay",
        embedding_provider=FakeEmbeddingProvider(),
        corpus_identity=cast(
            CorpusIdentity,
            LookalikeCorpusIdentity(
                manifest_id="fixture-corpus",
                corpus_version="v1",
                task_family="game24",
                embedding_provider_identity="fake-deterministic-embedding@local",
            ),
        ),
    )

    assert outcome.status == "failed"
    assert outcome.error_type == "CorpusContractError"
    assert outcome.failure_disposition == "rag_manifest_invalid"
    assert outcome.scientific_ineligibility_reason == "manifest_invalid"
