from __future__ import annotations

from dataclasses import fields
import importlib
import importlib.util

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
