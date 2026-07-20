from __future__ import annotations

import importlib
import importlib.util


def test_corpus_identity_has_one_shared_definition() -> None:
    assert importlib.util.find_spec("memcontam.baselines.contracts"), (
        "CorpusIdentity must be defined once in memcontam.baselines.contracts"
    )
    contracts = importlib.import_module("memcontam.baselines.contracts")

    identity = getattr(contracts, "CorpusIdentity", None)
    assert identity is not None
    assert identity.__module__ == "memcontam.baselines.contracts"
