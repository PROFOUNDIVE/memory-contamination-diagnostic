from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from memcontam.rag.leakage import RagContractError, audit_leakage, validate_rag_frozen_inputs
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora


FIXTURE = Path(__file__).parent / "fixtures" / "phase12" / "FX-RAG-001.json"


def test_rejects_stale_leaked_extreme_mixed_or_inv03_leaking_inputs() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    corpus = build_branch_corpora(
        CleanCorpus.from_documents(fixture["clean_corpus"], corpus_id="game24-clean-v1"),
        fixture["triplet_documents"],
    )
    stale = audit_leakage(corpus, {"corpus_hash": "stale", "splits": {}})
    leaked = audit_leakage(
        corpus,
        {
            "corpus_hash": corpus.clean.content_hash,
            "splits": {
                "construction": [{"id": "a", "text": "same rule"}],
                "evaluation": [{"id": "b", "text": "  SAME   RULE "}],
            },
        },
    )

    assert stale.codes == ("STALE_CORPUS_MANIFEST",)
    assert leaked.codes == ("CORPUS_LEAKAGE",)
    for kwargs, code in (
        ({"affinity_band": "extreme"}, "AFFINITY_BAND_FORBIDDEN"),
        ({"rag_mode": "online"}, "RAG_MODE_MISMATCH"),
        (
            {"input_surfaces": {"embedding": {"inv03_metadata": "leaked"}}},
            "INV03_METADATA_REACHED_EMBEDDING",
        ),
        (
            {"input_surfaces": {"renderer": {"inv03_metadata": "leaked"}}},
            "INV03_METADATA_REACHED_RENDERER",
        ),
        (
            {"input_surfaces": {"filter": {"inv03_metadata": "leaked"}}},
            "INV03_METADATA_REACHED_FILTER",
        ),
        (
            {"input_surfaces": {"verifier": {"inv03_metadata": "leaked"}}},
            "INV03_METADATA_REACHED_VERIFIER",
        ),
        ({"input_surfaces": {"ranking": {"inv03_metadata": "leaked"}}}, "INV03_RANKING_CHANGED"),
    ):
        with pytest.raises(RagContractError, match=code):
            validate_rag_frozen_inputs(
                corpus,
                {"corpus_hash": corpus.clean.content_hash, "splits": {}},
                **cast(Any, kwargs),
            )
