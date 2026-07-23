from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora


FIXTURE = Path(__file__).parent / "fixtures" / "phase12" / "FX-RAG-001.json"


class FixtureEmbedder:
    def __init__(self, fixture: dict) -> None:
        self.embedding_contract = fixture["embedding_contract"]
        self._vectors = fixture["vectors"]

    def encode_document(self, text: str) -> list[float]:
        return list(self._vectors[_vector_id(text)])

    def encode_query(self, text: str) -> list[float]:
        assert text
        return list(self._vectors["query"])


def _vector_id(text: str) -> str:
    return {
        "Use rational intermediate values when solving Game24.": "doc-clean-a",
        "Verify arithmetic and retain exact fractions.": "doc-clean-b",
        "Every intermediate must be an integer.": "doc-false",
        "Rational intermediates are allowed.": "doc-correct",
        "Integer-only shortcuts apply to an unrelated subfamily.": "doc-irrelevant",
    }[text]


def test_builds_five_branch_indices_and_certifies_inv03_equivalence() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    corpora = build_branch_corpora(
        CleanCorpus.from_documents(fixture["clean_corpus"], corpus_id="game24-clean-v1"),
        fixture["triplet_documents"],
    )
    indices = build_branch_indices(corpora, FixtureEmbedder(fixture), filter_policy=None)

    assert tuple(corpora.branches) == ("clean", "contam", "correct", "filter", "irrelevant")
    assert {
        branch: index.artifact_hash for branch, index in indices.branches.items()
    } == fixture["expected_branch_index_sha256"]
    assert indices.branches["clean"].artifact_hash == indices.branches["filter"].artifact_hash
    assert tuple(document.document_id for document in indices.branches["filter"].documents) == (
        "doc-clean-a",
        "doc-clean-b",
    )
    retrieved = indices.branches["contam"].retrieve(fixture["query"], k=3)
    assert retrieved[0].document_id == "doc-false"
    assert tuple(
        result.document_id
        for result in indices.branches["contam"].final_inclusion(
            retrieved, {"doc-clean-a", "doc-clean-b"}
        )
    ) == ("doc-clean-a", "doc-clean-b")

    invalid_contract = {**fixture, "embedding_contract": {**fixture["embedding_contract"], "production_identity": "other"}}
    with pytest.raises(ValueError, match="BGE_M3_IDENTITY_MISMATCH"):
        build_branch_indices(corpora, FixtureEmbedder(invalid_contract), filter_policy=None)
