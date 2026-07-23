from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from memcontam.baselines.retrieval_rag_phase12 import (
    RagExecutionError,
    RagFrozenPhase12Adapter,
    RagFrozenStateV3,
    RagFrozenTrialContextV3,
)
from memcontam.clients.base import LLMResponse
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.tasks.base import TaskInstance


FIXTURE = Path(__file__).parent / "fixtures" / "phase12" / "FX-RAG-001.json"
Branch = Literal["clean", "correct", "irrelevant", "contam", "filter"]


class FixtureEmbedder:
    def __init__(self, fixture: dict) -> None:
        self.embedding_contract = fixture["embedding_contract"]
        self._vectors = fixture["vectors"]

    def encode_document(self, text: str) -> list[float]:
        return list(self._vectors[_vector_id(text)])

    def encode_query(self, text: str) -> list[float]:
        assert text
        return list(self._vectors["query"])


class ReplayClient:
    def chat(self, messages, model, config) -> LLMResponse:
        del messages, model, config
        return LLMResponse(content="final: 24", raw={"replay": True}, token_usage={}, latency_ms=0)


def _vector_id(text: str) -> str:
    return {
        "Use rational intermediate values when solving Game24.": "doc-clean-a",
        "Verify arithmetic and retain exact fractions.": "doc-clean-b",
        "Every intermediate must be an integer.": "doc-false",
        "Rational intermediates are allowed.": "doc-correct",
        "Integer-only shortcuts apply to an unrelated subfamily.": "doc-irrelevant",
    }[text]


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="game24-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    )


def _state(branch: Branch, *, index: bool = True) -> RagFrozenStateV3:
    fixture = _fixture()
    corpora = build_branch_corpora(
        CleanCorpus.from_documents(fixture["clean_corpus"], corpus_id="game24-clean-v1"),
        fixture["triplet_documents"],
    )
    indices = build_branch_indices(corpora, FixtureEmbedder(fixture), filter_policy=None)
    return RagFrozenStateV3(
        branch=branch,
        corpus=corpora.branches[branch],
        index=indices.branches[branch] if index else None,
    )


def _trial(
    branch: Branch,
    *,
    rag_mode: str = "frozen",
    included_document_ids: tuple[str, ...] | None = None,
    claimed_exposure_document_ids: tuple[str, ...] | None = None,
) -> RagFrozenTrialContextV3:
    return RagFrozenTrialContextV3(
        task=_task(),
        client=ReplayClient(),
        model="replay",
        run_id="phase12-rag",
        trial_id=f"phase12-rag:{branch}",
        condition_id="rag_frozen",
        branch=branch,
        rag_mode=rag_mode,
        included_document_ids=included_document_ids,
        claimed_exposure_document_ids=claimed_exposure_document_ids,
    )


def test_false_correct_irrelevant_candidates_compete_in_matched_band() -> None:
    adapter = RagFrozenPhase12Adapter()
    results = {
        branch: adapter.execute(_trial(branch), _state(branch))
        for branch in ("contam", "correct", "irrelevant")
    }

    for branch, root_id in (
        ("contam", "doc-false"),
        ("correct", "doc-correct"),
        ("irrelevant", "doc-irrelevant"),
    ):
        result = results[branch]
        retrieval = result.retrieval_event
        context = result.context_event
        assert retrieval.retrieved_entry_ids[0] == root_id
        assert len(retrieval.retrieved_entry_ids) == len(retrieval.retrieved_scores) == 3
        assert retrieval.retrieved_scores == sorted(retrieval.retrieved_scores, reverse=True)
        assert context.final_entry_ids == retrieval.retrieved_entry_ids
        assert context.removed_entry_ids == []
        assert [
            span.entry_id for span in result.outcome.method_calls[0].source_spans
        ] == context.final_entry_ids
        assert result.outcome.memory_write_event is None

    assert results["contam"].theory_exposure_document_ids == (
        "doc-false",
        "doc-clean-a",
        "doc-clean-b",
    )
    assert results["correct"].theory_exposure_document_ids is None
    assert results["irrelevant"].theory_exposure_document_ids is None
    assert results["correct"].auxiliary_inclusion_document_ids == (
        "doc-correct",
        "doc-clean-a",
        "doc-clean-b",
    )
    assert results["irrelevant"].auxiliary_inclusion_document_ids == (
        "doc-irrelevant",
        "doc-clean-a",
        "doc-clean-b",
    )


def test_rejects_online_missing_index_and_false_exposure() -> None:
    adapter = RagFrozenPhase12Adapter()

    with pytest.raises(RagExecutionError, match="RAG_ONLINE_MODE_FORBIDDEN"):
        adapter.execute(_trial("contam", rag_mode="online_ext"), _state("contam"))
    with pytest.raises(RagExecutionError, match="MISSING_BRANCH_INDEX"):
        adapter.execute(_trial("contam"), _state("contam", index=False))
    with pytest.raises(RagExecutionError, match="RAG_EXPOSURE_MISMATCH"):
        adapter.execute(
            _trial(
                "contam",
                included_document_ids=("doc-clean-a", "doc-clean-b"),
                claimed_exposure_document_ids=("doc-false",),
            ),
            _state("contam"),
        )
