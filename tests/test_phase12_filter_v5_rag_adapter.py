from __future__ import annotations

import pytest

from memcontam.baselines.retrieval_rag_phase12 import (
    BaselineStepResultV3,
    RagFrozenStateV3,
    RagFrozenTrialContextV3,
)
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.filter_challenge.adapters.rag_frozen import (
    RagFrozenChallengeError,
    RagFrozenPairRequest,
    RagFrozenProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.experiment.phase12.filter_challenge.provenance import AnswerCallProvenanceObserver
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY, BranchIndex
from memcontam.rag.phase12_corpus import BranchCorpus, Document
from memcontam.tasks.base import TaskInstance


class _ScriptedClient:
    def __init__(self) -> None:
        self.provider_calls_issued = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model, config
        return LLMResponse(content="final: 24", raw={}, token_usage={}, latency_ms=0)


class _Embedder:
    def encode_document(self, text: str) -> list[float]:
        return [1.0, 0.0] if text == "candidate" else [0.0, 1.0]

    def encode_query(self, text: str) -> list[float]:
        assert text
        return [1.0, 0.0]


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
    )


def _candidate(*, entry_id: str = "candidate", content: str = "candidate") -> ChallengeCandidate:
    return ChallengeCandidate.model_validate(
        {
            "candidate_entry_id": entry_id,
            "candidate_native_content": content,
            "candidate_native_kind": "rag_document",
            "baseline_family": "rag_frozen",
            "rag_mode": "frozen",
            "source_checkpoint_id": "source-checkpoint",
            "source_active_state_hash": "source-state-hash",
            "routability": {
                "routability": "challenge_routable_v1",
                "challenge_suite_key": "suite-1",
            },
        }
    )


def _source_state(
    active_document_ids: tuple[str, ...] = ("source-a", "source-b", "source-c"),
) -> RagFrozenStateV3:
    documents = tuple(
        Document(document_id, document_id) for document_id in ("source-a", "source-b", "source-c")
    )
    active_documents = tuple(
        document for document in documents if document.document_id in active_document_ids
    )
    corpus = BranchCorpus("clean", documents, active_document_ids, "source-corpus")
    vectors = {
        "source-a": (0.8, 0.0),
        "source-b": (0.7, 0.0),
        "source-c": (0.6, 0.0),
    }
    index = BranchIndex(
        "clean",
        active_documents,
        {"production_identity": BGE_M3_PRIMARY_IDENTITY},
        {document.document_id: vectors[document.document_id] for document in active_documents},
        "source-index",
        _Embedder(),
    )
    return RagFrozenStateV3("clean", corpus, index)


def _trial(
    trial_id: str,
    observer: AnswerCallProvenanceObserver,
    included: tuple[str, ...] | None = None,
) -> RagFrozenTrialContextV3:
    return RagFrozenTrialContextV3(
        task=_task(),
        client=_ScriptedClient(),
        model="replay",
        run_id="rag-provisional",
        trial_id=trial_id,
        condition_id="rag_frozen",
        branch="clean",
        rag_mode="frozen",
        included_document_ids=included,
        verifier=lambda answer, _task: answer == "24",
        provenance_observer=observer,
    )


def _request(
    *,
    candidate: ChallengeCandidate | None = None,
    source_state: RagFrozenStateV3 | None = None,
    challenge_included: tuple[str, ...] = ("source-a", "source-b"),
) -> RagFrozenPairRequest:
    return RagFrozenPairRequest(
        control_trial=_trial("rag-provisional:control", AnswerCallProvenanceObserver()),
        challenge_trial=_trial(
            "rag-provisional:challenge",
            AnswerCallProvenanceObserver(),
            included=challenge_included,
        ),
        source_state=_source_state() if source_state is None else source_state,
        candidate=_candidate() if candidate is None else candidate,
    )


def _answer_source_ids(result: BaselineStepResultV3) -> tuple[str, ...]:
    outcome = result.outcome
    assert outcome.answer_call_id is not None
    answer_call = next(call for call in outcome.method_calls if call.call_id == outcome.answer_call_id)
    return tuple(span.entry_id for span in answer_call.source_spans)


def test_rag_provisional_adapter_uses_the_source_for_control_and_discarded_deterministic_copy_for_challenge() -> None:
    # Given: a frozen source corpus/index and a candidate that ranks above its three source documents.
    request = _request()
    source_corpus = request.source_state.corpus
    source_index = request.source_state.index
    assert source_corpus is not None and source_index is not None
    source_corpus_hash = source_corpus.content_hash
    source_index_hash = source_index.artifact_hash

    # When: read-only control and provisional challenge executions answer through native frozen RAG.
    first = RagFrozenProvisionalAdapter().execute(request)
    second = RagFrozenProvisionalAdapter().execute(_request())

    # Then: the control keeps source identity, challenge retrieval is deterministic, and no source artifact changes.
    assert first.control.retrieval_event.retrieved_entry_ids == ["source-a", "source-b", "source-c"]
    assert first.challenge.retrieval_event.retrieved_entry_ids == ["candidate", "source-a", "source-b"]
    assert first.provisional_index_artifact_hash == second.provisional_index_artifact_hash
    assert first.source_index_artifact_hash_after == source_index_hash
    assert first.control.outcome.metadata["index_artifact_hash"] == source_index_hash
    assert first.challenge.outcome.metadata["index_artifact_hash"] == first.provisional_index_artifact_hash
    assert first.control.outcome.memory_write_event is None
    assert first.challenge.outcome.memory_write_event is None
    assert source_corpus.content_hash == source_corpus_hash
    assert source_index.artifact_hash == source_index_hash
    assert tuple(document.document_id for document in source_corpus.documents) == (
        "source-a",
        "source-b",
        "source-c",
    )
    assert tuple(source_index.vectors) == ("source-a", "source-b", "source-c")


def test_rag_provisional_adapter_binds_exposure_to_final_answer_context_not_retrieval() -> None:
    # Given: a retrieved candidate removed before final inclusion and explicit observers for both answer calls.
    request = _request()

    # When: native RAG renders and answers from the final included records.
    result = RagFrozenProvisionalAdapter().execute(request)

    # Then: retrieval alone never marks exposure, and every final source ID binds to its exact answer call.
    assert "candidate" in result.challenge.retrieval_event.retrieved_entry_ids
    assert result.challenge_final_source_ids == ("source-a", "source-b")
    assert result.candidate_exposure.candidate_final_context_inclusion is False
    assert result.candidate_exposure.candidate_final_context_source_ids == ("source-a", "source-b")
    assert tuple(result.challenge.context_event.final_entry_ids) == result.challenge_final_source_ids
    assert _answer_source_ids(result.control) == result.control_final_source_ids
    assert _answer_source_ids(result.challenge) == result.challenge_final_source_ids
    for trial, outcome in ((request.control_trial, result.control.outcome), (request.challenge_trial, result.challenge.outcome)):
        assert outcome.answer_call_id is not None
        assert trial.provenance_observer is not None
        assert trial.provenance_observer._finalized[outcome.answer_call_id].answer_call_provenance_status == "explicit_matched"
        assert isinstance(trial.client, _ScriptedClient)
        assert trial.client.provider_calls_issued == 0


@pytest.mark.parametrize(("entry_id", "content"), [("", "candidate"), ("candidate", "")])
def test_rag_provisional_adapter_rejects_invalid_native_candidate_document(
    entry_id: str, content: str
) -> None:
    # Given: a candidate that cannot be parsed as a native RAG document.
    request = _request(candidate=_candidate(entry_id=entry_id, content=content))

    # When / Then: provisional construction rejects the invalid candidate at the native boundary.
    with pytest.raises(ValueError, match="INVALID_RAG_DOCUMENT"):
        RagFrozenProvisionalAdapter().execute(request)


def test_rag_provisional_adapter_rejects_candidate_duplicate_of_source_document() -> None:
    # Given: a candidate with the same document identity as a frozen source entry.
    request = _request(candidate=_candidate(entry_id="source-a"))

    # When / Then: provisional insertion is rejected before source state is used.
    with pytest.raises(RagFrozenChallengeError, match="DUPLICATE_RAG_DOCUMENT"):
        RagFrozenProvisionalAdapter().execute(request)


def test_rag_provisional_adapter_binds_admitted_candidate_to_retrieval_context_and_answer() -> None:
    # Given: a retrieved candidate retained in the final challenge context.
    request = _request(challenge_included=("candidate", "source-a"))

    # When: native RAG answers from the admitted challenge context.
    result = RagFrozenProvisionalAdapter().execute(request)
    challenge_answer_sources = _answer_source_ids(result.challenge)

    # Then: retrieval, final context, answer spans, and exposure agree exactly.
    assert result.challenge.retrieval_event.retrieved_entry_ids == ["candidate", "source-a", "source-b"]
    assert tuple(result.challenge.context_event.final_entry_ids) == challenge_answer_sources
    assert result.challenge_final_source_ids == challenge_answer_sources
    assert result.candidate_exposure.candidate_final_context_inclusion is True
    assert result.candidate_exposure.candidate_final_context_source_ids == challenge_answer_sources


def test_rag_provisional_adapter_excludes_inactive_source_and_preserves_source_state() -> None:
    # Given: an inactive source document outside the frozen source index.
    request = _request(
        source_state=_source_state(("source-a", "source-b")),
        challenge_included=("candidate", "source-a"),
    )
    source_corpus = request.source_state.corpus
    source_index = request.source_state.index
    assert source_corpus is not None and source_index is not None
    source_corpus_hash = source_corpus.content_hash
    source_index_hash = source_index.artifact_hash

    # When: the candidate is assessed through an active-only provisional state.
    result = RagFrozenProvisionalAdapter().execute(request)

    # Then: inactive sources remain absent and the discarded copy leaves both source artifacts unchanged.
    assert result.challenge.retrieval_event.retrieved_entry_ids == ["candidate", "source-a", "source-b"]
    assert "source-c" not in result.challenge.context_event.final_entry_ids
    assert "source-c" not in _answer_source_ids(result.challenge)
    assert source_corpus.content_hash == source_corpus_hash
    assert source_index.artifact_hash == source_index_hash
    assert source_corpus.active_document_ids == ("source-a", "source-b")
    assert tuple(document.document_id for document in source_index.documents) == ("source-a", "source-b")
