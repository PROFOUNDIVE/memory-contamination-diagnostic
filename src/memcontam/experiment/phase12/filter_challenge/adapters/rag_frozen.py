from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, assert_never

from memcontam.baselines.retrieval_rag_phase12 import (
    BaselineStepResultV3,
    RagFrozenPhase12Adapter,
    RagFrozenStateV3,
    RagFrozenTrialContextV3,
)
from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeCandidate,
)
from memcontam.experiment.phase12.filter_challenge.provenance import (
    AnswerCallProvenanceObserver,
)
from memcontam.rag.branch_index import BranchIndex
from memcontam.rag.phase12_corpus import BranchCorpus, Document
from memcontam.contamination.phase12.models import canonical_json_hash


class RagFrozenChallengeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RagFrozenPairRequest:
    control_trial: RagFrozenTrialContextV3
    challenge_trial: RagFrozenTrialContextV3
    source_state: RagFrozenStateV3
    candidate: ChallengeCandidate
    cached_control: RagFrozenCachedControl | None = None
    execution_order: Literal["control_first", "challenge_first"] = "control_first"


@dataclass(frozen=True, slots=True)
class RagFrozenCachedControl:
    result: BaselineStepResultV3
    answer_relation: AnswerCallRelation


@dataclass(frozen=True, slots=True)
class RagFrozenPairResult:
    control: BaselineStepResultV3
    challenge: BaselineStepResultV3
    control_final_source_ids: tuple[str, ...]
    challenge_final_source_ids: tuple[str, ...]
    candidate_exposure: CandidateExposureRecord
    provisional_index_artifact_hash: str
    source_index_artifact_hash_after: str
    control_answer_relation: AnswerCallRelation
    challenge_answer_relation: AnswerCallRelation


class RagFrozenProvisionalAdapter:
    def execute(self, request: RagFrozenPairRequest) -> RagFrozenPairResult:
        _validate_request(request)
        source_index = request.source_state.index
        assert source_index is not None
        challenge_state = _provisional_state(request.source_state, request.candidate)
        provisional_index = challenge_state.index
        assert provisional_index is not None
        control_trial, control_observer = _with_observer(request.control_trial)
        challenge_trial, challenge_observer = _with_observer(request.challenge_trial)
        if request.cached_control is not None:
            control = request.cached_control.result
            control_relation = request.cached_control.answer_relation
            challenge = RagFrozenPhase12Adapter().execute(challenge_trial, challenge_state)
        else:
            match request.execution_order:
                case "control_first":
                    control = RagFrozenPhase12Adapter().execute(control_trial, request.source_state)
                    challenge = RagFrozenPhase12Adapter().execute(challenge_trial, challenge_state)
                case "challenge_first":
                    challenge = RagFrozenPhase12Adapter().execute(challenge_trial, challenge_state)
                    control = RagFrozenPhase12Adapter().execute(control_trial, request.source_state)
                case unreachable:
                    assert_never(unreachable)
            control_call_id = control.outcome.answer_call_id
            if control_call_id is None:
                raise RagFrozenChallengeError("MISSING_ANSWER_CALL")
            control_relation = control_observer.finalized_relation(control_call_id)
        challenge_call_id = challenge.outcome.answer_call_id
        if challenge_call_id is None:
            raise RagFrozenChallengeError("MISSING_ANSWER_CALL")
        challenge_relation = challenge_observer.finalized_relation(challenge_call_id)
        control_final_source_ids = _answer_source_ids(control)
        challenge_final_source_ids = _answer_source_ids(challenge)
        return RagFrozenPairResult(
            control=control,
            challenge=challenge,
            control_final_source_ids=control_final_source_ids,
            challenge_final_source_ids=challenge_final_source_ids,
            candidate_exposure=CandidateExposureRecord(
                candidate_entry_id=request.candidate.candidate_entry_id,
                candidate_final_context_inclusion=(
                    request.candidate.candidate_entry_id in challenge_final_source_ids
                ),
                candidate_final_context_source_ids=challenge_final_source_ids,
            ),
            provisional_index_artifact_hash=provisional_index.artifact_hash,
            source_index_artifact_hash_after=source_index.artifact_hash,
            control_answer_relation=control_relation,
            challenge_answer_relation=challenge_relation,
        )


def _with_observer(
    trial: RagFrozenTrialContextV3,
) -> tuple[RagFrozenTrialContextV3, AnswerCallProvenanceObserver]:
    observer = trial.provenance_observer or AnswerCallProvenanceObserver()
    return replace(trial, provenance_observer=observer), observer


def _validate_request(request: RagFrozenPairRequest) -> None:
    source_corpus = request.source_state.corpus
    source_index = request.source_state.index
    if source_corpus is None or source_index is None:
        raise RagFrozenChallengeError("MISSING_RAG_SOURCE")
    if request.candidate.baseline_family != "rag_frozen" or request.candidate.rag_mode != "frozen":
        raise RagFrozenChallengeError("INVALID_RAG_CANDIDATE")
    if request.candidate.candidate_native_kind != "rag_document":
        raise RagFrozenChallengeError("INVALID_RAG_CANDIDATE")
    if request.control_trial.branch != source_corpus.branch:
        raise RagFrozenChallengeError("CONTROL_BRANCH_MISMATCH")
    if request.challenge_trial.branch != source_corpus.branch:
        raise RagFrozenChallengeError("CHALLENGE_BRANCH_MISMATCH")
    if request.control_trial.rag_mode != "frozen" or request.challenge_trial.rag_mode != "frozen":
        raise RagFrozenChallengeError("RAG_ONLINE_MODE_FORBIDDEN")
    if request.candidate.candidate_entry_id in {
        *(document.document_id for document in source_corpus.documents),
        *(document.document_id for document in source_index.documents),
    }:
        raise RagFrozenChallengeError("DUPLICATE_RAG_DOCUMENT")


def _provisional_state(
    source_state: RagFrozenStateV3, candidate: ChallengeCandidate
) -> RagFrozenStateV3:
    source_corpus = source_state.corpus
    source_index = source_state.index
    assert source_corpus is not None and source_index is not None
    candidate_document = Document.from_mapping(
        {"id": candidate.candidate_entry_id, "text": candidate.candidate_native_content}
    )
    identity = canonical_json_hash(candidate_document.payload())
    corpus = BranchCorpus(
        branch=source_corpus.branch,
        documents=(*source_corpus.documents, candidate_document),
        active_document_ids=(*source_corpus.active_document_ids, candidate_document.document_id),
        serialization_id=f"{source_corpus.serialization_id}|challenge:{identity}",
        corpus_version=source_corpus.corpus_version,
    )
    vectors = dict(source_index.vectors)
    vectors[candidate_document.document_id] = tuple(
        float(value) for value in source_index._embedder.encode_document(candidate_document.text)
    )
    index = BranchIndex(
        branch=source_index.branch,
        documents=(*source_index.documents, candidate_document),
        embedding_contract=source_index.embedding_contract,
        vectors=vectors,
        serialization_id=f"{source_index.serialization_id}|challenge:{identity}",
        _embedder=source_index._embedder,
        index_version=source_index.index_version,
    )
    return RagFrozenStateV3(source_state.branch, corpus, index)


def _answer_source_ids(result: BaselineStepResultV3) -> tuple[str, ...]:
    answer_call_id = result.outcome.answer_call_id
    if answer_call_id is None:
        raise RagFrozenChallengeError("MISSING_ANSWER_CALL")
    answer_calls = [call for call in result.outcome.method_calls if call.call_id == answer_call_id]
    if len(answer_calls) != 1:
        raise RagFrozenChallengeError("ANSWER_CALL_BINDING_MISMATCH")
    final_source_ids = tuple(result.context_event.final_entry_ids)
    if tuple(span.entry_id for span in answer_calls[0].source_spans) != final_source_ids:
        raise RagFrozenChallengeError("ANSWER_CONTEXT_BINDING_MISMATCH")
    return final_source_ids
