from __future__ import annotations

from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import RecordedResponse
from memcontam.experiment.phase12.filter_challenge.contracts import AnswerCallRelation
from memcontam.experiment.phase12.filter_challenge.mft_safety_types import (
    GateEvidence,
    MftIdentity,
    assertion,
    relation,
)
from memcontam.experiment.phase12.filter_challenge.provenance import (
    AnswerCallProvenanceObserver,
    known_valid_batch_is_healthy,
    pair_is_evaluable,
)


def _response() -> LLMResponse:
    return LLMResponse(content="fixture", raw={}, token_usage={"total_tokens": 0}, latency_ms=0)


def _complete(observer: AnswerCallProvenanceObserver, call_id: str) -> None:
    recorded = RecordedResponse(call_id, _response())
    observer.record_answer(recorded)
    observer.record_context(call_id, recorded.response, ())
    observer.record_parser(call_id, call_id, recorded.response, "parsed")
    observer.record_verifier(call_id, recorded.response, True)


def _relations() -> tuple[AnswerCallRelation, ...]:
    explicit = AnswerCallProvenanceObserver()
    _complete(explicit, "explicit")
    missing = AnswerCallProvenanceObserver()
    missing.record_answer(RecordedResponse("missing", _response()))
    ambiguous = AnswerCallProvenanceObserver()
    shared = RecordedResponse("ambiguous", _response())
    ambiguous.record_answer(shared)
    ambiguous.record_context("ambiguous", shared.response, ())
    ambiguous.record_context("ambiguous", shared.response, ())
    ambiguous.record_parser("ambiguous", "ambiguous", shared.response, "parsed")
    ambiguous.record_verifier("ambiguous", shared.response, True)
    historical = AnswerCallProvenanceObserver()
    _complete(historical, "historical")
    historical.mark_historical_audit_input("historical")
    mismatched = AnswerCallProvenanceObserver()
    mismatch = RecordedResponse("mismatched", _response())
    mismatched.record_answer(mismatch)
    mismatched.record_context("mismatched", mismatch.response, ())
    mismatched.record_parser("mismatched", "other-call", _response(), "parsed")
    mismatched.record_verifier("mismatched", mismatch.response, True)
    return (
        explicit.finalize("explicit"), missing.finalize("missing"),
        ambiguous.finalize("ambiguous"), historical.finalize("historical"),
        mismatched.finalize("mismatched"),
    )


def gate_provenance(mutated: bool) -> GateEvidence:
    relations = _relations()
    partner = relation("explicit_matched", "partner")
    admissibility = tuple(str(pair_is_evaluable((item, partner))).lower() for item in relations)
    health = str(known_valid_batch_is_healthy(relations)).lower()
    if mutated:
        health = "true"
    return GateEvidence(
        tuple(MftIdentity(field="answer_call_id", value=item.answer_call_id) for item in relations),
        (assertion("relation_statuses",
                   ("explicit_matched", "missing", "ambiguous", "historically_reconstructed", "mismatched"),
                   tuple(item.answer_call_provenance_status for item in relations)),
         assertion("primary_admissibility", ("true", "false", "false", "false", "false"), admissibility),
         assertion("known_valid_batch_health", ("false",), (health,))),
    )
