from __future__ import annotations

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder, RecordedResponse
from memcontam.experiment.phase12.filter_challenge.provenance import (
    AnswerCallFinalizationError,
    AnswerCallProvenanceObserver,
    known_valid_batch_is_healthy,
    pair_is_evaluable,
)


class _ScriptedClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.provider_calls_issued = 0
        self.seen_configs: list[dict] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model
        self.seen_configs.append(config)
        return self._responses.pop(0)


def _response(content: str = "final: 24") -> LLMResponse:
    return LLMResponse(
        content=content,
        raw={"provider_secret": "must-not-serialize"},
        token_usage={"total_tokens": 1},
        latency_ms=1,
    )


def _complete(observer: AnswerCallProvenanceObserver, recorded: RecordedResponse) -> None:
    observer.record_answer(recorded)
    observer.record_context(recorded.call_id, recorded.response, ("entry-1",))
    observer.record_parser(recorded.call_id, recorded.call_id, recorded.response, "parsed")
    observer.record_verifier(recorded.call_id, recorded.response, True)


def test_recorder_returns_explicit_call_id_and_preserves_legacy_response() -> None:
    # Given: scripted responses and a recorder with logging-only configuration.
    first, second = _response("auxiliary"), _response("answer")
    client = _ScriptedClient([first, second])
    recorder = MethodCallRecorder(client, trial_context={"trial_id": "trial"})

    # When: auxiliary and answer calls use the explicit and legacy APIs.
    auxiliary = recorder.chat_with_call_id([], "replay", {"_logging_marker": "hidden"})
    answer = recorder.chat([], "replay", {"_logging_marker": "hidden"})

    # Then: IDs bind the precise responses without exposing logging values to the client.
    assert auxiliary.call_id == "trial:call:1"
    assert auxiliary.response is first
    assert answer is second
    assert [config for config in client.seen_configs] == [{}, {}]
    assert client.provider_calls_issued == 0


def test_explicit_events_finalize_once_with_identity_and_redaction() -> None:
    # Given: one exact answer response and all explicitly bound provenance events.
    observer = AnswerCallProvenanceObserver()
    recorded = RecordedResponse("answer-1", _response())
    _complete(observer, recorded)

    # When: its relation is finalized.
    relation = observer.finalize("answer-1")

    # Then: it is explicit, serializable, and cannot retain raw provider data.
    assert relation.answer_call_provenance_status == "explicit_matched"
    assert relation.answer_call_id == relation.parsed_response_source_call_id == "answer-1"
    assert "provider_secret" not in relation.model_dump_json()
    with pytest.raises(AnswerCallFinalizationError):
        observer.finalize("answer-1")


def test_status_priority_uses_only_explicit_ids_and_identity() -> None:
    # Given: one observer per complete or deliberately incomplete answer binding.
    missing = AnswerCallProvenanceObserver()
    missing.record_answer(RecordedResponse("missing", _response()))
    missing.record_context("missing", _response(), ())

    ambiguous = AnswerCallProvenanceObserver()
    ambiguous_recorded = RecordedResponse("ambiguous", _response())
    _complete(ambiguous, ambiguous_recorded)
    ambiguous.record_context("ambiguous", ambiguous_recorded.response, ())

    historical = AnswerCallProvenanceObserver()
    historical_recorded = RecordedResponse("historical", _response())
    _complete(historical, historical_recorded)
    historical.mark_historical_audit_input("historical")

    mismatched = AnswerCallProvenanceObserver()
    mismatched_recorded = RecordedResponse("mismatched", _response("same-text"))
    mismatched.record_answer(mismatched_recorded)
    mismatched.record_context("mismatched", mismatched_recorded.response, ())
    mismatched.record_parser("mismatched", "other-call", _response("same-text"), "parsed")
    mismatched.record_verifier("mismatched", mismatched_recorded.response, True)

    # When: each binding is finalized.
    statuses = (
        missing.finalize("missing").answer_call_provenance_status,
        ambiguous.finalize("ambiguous").answer_call_provenance_status,
        historical.finalize("historical").answer_call_provenance_status,
        mismatched.finalize("mismatched").answer_call_provenance_status,
    )

    # Then: missing, duplicate, audit-only, and identity differences follow the fixed law.
    assert statuses == ("missing", "ambiguous", "historically_reconstructed", "mismatched")


def test_interleaved_identical_text_and_retried_answers_never_use_lookup() -> None:
    # Given: interleaved answer events with identical contents but distinct raw objects.
    observer = AnswerCallProvenanceObserver()
    first = RecordedResponse("retry:1", _response("final: same"))
    second = RecordedResponse("retry:2", _response("final: same"))
    observer.record_answer(first)
    observer.record_answer(second)
    observer.record_context("retry:2", second.response, ())
    observer.record_context("retry:1", first.response, ())
    observer.record_parser("retry:1", "retry:1", first.response, "first")
    observer.record_parser("retry:2", "retry:2", second.response, "second")
    observer.record_verifier("retry:2", second.response, True)
    observer.record_verifier("retry:1", first.response, True)

    # When: both calls finalize after interleaving.
    relations = (observer.finalize("retry:1"), observer.finalize("retry:2"))

    # Then: both retain their explicit call/response pairing despite duplicate text and retry order.
    assert all(relation.answer_call_provenance_status == "explicit_matched" for relation in relations)


def test_shared_response_ids_are_ambiguous_and_finalization_releases_stale_state() -> None:
    # Given: a response accidentally attached to two active call IDs.
    observer = AnswerCallProvenanceObserver()
    shared = _response()
    first = RecordedResponse("first", shared)
    _complete(observer, first)
    observer.record_context("second", shared, ())

    # When: both active relations finalize and the response is later reused after cleanup.
    first_relation = observer.finalize("first")
    second_relation = observer.finalize("second")
    third = RecordedResponse("third", shared)
    _complete(observer, third)
    third_relation = observer.finalize("third")

    # Then: active cross-ID reuse is ambiguous, while finalized relations leave no stale binding.
    assert first_relation.answer_call_provenance_status == "ambiguous"
    assert second_relation.answer_call_provenance_status == "missing"
    assert third_relation.answer_call_provenance_status == "explicit_matched"


def test_pairs_require_two_distinct_explicit_relations_and_known_valid_health() -> None:
    # Given: a complete control/challenge pair and an unresolved relation.
    observer = AnswerCallProvenanceObserver()
    control = RecordedResponse("control", _response())
    challenge = RecordedResponse("challenge", _response())
    _complete(observer, control)
    _complete(observer, challenge)

    # When: the pair is finalized through its two explicit answer IDs.
    relations = observer.finalize_pair("control", "challenge")

    # Then: both sides are required for evaluability and the known-valid health gate is all-or-nothing.
    assert len(relations) == 2
    assert pair_is_evaluable(relations)
    assert known_valid_batch_is_healthy(relations)

    unresolved = AnswerCallProvenanceObserver()
    unresolved_control = RecordedResponse("unresolved-control", _response())
    unresolved_challenge = RecordedResponse("unresolved-challenge", _response())
    _complete(unresolved, unresolved_control)
    unresolved.record_answer(unresolved_challenge)
    unresolved_relations = unresolved.finalize_pair("unresolved-control", "unresolved-challenge")
    assert not pair_is_evaluable(unresolved_relations)
    assert not known_valid_batch_is_healthy(unresolved_relations)
    with pytest.raises(AnswerCallFinalizationError):
        observer.finalize_pair("control", "control")
