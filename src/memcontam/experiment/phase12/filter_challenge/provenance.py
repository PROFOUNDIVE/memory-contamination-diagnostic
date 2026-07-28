from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

from pydantic import TypeAdapter

from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import FailedRecordedCall, RecordedResponse
from memcontam.experiment.phase12.filter_challenge.contracts import AnswerCallRelation


_RELATION_ADAPTER: TypeAdapter[AnswerCallRelation] = TypeAdapter(AnswerCallRelation)
RelationStatus = Literal[
    "explicit_matched", "missing", "ambiguous", "historically_reconstructed", "mismatched"
]


@dataclass(frozen=True, slots=True)
class AnswerContextEvent:
    answer_call_id: str
    response: LLMResponse
    final_source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedResponseEvent:
    answer_call_id: str
    parsed_response_source_call_id: str
    response: LLMResponse
    parser_result_id: str
    parsed_output: str | None


@dataclass(frozen=True, slots=True)
class VerifiedResponseEvent:
    answer_call_id: str
    response: LLMResponse
    verifier_result_id: str
    is_correct: bool | None


class AnswerCallFinalizationError(ValueError):
    pass


class AnswerCallProvenanceObserver:
    """Retains exact answer responses only until their relation is finalized."""

    def __init__(self) -> None:
        self._answers: dict[str, list[RecordedResponse]] = {}
        self._contexts: dict[str, list[AnswerContextEvent]] = {}
        self._parsers: dict[str, list[ParsedResponseEvent]] = {}
        self._verifiers: dict[str, list[VerifiedResponseEvent]] = {}
        self._response_bindings: dict[int, dict[str, int]] = {}
        self._historical_audit_inputs: set[str] = set()
        self._finalized: dict[str, AnswerCallRelation] = {}

    def record_answer(self, recorded: RecordedResponse) -> None:
        self._answers.setdefault(recorded.call_id, []).append(recorded)
        self._bind_response(recorded.call_id, recorded.response)

    def record_context(
        self, answer_call_id: str, response: LLMResponse, final_source_ids: tuple[str, ...]
    ) -> None:
        self._contexts.setdefault(answer_call_id, []).append(
            AnswerContextEvent(answer_call_id, response, final_source_ids)
        )
        self._bind_response(answer_call_id, response)

    def record_parser(
        self,
        answer_call_id: str,
        parsed_response_source_call_id: str,
        response: LLMResponse,
        parsed_output: str | None,
    ) -> None:
        parser_events = self._parsers.setdefault(answer_call_id, [])
        parser_events.append(
            ParsedResponseEvent(
                answer_call_id,
                parsed_response_source_call_id,
                response,
                f"{parsed_response_source_call_id}:parser:{len(parser_events) + 1}",
                parsed_output,
            )
        )
        self._bind_response(parsed_response_source_call_id, response)

    def record_verifier(
        self, answer_call_id: str, response: LLMResponse, is_correct: bool | None
    ) -> None:
        verifier_events = self._verifiers.setdefault(answer_call_id, [])
        verifier_events.append(
            VerifiedResponseEvent(
                answer_call_id,
                response,
                f"{answer_call_id}:verifier:{len(verifier_events) + 1}",
                is_correct,
            )
        )
        self._bind_response(answer_call_id, response)

    def mark_historical_audit_input(self, answer_call_id: str) -> None:
        self._historical_audit_inputs.add(answer_call_id)

    def finalize(self, answer_call_id: str) -> AnswerCallRelation:
        if answer_call_id in self._finalized:
            raise AnswerCallFinalizationError(answer_call_id)
        answers = self._answers.get(answer_call_id, [])
        contexts = self._contexts.get(answer_call_id, [])
        parsers = self._parsers.get(answer_call_id, [])
        verifiers = self._verifiers.get(answer_call_id, [])
        if not all((answers, contexts, parsers, verifiers)):
            relation = _relation("missing", answer_call_id)
        elif any(len(events) != 1 for events in (answers, contexts, parsers, verifiers)):
            relation = _relation("ambiguous", answer_call_id)
        else:
            answer = answers[0]
            context = contexts[0]
            parser = parsers[0]
            verifier = verifiers[0]
            if len(self._response_bindings[id(answer.response)]) != 1:
                relation = _relation("ambiguous", answer_call_id)
            elif answer_call_id in self._historical_audit_inputs:
                relation = _relation("historically_reconstructed", answer_call_id)
            elif (
                context.answer_call_id != answer_call_id
                or parser.parsed_response_source_call_id != answer_call_id
                or verifier.answer_call_id != answer_call_id
                or context.response is not answer.response
                or parser.response is not answer.response
                or verifier.response is not answer.response
            ):
                relation = _relation("mismatched", answer_call_id)
            else:
                relation = _relation("explicit_matched", answer_call_id, parser, verifier)
        self._release(answer_call_id)
        self._finalized[answer_call_id] = relation
        return relation

    def finalize_pair(
        self, control_answer_call_id: str, challenge_answer_call_id: str
    ) -> tuple[AnswerCallRelation, AnswerCallRelation]:
        if control_answer_call_id == challenge_answer_call_id:
            raise AnswerCallFinalizationError(control_answer_call_id)
        return self.finalize(control_answer_call_id), self.finalize(challenge_answer_call_id)

    def _bind_response(self, call_id: str, response: LLMResponse) -> None:
        bindings = self._response_bindings.setdefault(id(response), {})
        bindings[call_id] = bindings.get(call_id, 0) + 1

    def _release(self, answer_call_id: str) -> None:
        bindings = [
            *((event.call_id, event.response) for event in self._answers.pop(answer_call_id, [])),
            *((event.answer_call_id, event.response) for event in self._contexts.pop(answer_call_id, [])),
            *(
                (event.parsed_response_source_call_id, event.response)
                for event in self._parsers.pop(answer_call_id, [])
            ),
            *((event.answer_call_id, event.response) for event in self._verifiers.pop(answer_call_id, [])),
        ]
        for bound_call_id, response in bindings:
            response_bindings = self._response_bindings[id(response)]
            response_bindings[bound_call_id] -= 1
            if response_bindings[bound_call_id] == 0:
                del response_bindings[bound_call_id]
            if not response_bindings:
                del self._response_bindings[id(response)]
        self._historical_audit_inputs.discard(answer_call_id)


def pair_is_evaluable(relations: tuple[AnswerCallRelation, AnswerCallRelation]) -> bool:
    control_relation, challenge_relation = relations
    return (
        control_relation.answer_call_id != challenge_relation.answer_call_id
        and control_relation.answer_call_provenance_status == "explicit_matched"
        and challenge_relation.answer_call_provenance_status == "explicit_matched"
    )


def known_valid_batch_is_healthy(relations: Sequence[AnswerCallRelation]) -> bool:
    return bool(relations) and all(
        relation.answer_call_provenance_status == "explicit_matched" for relation in relations
    )


def finalize_answer_call(
    observer: AnswerCallProvenanceObserver | None,
    recorded: RecordedResponse,
    final_source_ids: tuple[str, ...],
    parsed_output: str | None,
    verifier_result: bool | None,
) -> AnswerCallRelation | None:
    if observer is None:
        return None
    observer.record_answer(recorded)
    observer.record_context(recorded.call_id, recorded.response, final_source_ids)
    observer.record_parser(recorded.call_id, recorded.call_id, recorded.response, parsed_output)
    if verifier_result is not None:
        observer.record_verifier(recorded.call_id, recorded.response, verifier_result)
    return observer.finalize(recorded.call_id)


def finalize_failed_answer_call(
    observer: AnswerCallProvenanceObserver | None, failed_call: FailedRecordedCall
) -> AnswerCallRelation | None:
    if observer is None:
        return None
    return observer.finalize(failed_call.call_id)


def _relation(
    status: RelationStatus,
    answer_call_id: str,
    parser: ParsedResponseEvent | None = None,
    verifier: VerifiedResponseEvent | None = None,
) -> AnswerCallRelation:
    match status:
        case "explicit_matched":
            assert parser is not None and verifier is not None
            payload = {
                "answer_call_provenance_status": status,
                "answer_call_id": answer_call_id,
                "parsed_response_source_call_id": parser.parsed_response_source_call_id,
                "parser_result_id": parser.parser_result_id,
                "verifier_result_id": verifier.verifier_result_id,
            }
        case "missing" | "ambiguous" | "historically_reconstructed" | "mismatched":
            payload = {"answer_call_provenance_status": status, "answer_call_id": answer_call_id}
        case unreachable:
            assert_never(unreachable)
    return _RELATION_ADAPTER.validate_python(payload)
