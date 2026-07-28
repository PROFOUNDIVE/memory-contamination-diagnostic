from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.experiment.phase12.filter_challenge.adapters.bot_style import (
    BoTChallengeExecution,
)
from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeCandidate,
)
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    ControlCacheValue,
    PairArmResult,
    PairExecutorError,
)
from memcontam.experiment.phase12.filter_challenge.provenance import (
    AnswerCallProvenanceObserver,
)
from memcontam.logging.schema import MethodCall

WriteEventValue = str | int | float | bool | None | list[str]


@dataclass(frozen=True, slots=True)
class NativePairResult:
    adapter_name: str
    control: PairArmResult
    challenge: PairArmResult
    cache_value: ControlCacheValue


def bot_execution(
    execution: BoTChallengeExecution, observer: AnswerCallProvenanceObserver
) -> BoTChallengeExecution:
    return replace(
        execution,
        config={
            **execution.config,
            "update_enabled": False,
            "_logging_answer_call_provenance_observer": observer,
        },
    )


def relation(
    outcome: BaselineExecutionOutcome, observer: AnswerCallProvenanceObserver
) -> AnswerCallRelation:
    if outcome.answer_call_id is None:
        raise PairExecutorError("ANSWER_CALL_RELATION_UNRESOLVED")
    return observer.finalized_relation(outcome.answer_call_id)


def exposure(
    candidate: ChallengeCandidate, final_source_ids: tuple[str, ...]
) -> CandidateExposureRecord:
    return CandidateExposureRecord(
        candidate_entry_id=candidate.candidate_entry_id,
        candidate_final_context_inclusion=candidate.candidate_entry_id in final_source_ids,
        candidate_final_context_source_ids=final_source_ids,
    )


def arm(
    outcome: BaselineExecutionOutcome,
    answer_relation: AnswerCallRelation,
    candidate_exposure: CandidateExposureRecord,
    displaced: tuple[str, ...],
) -> PairArmResult:
    write_event_id = writeback_id(outcome.memory_write_event)
    return PairArmResult(
        answer_relation,
        candidate_exposure,
        outcome.memory_before != outcome.memory_after or outcome.memory_write_event is not None,
        write_event_id,
        displaced,
    )


def writeback_id(event: Mapping[str, WriteEventValue] | None) -> str | None:
    if event is None:
        return None
    for key in ("event_id", "memory_id", "entry_id"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return "writeback-present"


def answer_source_ids(outcome: BaselineExecutionOutcome) -> tuple[str, ...]:
    answer_call_id = outcome.answer_call_id
    answer_calls = tuple(
        call
        for call in outcome.method_calls
        if isinstance(call, MethodCall) and call.call_id == answer_call_id
    )
    if answer_call_id is None or len(answer_calls) != 1:
        raise PairExecutorError("ANSWER_CALL_RELATION_UNRESOLVED")
    return tuple(span.entry_id for span in answer_calls[0].source_spans if span.entry_id is not None)


def without_candidate(
    entry_ids: tuple[str, ...], candidate: ChallengeCandidate
) -> tuple[str, ...]:
    return tuple(entry_id for entry_id in entry_ids if entry_id != candidate.candidate_entry_id)
