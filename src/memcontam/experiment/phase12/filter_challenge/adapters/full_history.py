from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.full_history import FullHistoryPolicy
from memcontam.baselines.full_history_context import render_context_bounded_history
from memcontam.clients.base import LLMClient
from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeCandidate,
)
from memcontam.experiment.phase12.filter_challenge.provenance import (
    AnswerCallProvenanceObserver,
)
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import (
    NATIVE_ENTRY_V1,
    NativeEntry,
    NativeState,
    Phase12Checkpoint,
    append_native_entry,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


class FullHistoryChallengeError(ValueError):
    pass


class _RelationObserver(AnswerCallProvenanceObserver):
    def __init__(self) -> None:
        super().__init__()
        self.relations: dict[str, AnswerCallRelation] = {}

    def finalize(self, answer_call_id: str) -> AnswerCallRelation:
        relation = super().finalize(answer_call_id)
        self.relations[answer_call_id] = relation
        return relation


@dataclass(frozen=True, slots=True)
class FullHistoryPairRequest:
    task: TaskInstance
    checkpoint: Phase12Checkpoint
    candidate: ChallengeCandidate
    control_client: LLMClient
    challenge_client: LLMClient
    model: str
    context_config: Mapping[str, object]
    verifier: Callable[[str, TaskInstance], bool] | None = None


@dataclass(frozen=True, slots=True)
class FullHistoryPairResult:
    control_outcome: BaselineExecutionOutcome
    challenge_outcome: BaselineExecutionOutcome
    control_final_source_ids: tuple[str, ...]
    challenge_final_source_ids: tuple[str, ...]
    challenge_removed_entry_ids: tuple[str, ...]
    candidate_exposure: CandidateExposureRecord
    source_checkpoint_sha256_after: str


class FullHistoryProvisionalAdapter:
    def execute(self, request: FullHistoryPairRequest) -> FullHistoryPairResult:
        source_state = deserialize_checkpoint(request.checkpoint)
        _require_full_history_state(source_state)
        candidate_entry = _candidate_entry(request.candidate)
        provisional_checkpoint = append_native_entry(request.checkpoint, candidate_entry)
        control_records = _records(source_state)
        challenge_state = deserialize_checkpoint(provisional_checkpoint)
        challenge_records = _records(challenge_state)
        control_outcome = _execute_read_only(request, request.control_client, control_records)
        challenge_outcome = _execute_read_only(request, request.challenge_client, challenge_records)
        control_context = render_context_bounded_history(
            request.task, control_records, request.context_config
        )
        challenge_context = render_context_bounded_history(
            request.task, challenge_records, request.context_config
        )
        control_source_ids = _answer_source_ids(control_outcome, tuple(control_context.post_record_ids))
        challenge_source_ids = _answer_source_ids(
            challenge_outcome, tuple(challenge_context.post_record_ids)
        )
        candidate_exposure = CandidateExposureRecord(
            candidate_entry_id=request.candidate.candidate_entry_id,
            candidate_final_context_inclusion=request.candidate.candidate_entry_id in challenge_source_ids,
            candidate_final_context_source_ids=challenge_source_ids,
        )
        source_checkpoint_after = serialize_checkpoint(deserialize_checkpoint(request.checkpoint))
        return FullHistoryPairResult(
            control_outcome=control_outcome,
            challenge_outcome=challenge_outcome,
            control_final_source_ids=control_source_ids,
            challenge_final_source_ids=challenge_source_ids,
            challenge_removed_entry_ids=tuple(challenge_context.removed_record_ids),
            candidate_exposure=candidate_exposure,
            source_checkpoint_sha256_after=source_checkpoint_after.canonical_sha256,
        )


def _execute_read_only(
    request: FullHistoryPairRequest, client: LLMClient, records: list[MemoryEntry]
) -> BaselineExecutionOutcome:
    observer = _RelationObserver()
    outcome = FullHistoryPolicy().execute(
        request.task,
        MemoryState(entries=list(records)),
        client=client,
        model=request.model,
        config={
            **request.context_config,
            "_logging_answer_call_provenance_observer": observer,
        },
        verifier=request.verifier,
        update_enabled=False,
    )
    answer_call_id = outcome.answer_call_id
    if (
        answer_call_id is None
        or len(observer.relations) != 1
        or answer_call_id not in observer.relations
        or observer.relations[answer_call_id].answer_call_provenance_status != "explicit_matched"
    ):
        raise FullHistoryChallengeError("ANSWER_CALL_RELATION_MISMATCH")
    return outcome


def _answer_source_ids(
    outcome: BaselineExecutionOutcome, expected_source_ids: tuple[str, ...]
) -> tuple[str, ...]:
    answer_call_id = outcome.answer_call_id
    if answer_call_id is None:
        raise FullHistoryChallengeError("ANSWER_CALL_BINDING")
    answer_calls = [call for call in outcome.method_calls if call.call_id == answer_call_id]
    if len(answer_calls) != 1:
        raise FullHistoryChallengeError("ANSWER_CALL_BINDING")
    source_ids = tuple(span.entry_id for span in answer_calls[0].source_spans)
    if source_ids != expected_source_ids:
        raise FullHistoryChallengeError("ANSWER_SOURCE_SPAN_MISMATCH")
    return source_ids


def _require_full_history_state(state: NativeState) -> None:
    if state.baseline != "fh_bounded":
        raise FullHistoryChallengeError("INVALID_FULL_HISTORY_CHECKPOINT")


def _candidate_entry(candidate: ChallengeCandidate) -> NativeEntry:
    if candidate.baseline_family != "full_history":
        raise FullHistoryChallengeError("INVALID_FULL_HISTORY_CANDIDATE")
    if candidate.candidate_native_kind != "full_history_transcript":
        raise FullHistoryChallengeError("INVALID_FULL_HISTORY_CANDIDATE")
    return NativeEntry(
        entry_id=candidate.candidate_entry_id,
        semantic_kind=candidate.candidate_native_kind,
        schema_version=NATIVE_ENTRY_V1,
        native_component="history",
        content=candidate.candidate_native_content,
        content_hash=canonical_content_hash(candidate.candidate_native_content),
    )


def _records(state: NativeState) -> list[MemoryEntry]:
    records: list[MemoryEntry] = []
    for entry in state.entries:
        match entry:
            case NativeEntry(
                entry_id=entry_id,
                semantic_kind="full_history_transcript",
                native_component="history",
                content=content,
            ):
                records.append(
                    MemoryEntry(
                        entry_id=entry_id,
                        content=content,
                        memory_type="full_history_transcript",
                    )
                )
            case NativeEntry():
                raise FullHistoryChallengeError("INVALID_FULL_HISTORY_ENTRY")
            case str():
                raise FullHistoryChallengeError("INVALID_FULL_HISTORY_ENTRY")
    return records
