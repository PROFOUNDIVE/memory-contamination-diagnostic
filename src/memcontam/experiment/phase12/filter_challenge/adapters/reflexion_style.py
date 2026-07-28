from __future__ import annotations

from dataclasses import dataclass, replace

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.reflexion_phase12 import (
    ReflexionPhase12Adapter,
    ReflexionStateV3,
    ReflexionTrialContextV3,
)
from memcontam.experiment.phase12.filter_challenge.adapters.base import FrozenCheckpoint
from memcontam.experiment.phase12.filter_challenge.contracts import (
    CandidateExposureRecord,
    ChallengeCandidate,
)
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import (
    NATIVE_ENTRY_V1,
    NativeEntry,
    Phase12Checkpoint,
    deserialize_checkpoint,
)
from memcontam.memory.stores import MemoryEntry


class ReflexionChallengeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ReflexionFrozenCheckpoint:
    source_checkpoint: Phase12Checkpoint
    trial: ReflexionTrialContextV3

    def snapshot_id(self) -> str:
        return self.source_checkpoint.identity.checkpoint_id


class ReflexionProvisionalResult(CandidateExposureRecord):
    outcome: BaselineExecutionOutcome
    final_context_source_ids: tuple[str, ...]
    displaced_reflection_ids: tuple[str, ...]


class ReflexionProvisionalAdapter:
    def execute(
        self,
        checkpoint: FrozenCheckpoint,
        candidate: ChallengeCandidate,
    ) -> ReflexionProvisionalResult:
        frozen_checkpoint = _require_reflexion_checkpoint(checkpoint)
        _validate_candidate(frozen_checkpoint, candidate)
        source_state = reflexion_source_state(frozen_checkpoint.source_checkpoint)
        native_candidate = _native_candidate(candidate)
        reflections = [*source_state.reflections, native_candidate]
        displaced_reflection_ids: list[str] = []
        if source_state.active_capacity is not None:
            while len(reflections) > source_state.active_capacity:
                displaced_reflection_ids.append(reflections.pop(0).entry_id)
        provisional_state = ReflexionStateV3(
            reflections=reflections,
            active_capacity=source_state.active_capacity,
        )
        result = ReflexionPhase12Adapter().execute(
            replace(frozen_checkpoint.trial, config={**frozen_checkpoint.trial.config, "update_enabled": False}),
            provisional_state,
        )
        final_context_source_ids = _answer_source_ids(result.outcome)
        return ReflexionProvisionalResult(
            candidate_entry_id=candidate.candidate_entry_id,
            candidate_final_context_inclusion=candidate.candidate_entry_id in final_context_source_ids,
            candidate_final_context_source_ids=final_context_source_ids,
            outcome=result.outcome,
            final_context_source_ids=final_context_source_ids,
            displaced_reflection_ids=tuple(displaced_reflection_ids),
        )


def _require_reflexion_checkpoint(checkpoint: FrozenCheckpoint) -> ReflexionFrozenCheckpoint:
    if not isinstance(checkpoint, ReflexionFrozenCheckpoint):
        raise ReflexionChallengeError("REFLEXION_CHECKPOINT_INVALID")
    return checkpoint


def _validate_candidate(
    checkpoint: ReflexionFrozenCheckpoint, candidate: ChallengeCandidate
) -> None:
    if candidate.baseline_family != "reflexion_style":
        raise ReflexionChallengeError("REFLEXION_BASELINE_MISMATCH")
    if candidate.rag_mode != "not_applicable":
        raise ReflexionChallengeError("REFLEXION_RAG_MODE_MISMATCH")
    if candidate.candidate_native_kind != "verbal_reflection":
        raise ReflexionChallengeError("REFLEXION_NATIVE_KIND_MISMATCH")
    if candidate.source_checkpoint_id != checkpoint.snapshot_id():
        raise ReflexionChallengeError("REFLEXION_CHECKPOINT_ID_MISMATCH")
    if candidate.source_active_state_hash != checkpoint.source_checkpoint.canonical_sha256:
        raise ReflexionChallengeError("REFLEXION_CHECKPOINT_HASH_MISMATCH")
    if checkpoint.source_checkpoint.identity.baseline != "reflexion_style":
        raise ReflexionChallengeError("REFLEXION_CHECKPOINT_BASELINE_MISMATCH")
    if candidate.routability.routability != "challenge_routable_v1":
        raise ReflexionChallengeError("REFLEXION_ROUTABILITY_MISMATCH")


def reflexion_source_state(checkpoint: Phase12Checkpoint) -> ReflexionStateV3:
    state = deserialize_checkpoint(checkpoint)
    if state.baseline != "reflexion_style":
        raise ReflexionChallengeError("REFLEXION_CHECKPOINT_STATE_INVALID")
    reflections: list[MemoryEntry | NativeEntry] = []
    for entry in state.entries:
        if not isinstance(entry, NativeEntry):
            raise ReflexionChallengeError("REFLEXION_CHECKPOINT_STATE_INVALID")
        reflections.append(entry)
    capacity = state.native_state.get("active_capacity")
    if capacity is not None and (type(capacity) is not int or capacity < 1):
        raise ReflexionChallengeError("REFLEXION_CHECKPOINT_CAPACITY_INVALID")
    return ReflexionStateV3(
        reflections=reflections,
        active_capacity=capacity,
    )


def _native_candidate(candidate: ChallengeCandidate) -> NativeEntry:
    return NativeEntry(
        entry_id=candidate.candidate_entry_id,
        semantic_kind=candidate.candidate_native_kind,
        schema_version=NATIVE_ENTRY_V1,
        native_component="reflections",
        content=candidate.candidate_native_content,
        content_hash=canonical_content_hash(candidate.candidate_native_content),
    )


def _answer_source_ids(outcome: BaselineExecutionOutcome) -> tuple[str, ...]:
    if outcome.answer_call_id is None:
        return ()
    answer_calls = [call for call in outcome.method_calls if call.call_id == outcome.answer_call_id]
    if len(answer_calls) != 1:
        raise ReflexionChallengeError("REFLEXION_ANSWER_CALL_MISMATCH")
    return tuple(span.entry_id for span in answer_calls[0].source_spans)
