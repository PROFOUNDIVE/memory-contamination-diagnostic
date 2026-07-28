from __future__ import annotations

from dataclasses import dataclass, replace

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.reflexion_phase12 import (
    ReflexionPhase12Adapter,
    ReflexionStateV3,
    ReflexionTrialContextV3,
)
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.memory.stores import MemoryEntry


@dataclass(frozen=True, slots=True)
class ReflexionProvisionalResult:
    outcome: BaselineExecutionOutcome
    final_context_source_ids: tuple[str, ...]
    candidate_final_context_inclusion: bool
    displaced_reflection_ids: tuple[str, ...]


class ReflexionProvisionalAdapter:
    def execute(
        self,
        trial: ReflexionTrialContextV3,
        source_state: ReflexionStateV3,
        candidate: NativeEntry,
    ) -> ReflexionProvisionalResult:
        reflections = [*source_state.reflections, candidate]
        displaced_reflections: list[MemoryEntry | NativeEntry] = []
        if source_state.active_capacity is not None:
            while len(reflections) > source_state.active_capacity:
                displaced_reflections.append(reflections.pop(0))
        provisional_state = ReflexionStateV3(
            reflections=reflections,
            active_capacity=source_state.active_capacity,
        )
        result = ReflexionPhase12Adapter().execute(
            replace(trial, config={**trial.config, "update_enabled": False}),
            provisional_state,
        )
        answer_call = next(
            call for call in result.outcome.method_calls if call.call_id == result.outcome.answer_call_id
        )
        final_context_source_ids = tuple(span.entry_id for span in answer_call.source_spans)
        return ReflexionProvisionalResult(
            outcome=result.outcome,
            final_context_source_ids=final_context_source_ids,
            candidate_final_context_inclusion=candidate.entry_id in final_context_source_ids,
            displaced_reflection_ids=tuple(entry.entry_id for entry in displaced_reflections),
        )
