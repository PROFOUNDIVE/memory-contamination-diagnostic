from __future__ import annotations

from typing import Any, Callable

from memcontam.baselines.reflexion_adapter import (
    ReflectionGenerationResult,
    ReflectionPayload,
    ReflexionAdapter,
    ReflexionState,
    record_attempt_outcome,
    record_reflection_event,
)
from memcontam.clients.base import LLMClient
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState, apply_keep_last_3
from memcontam.tasks.base import TaskInstance

__all__ = [
    "ReflectionGenerationResult",
    "ReflectionPayload",
    "ReflexionAdapter",
    "ReflexionState",
    "ReflexionStylePolicy",
    "apply_keep_last_3",
    "record_attempt_outcome",
    "record_reflection_event",
]


class ReflexionStylePolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        reflections = "\n".join(entry.content for entry in memory.entries[-3:])
        return [{"role": "user", "content": f"Reflections:\n{reflections}\n\nSolve: {task.input}"}]

    def run(
        self,
        task: TaskInstance,
        memory: MemoryState,
        *,
        client: LLMClient,
        model: str,
        config: dict[str, Any] | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None = None,
    ) -> dict[str, Any]:
        memory_before = [entry.model_dump() for entry in memory.entries]
        original_reflections = _reflection_entries(memory)
        state = ReflexionState(reflections=list(original_reflections))
        captured_verifier_result: VerifierResult | bool | None = None

        def capture_verifier(answer: str, seen_task: TaskInstance) -> VerifierResult | bool:
            nonlocal captured_verifier_result
            if verifier is None:
                return True
            captured_verifier_result = verifier(answer, seen_task)
            return captured_verifier_result

        outcome = ReflexionAdapter().execute(
            task,
            state,
            client=client,
            model=model,
            config=config,
            verifier=capture_verifier,
        )
        if state.reflections != original_reflections:
            memory.entries = [entry for entry in memory.entries if entry not in original_reflections]
            memory.entries.extend(state.reflections)
        verifier_result = captured_verifier_result if captured_verifier_result is not None else outcome.verifier_result
        if isinstance(verifier_result, bool):
            verifier_result = VerifierResult(
                is_correct=verifier_result, parsed_answer=outcome.parsed_answer
            )
        return {
            "status": outcome.status,
            "final_response": outcome.final_response,
            "parsed_answer": outcome.parsed_answer,
            "verifier_result": verifier_result,
            "method_calls": list(outcome.method_calls),
            "memory_before": memory_before,
            "memory_after": [entry.model_dump() for entry in memory.entries],
            "memory_write_event": outcome.memory_write_event,
            "metadata": outcome.metadata,
            "retrieved_records": [],
            "retrieved_memory": [],
            "retrieved_scores": [],
            "answer_call_id": outcome.answer_call_id,
            "error_type": outcome.error_type,
            "failure_disposition": outcome.failure_disposition,
            "scientific_ineligibility_reason": outcome.scientific_ineligibility_reason,
        }


def _reflection_entries(memory: MemoryState) -> list[MemoryEntry]:
    return [
        entry
        for entry in memory.entries
        if entry.memory_type == "verbal_reflection" or entry.content.startswith("Reflection:")
    ]
