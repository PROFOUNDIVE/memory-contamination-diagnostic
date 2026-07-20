from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable

from memcontam.clients.base import LLMClient
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


@dataclass(frozen=True)
class FullHistoryPayload:
    task_input: str
    raw_response: str


@dataclass
class FullHistoryState:
    records: list[MemoryEntry] = field(default_factory=list)


def render_full_history(entry_id: str, payload: FullHistoryPayload) -> str:
    return (
        f'<BEGIN_HISTORY_RECORD id="{entry_id}">\n'
        f"TASK:\n{payload.task_input}\n\n"
        f"RESPONSE:\n{payload.raw_response}\n"
        "<END_HISTORY_RECORD>"
    )


class FullHistoryPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        return import_module("memcontam.baselines.full_history_adapter")._messages(
            task, FullHistoryState(records=memory.entries)
        )[0]

    def run(
        self,
        task: TaskInstance,
        memory: MemoryState,
        *,
        client: LLMClient,
        model: str,
        config: dict[str, Any] | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult] | None = None,
    ) -> dict[str, Any]:
        state = FullHistoryState(records=memory.entries)
        captured_verifier_result: VerifierResult | None = None

        def capture_verifier(answer: str, seen_task: TaskInstance) -> VerifierResult | bool:
            nonlocal captured_verifier_result
            if verifier is None:
                return True
            captured_verifier_result = verifier(answer, seen_task)
            return captured_verifier_result

        outcome = import_module("memcontam.baselines.full_history_adapter").FullHistoryAdapter().execute(
            task,
            state,
            client=client,
            model=model,
            config=config,
            verifier=capture_verifier,
        )
        memory.entries = state.records
        verifier_result = captured_verifier_result or outcome.verifier_result
        if isinstance(verifier_result, bool):
            verifier_result = VerifierResult(
                is_correct=verifier_result,
                parsed_answer=outcome.parsed_answer,
            )
        memory_write_event = outcome.memory_write_event
        if memory_write_event is not None:
            memory_write_event = {**memory_write_event, "parent_entry_ids": []}
        return {
            "final_response": outcome.final_response,
            "parsed_answer": outcome.parsed_answer,
            "verifier_result": verifier_result,
            "method_calls": list(outcome.method_calls),
            "memory_before": list(outcome.memory_before),
            "memory_after": list(outcome.memory_after),
            "memory_write_event": memory_write_event,
            "metadata": outcome.metadata,
            "retrieved_records": [],
            "retrieved_scores": [],
            "answer_call_id": outcome.answer_call_id,
        }


def __getattr__(name: str) -> Any:
    if name == "FullHistoryAdapter":
        return import_module("memcontam.baselines.full_history_adapter").FullHistoryAdapter
    raise AttributeError(name)
