from __future__ import annotations

from typing import Any, Callable

from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


class NoMemoryPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"Solve this {task.task_name} instance: {task.input}"}]

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
        config = dict(config or {})
        messages = self.build_prompt(task, memory)
        trial_id = ":".join(
            [
                str(config.get("run_id", "unknown")),
                task.task_name,
                task.sample_id,
                str(config.get("baseline", "no_memory")),
                str(config.get("arm", "clean")),
                str(config.get("model", model)),
            ]
        )
        recorder = MethodCallRecorder(
            client,
            event_callback=config.get("_logging_event_callback"),
            trial_context={**config.get("_logging_trial_context", {}), "trial_id": trial_id},
        )
        response = recorder.chat(
            messages,
            model=model,
            config={
                **config,
                "sample_id": config.get("sample_id", task.sample_id),
                "method_stage": "no_memory_generate",
            },
        )
        parsed_answer = _parse_answer(response.content)
        verifier_result = (
            verifier(parsed_answer, task)
            if verifier is not None
            else VerifierResult(is_correct=True, parsed_answer=parsed_answer)
        )
        if verifier_result.parsed_answer is None:
            verifier_result.parsed_answer = parsed_answer
        method_calls = recorder.get_records()
        answer_call_id = method_calls[0].call_id if method_calls else None
        return {
            "final_response": response.content,
            "parsed_answer": parsed_answer,
            "verifier_result": verifier_result,
            "method_calls": method_calls,
            "memory_before": [entry.model_dump() for entry in memory.entries],
            "memory_after": [entry.model_dump() for entry in memory.entries],
            "memory_write_event": None,
            "metadata": {},
            "retrieved_records": [],
            "retrieved_scores": [],
            "answer_call_id": answer_call_id,
        }


def _parse_answer(response: str) -> str:
    response = response.strip()
    if ":" in response:
        return response.split(":", 1)[1].strip()
    return response
