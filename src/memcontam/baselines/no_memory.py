from __future__ import annotations

from typing import Any, Callable

from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import (
    BaselineExecutionOutcome,
    ErrorType,
    FailureDisposition,
    ScientificIneligibilityReason,
)
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


class NoMemoryPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        if memory.entries:
            raise ValueError("no_memory is read-only and requires empty memory")
        return [
            {
                "role": "system",
                "content": "Solve the task. Use no persistent memory. Return only the final answer in the required task format.",
            },
            {
                "role": "user",
                "content": f"Task family:\n{task.task_name}\n\nCurrent task:\n{canonical_task_json(task)}",
            },
        ]

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
        outcome = NoMemoryAdapter().execute(
            task, memory, client=client, model=model, config=config, verifier=verifier
        )
        return {
            "status": outcome.status,
            "final_response": outcome.final_response,
            "parsed_answer": outcome.parsed_answer,
            "verifier_result": outcome.verifier_result,
            "method_calls": list(outcome.method_calls),
            "memory_before": list(outcome.memory_before),
            "memory_after": list(outcome.memory_after),
            "memory_write_event": outcome.memory_write_event,
            "metadata": outcome.metadata,
            "retrieved_records": [],
            "retrieved_scores": [],
            "answer_call_id": outcome.answer_call_id,
            "error_type": outcome.error_type,
            "failure_disposition": outcome.failure_disposition,
            "scientific_ineligibility_reason": outcome.scientific_ineligibility_reason,
        }


class NoMemoryAdapter:
    def execute(
        self,
        task: TaskInstance,
        memory: MemoryState,
        *,
        client: LLMClient,
        model: str,
        config: dict[str, Any] | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None = None,
    ) -> BaselineExecutionOutcome:
        config = dict(config or {})
        messages = NoMemoryPolicy().build_prompt(task, memory)
        memory_before = tuple(entry.model_dump() for entry in memory.entries)
        trial_id = ":".join(
            [str(config.get("run_id", "unknown")), task.task_name, task.sample_id,
             str(config.get("baseline", "no_memory")), str(config.get("arm", "clean")),
             str(config.get("model", model))]
        )
        recorder = MethodCallRecorder(
            client,
            event_callback=config.get("_logging_event_callback"),
            trial_context={**config.get("_logging_trial_context", {}), "trial_id": trial_id},
        )
        try:
            response = recorder.chat(messages, model=model, config={**config, "sample_id": config.get("sample_id", task.sample_id), "method_stage": "no_memory_generate"})
        except Exception:
            return _failed(recorder, memory_before, "ProviderCallFailure", "provider_call_failed", "provider_call_failed")
        parsed_answer = parse_final_answer(response.content)
        answer_call_id = recorder.get_records()[0].call_id if recorder.get_records() else None
        if not parsed_answer:
            return _failed(recorder, memory_before, "BaselineOutputError", "no_memory_invalid_final_answer", "invalid_final_answer", final_response=response.content, answer_call_id=answer_call_id)
        try:
            verifier_result = verifier(parsed_answer, task) if verifier else VerifierResult(is_correct=True, parsed_answer=parsed_answer)
        except Exception:
            return _failed(recorder, memory_before, "VerifierContractError", "verifier_contract_failed", "verifier_contract_failed", final_response=response.content, parsed_answer=parsed_answer, answer_call_id=answer_call_id)
        return BaselineExecutionOutcome(status="succeeded", final_response=response.content, parsed_answer=parsed_answer, verifier_result=verifier_result, answer_call_id=answer_call_id, method_calls=tuple(recorder.get_records()), memory_before=memory_before, memory_after=memory_before)


def _failed(
    recorder: MethodCallRecorder,
    memory: tuple[dict[str, Any], ...],
    error_type: ErrorType,
    failure_disposition: FailureDisposition,
    reason: ScientificIneligibilityReason,
    **values: Any,
) -> BaselineExecutionOutcome:
    return BaselineExecutionOutcome(
        status="failed",
        method_calls=tuple(recorder.get_records()),
        memory_before=memory,
        memory_after=memory,
        error_type=error_type,
        failure_disposition=failure_disposition,
        scientific_ineligibility_reason=reason,
        **values,
    )
