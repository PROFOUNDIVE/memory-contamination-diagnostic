from __future__ import annotations

from typing import Any, Callable, Sequence

from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.no_memory import NoMemoryPolicy, _failed
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance
from memcontam.tools.base import ToolExecutionError, ToolExecutor, ToolPolicyError, ToolRuntimeContract
from memcontam.tools.execution_loop import LlmCall, ToolProtocolError, run_tool_loop


class NoMemoryContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class NoMemoryToolAdapter:
    def execute(
        self,
        task: TaskInstance,
        memory: MemoryState,
        *,
        client: LLMClient,
        model: str,
        executor: ToolExecutor,
        policy: ToolRuntimeContract,
        config: dict[str, Any] | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None = None,
        memory_events: Sequence[Any] = (),
    ) -> BaselineExecutionOutcome:
        if memory.entries:
            raise NoMemoryContractError("NOMEM_MEMORY_STATE_FORBIDDEN")
        if memory_events:
            raise NoMemoryContractError("NOMEM_MEMORY_EVENTS_FORBIDDEN")

        config = dict(config or {})
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
        messages = NoMemoryPolicy().build_prompt(task, MemoryState())
        messages[0] = {
            "role": "system",
            "content": (
                "Solve the task. Use no persistent memory. You may use the Python sandbox. "
                'Return exactly one JSON action: {"action":"execute_python","code":"..."} '
                'or {"action":"final","answer":"final: ..."}. '
            ),
        }
        try:
            response = recorder.chat(
                messages,
                model=model,
                config={
                    **config,
                    "sample_id": config.get("sample_id", task.sample_id),
                    "method_stage": "no_memory_generate",
                },
            )
        except Exception:
            return _failed(
                recorder,
                (),
                "ProviderCallFailure",
                "provider_call_failed",
                "provider_call_failed",
                metadata={"memory_events": (), "tool_events": ()},
            )

        records = recorder.get_records()
        initial_call_id = records[-1].call_id if records else None
        if initial_call_id is None:
            return _failed(
                recorder,
                (),
                "BaselineOutputError",
                "no_memory_invalid_final_answer",
                "invalid_final_answer",
                metadata={
                    "memory_events": (),
                    "tool_error_code": "MISSING_INITIAL_CALL",
                    "tool_events": (),
                },
            )
        initial_call = LlmCall(
            call_id=initial_call_id,
            content=response.content,
            messages=messages,
            model=model,
            config={
                **config,
                "sample_id": config.get("sample_id", task.sample_id),
                "method_stage": "no_memory_generate",
            },
            run_id=str(config.get("run_id", "unknown")),
            trial_id=trial_id,
            max_rounds=int(config.get("max_tool_rounds", 3)),
        )
        try:
            tool_result = run_tool_loop(initial_call, recorder, executor, policy)
        except (ToolExecutionError, ToolPolicyError, ToolProtocolError) as error:
            return _failed(
                recorder,
                (),
                "BaselineOutputError",
                "no_memory_invalid_final_answer",
                "invalid_final_answer",
                metadata={"memory_events": (), "tool_error_code": error.code, "tool_events": ()},
            )

        try:
            parsed_answer = parse_final_answer(tool_result.answer)
        except ValueError:
            parsed_answer = ""
        metadata = {"memory_events": (), "tool_events": tool_result.tool_events}
        if not parsed_answer:
            return _failed(
                recorder,
                (),
                "BaselineOutputError",
                "no_memory_invalid_final_answer",
                "invalid_final_answer",
                final_response=tool_result.answer,
                answer_call_id=tool_result.answer_call_id,
                metadata=metadata,
            )
        try:
            verifier_result = (
                verifier(parsed_answer, task)
                if verifier
                else VerifierResult(is_correct=True, parsed_answer=parsed_answer)
            )
        except Exception:
            return _failed(
                recorder,
                (),
                "VerifierContractError",
                "verifier_contract_failed",
                "verifier_contract_failed",
                final_response=tool_result.answer,
                parsed_answer=parsed_answer,
                answer_call_id=tool_result.answer_call_id,
                metadata=metadata,
            )
        return BaselineExecutionOutcome(
            status="succeeded",
            final_response=tool_result.answer,
            parsed_answer=parsed_answer,
            verifier_result=verifier_result,
            answer_call_id=tool_result.answer_call_id,
            method_calls=tuple(recorder.get_records()),
            memory_before=(),
            memory_after=(),
            metadata=metadata,
        )


__all__ = ["NoMemoryContractError", "NoMemoryToolAdapter"]
