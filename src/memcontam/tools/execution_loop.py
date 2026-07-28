from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping, Protocol

from memcontam.clients.recording import FailedRecordedCall, RecordedCallFailure, RecordedResponse
from memcontam.logging.schema_v3 import ToolEvent
from memcontam.tools.base import ToolExecutionError, ToolExecutor, ToolRequest, ToolRuntimeContract
from memcontam.tools.python_sandbox import PythonSandbox


class ToolProtocolError(RuntimeError):
    def __init__(
        self,
        code: str,
        recorded_response: RecordedResponse | None = None,
        failed_call: FailedRecordedCall | None = None,
    ) -> None:
        self.code = code
        self.recorded_response = recorded_response
        self.failed_call = failed_call
        super().__init__(code)


class ToolTimeoutError(ToolExecutionError):
    pass


class RecordedCallClient(Protocol):
    def chat_with_call_id(
        self, messages: list[dict[str, str]], model: str, config: dict
    ) -> RecordedResponse: ...


@dataclass(frozen=True)
class LlmCall:
    call_id: str
    content: str
    messages: list[dict[str, str]]
    model: str
    config: Mapping[str, Any]
    run_id: str
    trial_id: str
    max_rounds: int = 3
    recorded_response: RecordedResponse | None = None

    def __post_init__(self) -> None:
        if not all((self.call_id, self.run_id, self.trial_id)) or self.max_rounds < 1:
            raise ToolProtocolError("INVALID_TOOL_LOOP_CALL")
        if self.recorded_response is not None and self.recorded_response.call_id != self.call_id:
            raise ToolProtocolError("RECORDED_CALL_ID_MISMATCH")


@dataclass(frozen=True)
class ToolAugmentedResult:
    answer: str
    answer_call_id: str
    tool_events: tuple[ToolEvent, ...]
    recorded_response: RecordedResponse | None


def run_tool_loop(
    initial_call: LlmCall,
    client: RecordedCallClient,
    executor: ToolExecutor,
    policy: ToolRuntimeContract,
    *,
    writer: Any | None = None,
) -> ToolAugmentedResult:
    """Run the only supported code-tool protocol until its final action."""
    sandbox = PythonSandbox(policy, executor=executor)
    current_call = initial_call
    events: list[ToolEvent] = []

    for round_index in range(1, initial_call.max_rounds + 2):
        try:
            action = _parse_action(current_call.content)
        except ToolProtocolError as error:
            raise ToolProtocolError(error.code, current_call.recorded_response) from error
        if action["action"] == "final":
            return ToolAugmentedResult(
                action["answer"], current_call.call_id, tuple(events), current_call.recorded_response
            )
        if round_index > initial_call.max_rounds:
            raise ToolProtocolError("MAX_TOOL_ROUNDS_EXCEEDED", current_call.recorded_response)

        request = ToolRequest(action["code"], timeout_seconds=action.get("timeout_seconds"))
        started_at = time.monotonic_ns()
        try:
            result = sandbox.execute(request)
        except ToolExecutionError as error:
            if error.code == "SANDBOX_TIMEOUT":
                raise ToolTimeoutError("TOOL_TIMEOUT") from error
            raise
        duration_ms = (time.monotonic_ns() - started_at) // 1_000_000
        continuation = _continuation_call(client, current_call, result)
        event = ToolEvent(
            record_type="tool_event",
            event_id=f"{current_call.call_id}:tool:{round_index}",
            run_id=initial_call.run_id,
            trial_id=initial_call.trial_id,
            event_seq=0,
            tool_mode="python_sandbox",
            action="execute_python",
            code_hash=hashlib.sha256(request.code.encode("utf-8")).hexdigest(),
            output=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            status="completed",
            duration_ms=duration_ms,
            executor_identity=result.executor_identity,
            parent_call_id=current_call.call_id,
            continuation_call_id=continuation.call_id,
        )
        events.append(_write_event(writer, event))
        current_call = continuation

    raise ToolProtocolError("MAX_TOOL_ROUNDS_EXCEEDED", current_call.recorded_response)


def _parse_action(content: str) -> dict[str, Any]:
    try:
        action = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise ToolProtocolError("MALFORMED_ACTION") from error
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise ToolProtocolError("MALFORMED_ACTION")
    if action["action"] == "final":
        if (
            set(action) != {"action", "answer"}
            or not isinstance(action.get("answer"), str)
            or not action["answer"].strip()
        ):
            raise ToolProtocolError("MALFORMED_FINAL")
        return action
    if action["action"] != "execute_python":
        raise ToolProtocolError("UNKNOWN_ACTION")
    if set(action) - {"action", "code", "timeout_seconds"} or not isinstance(
        action.get("code"), str
    ):
        raise ToolProtocolError("MALFORMED_ACTION")
    timeout = action.get("timeout_seconds")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))):
        raise ToolProtocolError("MALFORMED_ACTION")
    return action


def _continuation_call(client: RecordedCallClient, parent: LlmCall, result: Any) -> LlmCall:
    messages = [
        *parent.messages,
        {"role": "assistant", "content": parent.content},
        {
            "role": "user",
            "content": json.dumps(
                {"exit_code": result.exit_code, "stderr": result.stderr, "stdout": result.stdout},
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    ]
    try:
        recorded_response = client.chat_with_call_id(messages, parent.model, dict(parent.config))
    except RecordedCallFailure as error:
        raise ToolProtocolError("MISSING_CONTINUATION", failed_call=error.failed_call) from error
    except Exception as error:
        raise ToolProtocolError("MISSING_CONTINUATION") from error
    return LlmCall(
        call_id=recorded_response.call_id,
        content=recorded_response.response.content,
        messages=messages,
        model=parent.model,
        config=parent.config,
        run_id=parent.run_id,
        trial_id=parent.trial_id,
        max_rounds=parent.max_rounds,
        recorded_response=recorded_response,
    )


def _write_event(writer: Any | None, event: ToolEvent) -> ToolEvent:
    if writer is None:
        return event
    payload = writer.append_event(event)
    return ToolEvent.model_validate(payload)


__all__ = [
    "LlmCall",
    "ToolAugmentedResult",
    "ToolProtocolError",
    "ToolTimeoutError",
    "run_tool_loop",
]
