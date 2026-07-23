from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder
from memcontam.tools import SubprocessTestDouble, load_tool_runtime_contract
from memcontam.tools.execution_loop import (
    LlmCall,
    ToolProtocolError,
    ToolTimeoutError,
    run_tool_loop,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "containers" / "python-sandbox" / "image.lock.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "phase12" / "FX-TOOL-001.json"


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        if not self.responses:
            raise RuntimeError("no continuation")
        return LLMResponse(
            content=self.responses.pop(0), raw={}, token_usage={"total_tokens": 1}, latency_ms=1
        )


def _initial_call(recorder: MethodCallRecorder, *, max_rounds: int = 2) -> LlmCall:
    messages = [{"role": "user", "content": "Solve with JSON actions only."}]
    response = recorder.chat(messages, "gpt-replay", {"method_stage": "tool_generate"})
    record = recorder.get_records()[-1]
    assert record.call_id is not None
    return LlmCall(
        call_id=record.call_id,
        content=response.content,
        messages=messages,
        model="gpt-replay",
        config={"method_stage": "tool_generate"},
        run_id="run-1",
        trial_id="trial-1",
        max_rounds=max_rounds,
    )


def _policy():
    return load_tool_runtime_contract(LOCK_PATH, scientific=False)


def test_links_one_execution_to_one_continuation_and_final_answer() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    client = _FakeClient(
        [
            json.dumps({"action": "execute_python", "code": fixture["code"]}),
            json.dumps({"action": "final", "answer": "final: 43"}),
        ]
    )
    recorder = MethodCallRecorder(client, trial_context={"trial_id": "trial-1"})
    initial = _initial_call(recorder)

    result = run_tool_loop(initial, recorder, SubprocessTestDouble(), _policy())

    assert result.answer == "final: 43"
    assert result.answer_call_id == "trial-1:call:2"
    assert [record.call_id for record in recorder.get_records()] == ["trial-1:call:1", "trial-1:call:2"]
    assert len(result.tool_events) == 1
    event = result.tool_events[0]
    assert event.action == "execute_python"
    assert event.code_hash == hashlib.sha256(fixture["code"].encode("utf-8")).hexdigest()
    assert event.output == "43\n"
    assert event.status == "completed"
    assert event.executor_identity == "subprocess-test-double"
    assert event.parent_call_id == "trial-1:call:1"
    assert event.continuation_call_id == "trial-1:call:2"
    assert event.duration_ms >= 0


def test_rejects_invalid_action_missing_continuation_timeout_and_malformed_final() -> None:
    recorder = MethodCallRecorder(_FakeClient(["not json"]), trial_context={"trial_id": "trial-1"})
    with pytest.raises(ToolProtocolError, match="MALFORMED_ACTION"):
        run_tool_loop(_initial_call(recorder), recorder, SubprocessTestDouble(), _policy())

    recorder = MethodCallRecorder(
        _FakeClient([json.dumps({"action": "execute_shell", "command": "true"})]),
        trial_context={"trial_id": "trial-1"},
    )
    with pytest.raises(ToolProtocolError, match="UNKNOWN_ACTION"):
        run_tool_loop(_initial_call(recorder), recorder, SubprocessTestDouble(), _policy())

    recorder = MethodCallRecorder(
        _FakeClient([json.dumps({"action": "execute_python", "code": "print(1)\n"})]),
        trial_context={"trial_id": "trial-1"},
    )
    with pytest.raises(ToolProtocolError, match="MISSING_CONTINUATION"):
        run_tool_loop(_initial_call(recorder), recorder, SubprocessTestDouble(), _policy())

    recorder = MethodCallRecorder(
        _FakeClient(
            [
                json.dumps(
                    {
                        "action": "execute_python",
                        "code": "while True:\n    pass\n",
                        "timeout_seconds": 0.01,
                    }
                )
            ]
        ),
        trial_context={"trial_id": "trial-1"},
    )
    with pytest.raises(ToolTimeoutError, match="TOOL_TIMEOUT"):
        run_tool_loop(_initial_call(recorder), recorder, SubprocessTestDouble(), _policy())

    recorder = MethodCallRecorder(
        _FakeClient([json.dumps({"action": "final", "answer": 43})]),
        trial_context={"trial_id": "trial-1"},
    )
    with pytest.raises(ToolProtocolError, match="MALFORMED_FINAL"):
        run_tool_loop(_initial_call(recorder), recorder, SubprocessTestDouble(), _policy())

    recorder = MethodCallRecorder(
        _FakeClient([json.dumps({"action": "final", "answer": ""})]),
        trial_context={"trial_id": "trial-1"},
    )
    with pytest.raises(ToolProtocolError, match="MALFORMED_FINAL"):
        run_tool_loop(_initial_call(recorder), recorder, SubprocessTestDouble(), _policy())


def test_enforces_maximum_rounds_without_judging_successful_output() -> None:
    client = _FakeClient(
        [
            json.dumps({"action": "execute_python", "code": "print(0)\n"}),
            json.dumps({"action": "execute_python", "code": "print(1)\n"}),
        ]
    )
    recorder = MethodCallRecorder(client, trial_context={"trial_id": "trial-1"})

    with pytest.raises(ToolProtocolError, match="MAX_TOOL_ROUNDS_EXCEEDED"):
        run_tool_loop(_initial_call(recorder, max_rounds=1), recorder, SubprocessTestDouble(), _policy())


def test_preserves_syntactically_successful_wrong_execution_for_the_final_call() -> None:
    client = _FakeClient(
        [
            json.dumps({"action": "execute_python", "code": "print(0)\n"}),
            json.dumps({"action": "final", "answer": "final: 0"}),
        ]
    )
    recorder = MethodCallRecorder(client, trial_context={"trial_id": "trial-1"})

    result = run_tool_loop(_initial_call(recorder), recorder, SubprocessTestDouble(), _policy())

    assert result.answer == "final: 0"
    assert result.tool_events[0].output == "0\n"
    assert result.tool_events[0].exit_code == 0
