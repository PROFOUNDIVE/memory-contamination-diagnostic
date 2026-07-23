from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema_v3 import RunMetadataV3, parse_log_record_v3
from memcontam.logging.writer_v3 import Phase12RunWriter
from memcontam.tools import SubprocessTestDouble, load_tool_runtime_contract
from memcontam.tools.execution_loop import LlmCall, run_tool_loop


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "containers" / "python-sandbox" / "image.lock.json"
SCHEMA_FIXTURE = ROOT / "tests" / "fixtures" / "phase12" / "FX-SCHEMA-001.json"


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        return LLMResponse(content=self.responses.pop(0), raw={}, token_usage={}, latency_ms=1)


def _fixture() -> dict[str, Any]:
    return json.loads(SCHEMA_FIXTURE.read_text(encoding="utf-8"))


def test_writes_linked_execution_to_tool_events_jsonl(tmp_path: Path) -> None:
    fixture = _fixture()
    run_dir = tmp_path / "tool-run"
    writer = Phase12RunWriter.open(
        run_dir, cast(RunMetadataV3, parse_log_record_v3(fixture["valid_run_metadata"][0]))
    )
    writer.append_trial("trial-1", parse_log_record_v3(fixture["valid_trials"][0]))
    client = _FakeClient(
        [
            json.dumps({"action": "execute_python", "code": "print(43)\n"}),
            json.dumps({"action": "final", "answer": "final: 43"}),
        ]
    )
    recorder = MethodCallRecorder(client, trial_context={"trial_id": "trial-1"})
    messages = [{"role": "user", "content": "solve"}]
    initial_response = recorder.chat(messages, "gpt-replay", {"method_stage": "tool_generate"})
    initial_call_id = recorder.get_records()[-1].call_id
    assert isinstance(initial_call_id, str)
    initial = LlmCall(
        call_id=initial_call_id,
        content=initial_response.content,
        messages=messages,
        model="gpt-replay",
        config={"method_stage": "tool_generate"},
        run_id=run_dir.name,
        trial_id="trial-1",
    )

    result = run_tool_loop(
        initial,
        recorder,
        SubprocessTestDouble(),
        load_tool_runtime_contract(LOCK_PATH, scientific=False),
        writer=writer,
    )
    writer.finalize()

    rows = Phase12RunWriter.read_jsonl(run_dir, "tool_events.jsonl")
    assert result.answer_call_id == "trial-1:call:2"
    assert rows == [
        {
            "action": "execute_python",
            "code_hash": result.tool_events[0].code_hash,
            "continuation_call_id": "trial-1:call:2",
            "contract_level": "phase12",
            "duration_ms": result.tool_events[0].duration_ms,
            "event_id": "trial-1:call:1:tool:1",
            "event_seq": 1,
            "executor_identity": "subprocess-test-double",
            "exit_code": 0,
            "output": "43\n",
            "parent_call_id": "trial-1:call:1",
            "record_type": "tool_event",
            "run_id": "tool-run",
            "schema_version": "logging_v3",
            "status": "completed",
            "stderr": "",
            "tool_mode": "python_sandbox",
            "trial_id": "trial-1",
        }
    ]
