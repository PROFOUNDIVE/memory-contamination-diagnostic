from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from memcontam.baselines import full_history
from memcontam.baselines.full_history import FullHistoryPayload, FullHistoryState, render_full_history
from memcontam.clients.base import LLMResponse
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


FullHistoryAdapter = import_module("memcontam.baselines.full_history_adapter").FullHistoryAdapter


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    )


def _config() -> dict[str, str]:
    return {
        "run_id": "run-1",
        "baseline": "full_history",
        "arm": "clean",
        "model": "replay",
    }


class _Client:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], str, dict]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        self.calls.append((messages, model, config))
        return LLMResponse(
            content=self.response,
            raw={"replay": True},
            token_usage={},
            latency_ms=0,
        )


def _record(entry_id: str, task_input: str, raw_response: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=render_full_history(entry_id, FullHistoryPayload(task_input, raw_response)),
        memory_type="full_history_transcript",
        clean_or_contaminated="clean",
    )


def test_full_history_contract_exposes_one_native_adapter() -> None:
    adapter = FullHistoryAdapter()

    assert full_history.FullHistoryAdapter is FullHistoryAdapter
    assert callable(adapter.execute)
    assert not hasattr(adapter, "run")
    assert not hasattr(adapter, "build_prompt")


def test_render_full_history_matches_the_committed_raw_record_fixture() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "prompts" / "full_history_generate.json").read_text(
            encoding="utf-8"
        )
    )
    payload = FullHistoryPayload(task_input="{'numbers': [1, 3, 4, 6]}", raw_response="final: 24")

    assert render_full_history("history-1", payload) == fixture["history_record"].replace(
        "{{entry_id}}", "history-1"
    ).replace("{{task_input}}", payload.task_input).replace("{{raw_response}}", payload.raw_response)


def test_full_history_renders_raw_records_in_order_and_keeps_valid_incorrect_successful() -> None:
    first = _record("history-1", "first task", "first response")
    second = _record("history-2", "second task", "second response")
    state = FullHistoryState(records=[first, second])
    client = _Client("final: 24")

    outcome = FullHistoryAdapter().execute(
        _task(),
        state,
        client=client,
        model="replay",
        config=_config(),
        verifier=lambda answer, task: VerifierResult(
            is_correct=False,
            parsed_answer=answer,
            reason="must not enter history",
        ),
    )

    assert outcome.status == "succeeded"
    assert outcome.parsed_answer == "24"
    assert outcome.verifier_result is False
    assert outcome.error_type is None
    assert len(client.calls) == 1
    assert client.calls[0][2]["method_stage"] == "full_history_generate"
    prompt = client.calls[0][0][0]["content"]
    assert prompt == f"{first.content}\n\n{second.content}\n\nTASK:\n{_task().input}"
    assert "must not enter history" not in prompt
    assert "parsed_answer" not in prompt
    assert "parent_entry_ids" not in prompt
    assert [
        prompt[span.start : span.end] for span in outcome.method_calls[0].source_spans
    ] == [first.content, second.content]
    assert len(state.records) == 3
    appended = state.records[-1]
    assert appended.content == render_full_history(
        appended.entry_id,
        FullHistoryPayload(str(_task().input), "final: 24"),
    )
    assert "parent_entry_ids" not in appended.metadata
    assert "direct_parent_ids" not in appended.metadata
    assert appended.metadata["source_entry_ids"] == ["history-1", "history-2"]
    assert outcome.memory_write_event == {
        "type": "full_history_append",
        "status": "accepted",
        "new_entry_id": appended.entry_id,
        "source_trial_id": "run-1:game24:sample-1:full_history:clean:replay",
        "source_entry_ids": ["history-1", "history-2"],
    }


def test_full_history_empty_response_is_appended_before_the_failed_parse_outcome() -> None:
    state = FullHistoryState()

    outcome = FullHistoryAdapter().execute(
        _task(), state, client=_Client(""), model="replay", config=_config()
    )

    assert outcome.status == "failed"
    assert outcome.final_response == ""
    assert outcome.parsed_answer is None
    assert outcome.error_type == "BaselineOutputError"
    assert outcome.failure_disposition == "full_history_invalid_final_answer"
    assert outcome.scientific_ineligibility_reason == "invalid_final_answer"
    assert len(state.records) == 1
    assert "RESPONSE:\n\n<END_HISTORY_RECORD>" in state.records[0].content
    assert outcome.memory_after[-1]["entry_id"] == state.records[0].entry_id
    assert outcome.memory_write_event is not None
    assert outcome.memory_write_event["status"] == "accepted"


def test_full_history_parse_failure_never_rolls_back_the_completed_response() -> None:
    state = FullHistoryState()

    outcome = FullHistoryAdapter().execute(
        _task(), state, client=_Client("final:   "), model="replay", config=_config()
    )

    assert outcome.status == "failed"
    assert state.records[-1].content.endswith("RESPONSE:\nfinal:   \n<END_HISTORY_RECORD>")


def test_full_history_appends_before_verifier_contract_failure() -> None:
    state = FullHistoryState()

    def broken_verifier(answer: str, task: TaskInstance) -> VerifierResult:
        raise RuntimeError("verifier unavailable")

    outcome = FullHistoryAdapter().execute(
        _task(),
        state,
        client=_Client("final: 24"),
        model="replay",
        config=_config(),
        verifier=broken_verifier,
    )

    assert outcome.status == "failed"
    assert outcome.error_type == "VerifierContractError"
    assert len(state.records) == 1
