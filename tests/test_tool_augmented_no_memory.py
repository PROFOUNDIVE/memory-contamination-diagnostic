from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from memcontam.clients.replay import ReplayClient
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.game24 import build_instance
from memcontam.tools import SubprocessTestDouble, load_tool_runtime_contract
from memcontam.verifiers.game24 import verify_expression


ROOT = Path(__file__).resolve().parents[1]


def _tool_contract():
    return load_tool_runtime_contract(
        ROOT / "containers" / "python-sandbox" / "image.lock.json", scientific=False
    )


def _verify(answer: str, task):
    return verify_expression(answer, task.input["numbers"], task.verifier_spec["target"])


def test_solves_game24_with_one_python_call_and_no_memory() -> None:
    adapter_module = importlib.import_module("memcontam.baselines.no_memory_tool_adapter")

    task = build_instance({"sample_id": "game24-tool", "numbers": [1, 3, 4, 6], "target": 24})
    executor = SubprocessTestDouble()
    outcome = adapter_module.NoMemoryToolAdapter().execute(
        task,
        MemoryState(),
        client=ReplayClient(
            responses=[
                '{"action":"execute_python","code":"print(6 / (1 - 3 / 4))"}',
                '{"action":"final","answer":"final: 6 / (1 - 3 / 4)"}',
            ]
        ),
        model="replay",
        executor=executor,
        policy=_tool_contract(),
        config={"run_id": "tool-run"},
        verifier=_verify,
    )

    assert outcome.status == "succeeded"
    assert outcome.parsed_answer == "6 / (1 - 3 / 4)"
    assert outcome.verifier_result.is_correct is True
    assert outcome.answer_call_id == "tool-run:game24:game24-tool:no_memory:clean:replay:call:2"
    assert len(outcome.metadata["tool_events"]) == 1
    assert outcome.metadata["tool_events"][0].output == "24.0\n"
    assert executor.execution_count == 1
    assert outcome.memory_before == outcome.memory_after == ()
    assert outcome.memory_write_event is None
    assert outcome.metadata["memory_events"] == ()


def test_rejects_any_memory_state_or_event() -> None:
    adapter_module = importlib.import_module("memcontam.baselines.no_memory_tool_adapter")

    task = build_instance({"sample_id": "game24-tool", "numbers": [1, 3, 4, 6], "target": 24})
    adapter = adapter_module.NoMemoryToolAdapter()
    client = ReplayClient(responses=['{"action":"final","answer":"final: 24"}'])

    with pytest.raises(adapter_module.NoMemoryContractError, match="NOMEM_MEMORY_STATE_FORBIDDEN"):
        adapter.execute(
            task,
            MemoryState(
                entries=[
                    MemoryEntry(entry_id="forbidden", content="memory", memory_type="strategy")
                ]
            ),
            client=client,
            model="replay",
            executor=SubprocessTestDouble(),
            policy=_tool_contract(),
        )

    with pytest.raises(adapter_module.NoMemoryContractError, match="NOMEM_MEMORY_EVENTS_FORBIDDEN"):
        adapter.execute(
            task,
            MemoryState(),
            client=client,
            model="replay",
            executor=SubprocessTestDouble(),
            policy=_tool_contract(),
            memory_events=[{"type": "memory_write"}],
        )


def test_keeps_text_only_no_memory_prompt_unchanged() -> None:
    from memcontam.baselines.no_memory import NoMemoryPolicy

    task = build_instance({"sample_id": "game24-text", "numbers": [1, 3, 4, 6], "target": 24})

    assert NoMemoryPolicy().build_prompt(task, MemoryState())[0]["content"] == (
        "Solve the task. Use no persistent memory. "
        "Return only the final answer in the required task format."
    )
