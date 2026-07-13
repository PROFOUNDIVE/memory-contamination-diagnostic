from __future__ import annotations

import json

from memcontam.baselines.full_history import FullHistoryPolicy
from memcontam.clients.replay import ReplayClient
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


def test_build_prompt_keeps_legacy_history_format() -> None:
    task = TaskInstance(sample_id="s-1", task_name="full_history", input={"question": "2 + 6"})
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="seed-1",
                content="seed response",
                memory_type="memory_seed",
                clean_or_contaminated="clean",
            )
        ]
    )

    prompt = FullHistoryPolicy().build_prompt(task, memory)

    assert prompt == [
        {"role": "user", "content": "History:\nseed response\n\nSolve: {'question': '2 + 6'}"}
    ]


def test_run_returns_faithful_append_only_result() -> None:
    task = TaskInstance(sample_id="s-1", task_name="full_history", input={"question": "2 + 6"})
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="seed-1",
                content="seed response",
                memory_type="memory_seed",
                clean_or_contaminated="clean",
                source_trial_id="corpus:seed-1",
            )
        ]
    )
    client = ReplayClient(responses_by_sample={"s-1": {"full_history_generate": "final: 8"}})
    calls: list[tuple[str, str]] = []

    def verifier(parsed_answer: str, seen_task: TaskInstance) -> VerifierResult:
        calls.append((parsed_answer, seen_task.sample_id))
        return VerifierResult(is_correct=True, parsed_answer=parsed_answer, reason="ok")

    result = FullHistoryPolicy().run(
        task,
        memory,
        client=client,
        model="gpt-4o",
        config={
            "sample_id": "s-1",
            "run_id": "run-1",
            "baseline": "full_history",
            "arm": "clean",
            "model": "gpt-4o",
        },
        verifier=verifier,
    )

    assert calls == [("8", "s-1")]
    assert result["final_response"] == "final: 8"
    assert result["parsed_answer"] == "8"
    assert result["verifier_result"].is_correct is True
    assert result["retrieved_records"] == []
    assert result["retrieved_scores"] == []
    assert [call.stage for call in result["method_calls"]] == ["full_history_generate"]
    assert result["method_calls"][0].messages == [
        {
            "role": "user",
            "content": "History:\nPrevious input: <task prompt>\nPrevious response: seed response\n\nSolve: {'question': '2 + 6'}",
        }
    ]
    assert result["memory_before"] == [memory.entries[0].model_dump()]
    assert len(result["memory_after"]) == 2
    assert result["memory_after"][0] == memory.entries[0].model_dump()

    new_entry = result["memory_after"][1]
    assert new_entry["entry_id"].startswith("full_history:full_history:s-1:")
    assert new_entry["memory_type"] == "full_history_transcript"
    assert new_entry["clean_or_contaminated"] == "clean"
    assert new_entry["source_trial_id"] == "run-1:full_history:s-1:full_history:clean:gpt-4o"
    assert new_entry["metadata"]["parent_entry_ids"] == ["seed-1"]
    assert new_entry["metadata"]["source_entry_ids"] == []
    assert new_entry["metadata"]["lineage"] == "clean"
    assert new_entry["content"] == (
        "Previous input: {'question': '2 + 6'}\n"
        f"Previous prompt: {json.dumps(result['method_calls'][0].messages, ensure_ascii=False)}\n"
        "Previous response: final: 8\n"
        "Parsed answer: 8\n"
        "Correct: true"
    )
    assert result["memory_write_event"] == {
        "type": "full_history_append",
        "status": "accepted",
        "new_entry_id": new_entry["entry_id"],
        "source_trial_id": new_entry["source_trial_id"],
        "parent_entry_ids": ["seed-1"],
        "source_entry_ids": [],
    }


def test_run_marks_contaminated_lineage_from_any_parent() -> None:
    task = TaskInstance(sample_id="s-2", task_name="full_history", input={"question": "1 + 1"})
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="seed-1",
                content="seed response",
                memory_type="memory_seed",
                clean_or_contaminated="clean",
            ),
            MemoryEntry(
                entry_id="cont-1",
                content="bad seed",
                memory_type="memory_seed",
                clean_or_contaminated="contaminated",
            ),
        ]
    )
    client = ReplayClient(responses_by_sample={"s-2": {"full_history_generate": "final: 2"}})

    result = FullHistoryPolicy().run(
        task,
        memory,
        client=client,
        model="gpt-4o",
        config={
            "sample_id": "s-2",
            "run_id": "run-2",
            "baseline": "full_history",
            "arm": "contaminated",
            "model": "gpt-4o",
        },
    )

    new_entry = result["memory_after"][-1]
    assert new_entry["clean_or_contaminated"] == "contaminated"
    assert new_entry["metadata"]["lineage"] == "contaminated"
    assert new_entry["metadata"]["source_entry_ids"] == ["cont-1"]
    assert result["memory_write_event"]["source_entry_ids"] == ["cont-1"]
