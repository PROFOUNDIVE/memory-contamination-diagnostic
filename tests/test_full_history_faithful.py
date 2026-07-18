from __future__ import annotations

import json
from typing import Callable, cast

import pytest

from memcontam.baselines.full_history import FullHistoryPolicy, _call_verifier
from memcontam.clients.replay import ReplayClient
from memcontam.logging.provenance import compute_exposure_from_spans, normalize_memory_event
from memcontam.logging.schema import MemoryItemLog, PromptSourceSpan, VerifierResult
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
    call = result["method_calls"][0]
    assert call.call_id is not None
    assert result["answer_call_id"] == call.call_id
    assert call.messages == [
        {
            "role": "user",
            "content": "History:\nPrevious input: <task prompt>\nPrevious response: seed response\n\nSolve: {'question': '2 + 6'}",
        }
    ]
    assert len(call.source_spans) == 1
    span = call.source_spans[0]
    assert isinstance(span, PromptSourceSpan)
    assert span.message_index == 0
    assert span.entry_id == "seed-1"
    assert span.clean_or_contaminated == "clean"
    assert span.origin == "memory_seed"
    assert call.messages[0]["content"][span.start:span.end] == "seed response"
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
    assert result["retrieved_records"] == []
    assert result["retrieved_scores"] == []
    assert new_entry["clean_or_contaminated"] == "contaminated"
    assert new_entry["metadata"]["lineage"] == "contaminated"
    assert new_entry["metadata"]["source_entry_ids"] == ["cont-1"]
    assert result["memory_write_event"]["source_entry_ids"] == ["cont-1"]


def test_build_messages_returns_exact_text_and_source_spans() -> None:
    task = TaskInstance(sample_id="s-6", task_name="full_history", input={"question": "2 + 2"})
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="seed-6",
                content="seed response",
                memory_type="memory_seed",
                clean_or_contaminated="clean",
            )
        ]
    )

    messages, spans = FullHistoryPolicy()._build_messages(task, memory)

    assert messages == [
        {
            "role": "user",
            "content": "History:\nPrevious input: <task prompt>\nPrevious response: seed response\n\nSolve: {'question': '2 + 2'}",
        }
    ]
    assert len(spans) == 1
    assert messages[0]["content"][spans[0].start:spans[0].end] == "seed response"


def test_run_records_contaminated_source_span_as_final_prompt() -> None:
    task = TaskInstance(sample_id="s-5", task_name="full_history", input={"question": "1 + 1"})
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="cont-1",
                content="bad seed",
                memory_type="memory_seed",
                clean_or_contaminated="contaminated",
                source_trial_id="corpus:cont-1",
                metadata={"lineage_id": "cont-lineage-1"},
            )
        ]
    )
    client = ReplayClient(responses_by_sample={"s-5": {"full_history_generate": "final: 2"}})

    result = FullHistoryPolicy().run(
        task,
        memory,
        client=client,
        model="gpt-4o",
        config={
            "sample_id": "s-5",
            "run_id": "run-5",
            "baseline": "full_history",
            "arm": "contaminated",
            "model": "gpt-4o",
        },
    )

    call = result["method_calls"][0]
    assert len(call.source_spans) == 1
    span = call.source_spans[0]
    assert span.entry_id == "cont-1"
    assert span.clean_or_contaminated == "contaminated"
    assert span.lineage_id == "cont-lineage-1"
    exposure = compute_exposure_from_spans(
        result["answer_call_id"], call.source_spans, "contaminated"
    )
    assert exposure.status == "supported"
    assert exposure.is_exposed is True
    assert exposure.exposure_mode == "final_prompt"
    assert "cont-1" in exposure.exposed_source_ids


def test_call_verifier_requires_task_argument() -> None:
    task = TaskInstance(sample_id="s-3", task_name="full_history", input={})

    def one_argument_verifier(answer: str) -> VerifierResult:
        return VerifierResult(is_correct=True, parsed_answer=answer)

    with pytest.raises(TypeError):
        _call_verifier(
            cast(Callable[[str, TaskInstance], VerifierResult], one_argument_verifier), "answer", task
        )


def test_accepted_memory_event_normalizes_append_mutation() -> None:
    task = TaskInstance(sample_id="s-4", task_name="full_history", input={"question": "3 + 5"})
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="seed-4",
                content="seed response",
                memory_type="memory_seed",
                clean_or_contaminated="clean",
                source_trial_id="corpus:seed-4",
            )
        ]
    )
    client = ReplayClient(responses_by_sample={"s-4": {"full_history_generate": "final: 8"}})

    result = FullHistoryPolicy().run(
        task,
        memory,
        client=client,
        model="gpt-4o",
        config={
            "sample_id": "s-4",
            "run_id": "run-4",
            "baseline": "full_history",
            "arm": "clean",
            "model": "gpt-4o",
        },
    )

    source_trial_id = "run-4:full_history:s-4:full_history:clean:gpt-4o"
    before = [MemoryEntry.model_validate(entry) for entry in result["memory_before"]]
    after = [MemoryEntry.model_validate(entry) for entry in result["memory_after"]]
    event = normalize_memory_event(
        "full_history",
        source_trial_id,
        before,
        after,
        result["memory_write_event"],
    )

    assert event is not None
    assert event.event_type == "memory_write"
    assert event.operation == "append"
    assert event.baseline == "full_history"
    assert event.source_trial_id == source_trial_id
    assert event.status == "accepted"
    assert event.before_entry_ids == ["seed-4"]
    assert event.after_entry_ids == ["seed-4", result["memory_write_event"]["new_entry_id"]]
    assert event.new_entry_ids == [result["memory_write_event"]["new_entry_id"]]
    assert event.updated_entry_ids == []
    assert event.removed_entry_ids == []
    assert event.before_snapshot_hash != event.after_snapshot_hash
    assert event.parent_entry_ids == ["seed-4"]
    assert event.source_entry_ids == []
    assert event.contaminated_source_ids == []
    assert event.creation_origin == "full_history_transcript"
    assert event.memory_version == "v0"


def test_normalized_full_history_failure_is_natural_with_clean_ancestry() -> None:
    before = [
        MemoryEntry(
            entry_id="clean-seed",
            content="A clean hint.",
            memory_type="strategy",
            metadata={
                "contamination_class": "clean",
                "lineage_status": "exact",
                "lineage_basis": "seed",
            },
        )
    ]
    after = [
        *before,
        MemoryEntry(
            entry_id="failed-transcript",
            content="A failed response transcript.",
            memory_type="full_history_transcript",
            source_trial_id="trial-1",
            metadata={
                "parent_entry_ids": ["clean-seed"],
                "direct_parent_ids": ["clean-seed"],
                "memory_error_status": "satisfied",
            },
        ),
    ]

    event = normalize_memory_event(
        "full_history",
        "trial-1",
        before,
        after,
        {"type": "full_history_append", "status": "accepted", "new_entry_id": "failed-transcript"},
    )

    assert event is not None
    assert event.lineage_edges[0].child_entry_id == "failed-transcript"
    assert event.lineage_edges[0].parent_entry_id == "clean-seed"
    item = MemoryItemLog.from_memory_entry(after[-1], after)
    assert item.contamination_class == "natural"
    assert item.injected_root_ids == []
