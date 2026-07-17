from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memcontam.logging.schema import (
    LOGGING_V1,
    CallEvent,
    FailureEvent,
    FilterEvent,
    MemoryEvent,
    MethodCall,
    RunMetadata,
    TrialLog,
    VerifierResult,
)
from memcontam.logging.writer import RunLogWriter


def _metadata() -> RunMetadata:
    return RunMetadata(
        run_metadata_id="run-meta-1",
        run_id="run-1",
        git_commit="abc123",
        config_hash="sha256:config",
        provider="replay",
        model_snapshots={"replay-model": "fixture-v1"},
        query_date="2026-07-16",
        start_date="2026-07-16",
        seed=7,
        order="task-sample-baseline",
        decoding_defaults={"temperature": 0.0},
        sample_set_hash="sha256:samples",
        sample_order_hash="sha256:order",
        stage="replay",
        schema_version=LOGGING_V1,
        prompt_version="prompt-v1",
        memory_policy_version="memory-v1",
        contamination_catalog_version="catalog-v1",
        retry_policy_version="retry-v1",
    )


def _context() -> dict[str, Any]:
    return {
        "run_metadata_id": "run-meta-1",
        "run_id": "run-1",
        "trial_id": "trial-1",
        "trial_seq": 0,
        "event_seq": 0,
        "stage": "replay",
    }


def _call() -> CallEvent:
    return CallEvent(
        call_id="ignored",
        **_context(),
        messages=[{"role": "user", "content": "solve"}],
        model="replay-model",
        decoding_params={"temperature": 0.0},
        response_text="final: 24",
        token_usage={"total_tokens": 3},
        latency_ms=7,
        retry_count=0,
        source_spans=[],
        created_at="2026-07-16T00:00:00Z",
    )


def _filter() -> FilterEvent:
    return FilterEvent(
        filter_id="ignored",
        **_context(),
        arm="clean",
        baseline="no_memory",
        decisions=[],
        kept_source_ids=[],
        removed_source_ids=[],
        pre_source_ids=[],
        post_source_ids=[],
        ground_truth_contaminated_ids=[],
        action="apply",
        final_answer_source_ids=[],
        verdict=None,
        created_at="2026-07-16T00:00:00Z",
    )


def _failure() -> FailureEvent:
    return FailureEvent(
        failure_id="ignored",
        **_context(),
        origin="verifier",
        error_type="ValueError",
        failure_function="verify",
        failure_module="memcontam.tasks.game24",
        failure_line=12,
        retry_count=0,
        disposition="continued",
        created_at="2026-07-16T00:00:00Z",
    )


def _memory() -> MemoryEvent:
    return MemoryEvent(
        memory_id="ignored",
        **_context(),
        event_type="write",
        operation="append",
        baseline="full_history",
        source_trial_id="trial-1",
        parent_entry_ids=[],
        source_entry_ids=[],
        contaminated_source_ids=[],
        before_entry_ids=[],
        after_entry_ids=["entry-1"],
        before_snapshot_hash="sha256:before",
        after_snapshot_hash="sha256:after",
        new_entry_ids=["entry-1"],
        updated_entry_ids=[],
        removed_entry_ids=[],
        creation_origin="full_history_append",
        memory_version="v1",
        status="accepted",
        created_at="2026-07-16T00:00:00Z",
    )


def _trial(call_id: str) -> TrialLog:
    messages = [{"role": "user", "content": "solve"}]
    return TrialLog(
        trial_id="trial-1",
        run_id="run-1",
        task_name="game24",
        sample_id="sample-1",
        baseline="no_memory",
        arm="clean",
        backbone="replay-model",
        input={"numbers": [1, 3, 4, 6]},
        gold_or_verifier_spec={"target": 24},
        prompt_messages=messages,
        raw_response="final: 24",
        parsed_answer="24",
        verifier_result=VerifierResult(is_correct=True, parsed_answer="24"),
        method_calls=[
            MethodCall(
                call_id=call_id,
                stage="no_memory_generate",
                messages=messages,
                raw_response="final: 24",
                model="replay-model",
            )
        ],
        schema_version=LOGGING_V1,
        stage="replay",
        status="succeeded",
        run_metadata_id="run-meta-1",
        trial_seq=0,
        event_seq=0,
        answer_call_id=call_id,
    )


def test_writer_finalizes_complete_ordered_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.parent.mkdir()
    temp_root = tmp_path / "temporary"
    temp_root.mkdir()

    writer = RunLogWriter(run_dir, _metadata(), tmp_dir=temp_root)

    assert not run_dir.exists()
    assert {path.name for path in writer.temp_dir.iterdir()} == {
        "run.json",
        "trials.jsonl",
        "calls.jsonl",
        "failures.jsonl",
        "filter_events.jsonl",
        "memory_events.jsonl",
    }
    assert RunLogWriter.read_manifest(writer.temp_dir)["status"] == "running"

    filter_event = writer.write_filter(_filter())
    call = writer.write_call(_call())
    failure_event = writer.write_failure(_failure())
    memory_event = writer.write_memory(_memory())
    trial = writer.write_trial(_trial(call.call_id))
    writer.finalize()

    assert not writer.temp_dir.exists()
    assert run_dir.exists()
    assert [filter_event.filter_id, call.call_id, failure_event.failure_id, memory_event.memory_id] == [
        "trial-1:filter:1",
        "trial-1:call:1",
        "trial-1:failure:1",
        "trial-1:memory:1",
    ]
    assert [filter_event.event_seq, call.event_seq, failure_event.event_seq, memory_event.event_seq, trial.event_seq] == [
        1,
        2,
        3,
        4,
        5,
    ]
    manifest = RunLogWriter.read_manifest(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["counts"] == {
        "calls": 1,
        "failures": 1,
        "filter_events": 1,
        "memory_events": 1,
        "trials": 1,
    }
    rows = [
        *RunLogWriter.read_jsonl(run_dir, "filter_events.jsonl"),
        *RunLogWriter.read_jsonl(run_dir, "calls.jsonl"),
        *RunLogWriter.read_jsonl(run_dir, "failures.jsonl"),
        *RunLogWriter.read_jsonl(run_dir, "memory_events.jsonl"),
        *RunLogWriter.read_jsonl(run_dir, "trials.jsonl"),
    ]
    assert [row["event_seq"] for row in rows] == [1, 2, 3, 4, 5]


def test_writer_rejects_existing_final_path_without_writes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    sentinel = run_dir / "existing.txt"
    sentinel.write_text("do not change", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        RunLogWriter(run_dir, _metadata(), tmp_dir=tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "do not change"
    assert not list(tmp_path.glob("run-1.tmp-*"))


def test_writer_preserves_precomputed_recorder_call_id(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run-1", _metadata())
    call = _call().model_copy(update={"call_id": "trial-1:call:9"})

    written = writer.write_call(call)

    assert written.call_id == "trial-1:call:9"
    writer.finalize(status="failed")


def test_writer_marks_temp_manifest_failed_after_write_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    writer = RunLogWriter(run_dir, _metadata())
    writer._streams["calls.jsonl"].close()

    with pytest.raises(ValueError, match="closed file"):
        writer.write_call(_call())

    assert not run_dir.exists()
    assert writer.temp_dir.exists()
    assert RunLogWriter.read_manifest(writer.temp_dir)["status"] == "failed"


def test_writer_rejects_untyped_event_writes(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run-1", _metadata())

    with pytest.raises(TypeError, match="CallEvent"):
        writer.write_call({})  # type: ignore[arg-type]

    writer.finalize(status="failed")
