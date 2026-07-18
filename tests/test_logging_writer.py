from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memcontam.logging.schema import (
    LOGGING_V1,
    LOGGING_V2,
    CallEvent,
    FailureEvent,
    FilterEvent,
    MemoryEvent,
    MethodCall,
    PromptSourceSpan,
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


def _phase11_metadata() -> RunMetadata:
    payload = _metadata().model_dump()
    payload.update(
        {
            "schema_version": LOGGING_V2,
            "contract_level": "phase11",
            "evaluation_law": {
                "evaluation_law_id": "law-v2",
                "regime": "online",
                "task_law_id": "tasks-v1",
                "inference_law_id": "replay-v1",
            },
            "target_contamination_set": {
                "target_set_id": "target-v2",
                "definition_version": "phase11-v1",
                "included_classes": ["injected", "derived"],
                "require_exact_lineage": True,
            },
        }
    )
    return RunMetadata.model_validate(payload)


def _phase11_trial(call_id: str) -> TrialLog:
    payload = _trial(call_id).model_dump()
    payload.update(
        {
            "schema_version": LOGGING_V2,
            "evaluation_law_id": "law-v2",
            "target_set_id": "target-v2",
            "memory_update_mode": "not_applicable",
            "trajectory_pair_id": "trajectory-1",
            "checkpoint_index": 0,
            "pair_id": "trajectory-1:0:sample-1",
            "contamination_exposure": {
                "condition": "clean",
                "status": "not_applicable",
                "is_exposed": None,
                "answer_call_id": None,
                "target_entry_ids": [],
                "source_entry_ids": [],
                "exposed_source_ids": [],
                "exposure_mode": "clean",
                "reason": "clean arm",
                "target_set_id": "target-v2",
                "exposed_entry_ids": [],
                "exposed_injected_root_ids": [],
                "evidence_lineage_status": None,
            },
        }
    )
    return TrialLog.model_validate(payload)


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evaluation_law_id", "other-law", "evaluation_law_id"),
        ("target_set_id", "other-target", "target_set_id"),
        ("pair_id", "other-pair", "pair_id"),
    ],
)
def test_phase11_writer_rejects_trial_context_outside_manifest_contract(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    writer = RunLogWriter(tmp_path / "run-v2", _phase11_metadata())
    call = writer.write_call(_call())
    trial = _phase11_trial(call.call_id).model_copy(update={field: value})

    with pytest.raises(ValueError, match=message):
        writer.write_trial(trial)

    writer.finalize(status="failed")


def test_phase11_writer_does_not_accept_raw_metadata_as_target_evidence(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run-v2", _phase11_metadata())
    call = writer.write_call(_call())
    payload = _phase11_trial(call.call_id).model_dump()
    payload.update(
        {
            "baseline": "retrieval_rag",
            "arm": "contaminated",
            "memory_update_mode": "enabled",
            "memory_before": [
                {
                    "entry_id": "metadata-only-derived",
                    "content": "Unverified lineage.",
                    "memory_type": "note",
                    "clean_or_contaminated": "contaminated",
                    "metadata": {
                        "contamination_class": "derived",
                        "lineage_status": "exact",
                    },
                }
            ],
            "method_calls": [
                {
                    "call_id": call.call_id,
                    "stage": "rag_generate",
                    "messages": [{"role": "user", "content": "solve"}],
                    "raw_response": "final: 24",
                    "model": "replay-model",
                    "source_spans": [
                        {
                            "message_index": 0,
                            "start": 0,
                            "end": 5,
                            "rendered_hash": "sha256:prompt",
                            "entry_id": "metadata-only-derived",
                            "source_ids": ["metadata-only-derived"],
                            "parent_ids": [],
                            "lineage_id": "metadata-only-derived",
                            "version": "v2",
                            "origin": "test",
                            "clean_or_contaminated": "contaminated",
                            "contamination_class": "derived",
                            "injected_root_ids": ["root-1"],
                            "lineage_status": "exact",
                            "lineage_basis": "recorded_parent",
                            "direct_parent_ids": ["root-1"],
                            "target_set_id": "target-v2",
                            "is_target_contamination": True,
                        }
                    ],
                }
            ],
            "contamination_exposure": {
                "condition": "contaminated",
                "status": "supported",
                "is_exposed": True,
                "answer_call_id": call.call_id,
                "target_entry_ids": ["metadata-only-derived"],
                "source_entry_ids": ["metadata-only-derived"],
                "exposed_source_ids": ["metadata-only-derived"],
                "exposure_mode": "final_prompt",
                "reason": "raw metadata target",
                "target_set_id": "target-v2",
                "exposed_entry_ids": ["metadata-only-derived"],
                "exposed_injected_root_ids": ["root-1"],
                "evidence_lineage_status": "exact",
            },
        }
    )
    trial = TrialLog.model_validate(payload)

    with pytest.raises(ValueError, match="target_entry_ids"):
        writer.write_trial(trial)

    writer.finalize(status="failed")


def test_phase11_clean_natural_target_survives_writer_and_aggregate(tmp_path: Path) -> None:
    metadata_payload = _phase11_metadata().model_dump(mode="json")
    metadata_payload["target_contamination_set"] = {
        "target_set_id": "natural-target-v1",
        "definition_version": "phase11-v1",
        "included_classes": ["natural"],
        "require_exact_lineage": True,
    }
    metadata = RunMetadata.model_validate(metadata_payload)
    span = PromptSourceSpan.model_validate({
        "message_index": 0,
        "start": 0,
        "end": 7,
        "rendered_hash": "sha256:natural",
        "entry_id": "natural-1",
        "source_ids": ["natural-1"],
        "parent_ids": ["clean-seed"],
        "lineage_id": "natural-1",
        "version": "v1",
        "origin": "full_history_append",
        "clean_or_contaminated": "contaminated",
        "contamination_class": "natural",
        "injected_root_ids": [],
        "lineage_status": "exact",
        "lineage_basis": "recorded_parent",
        "direct_parent_ids": ["clean-seed"],
        "target_set_id": "natural-target-v1",
        "is_target_contamination": True,
    })
    writer = RunLogWriter(tmp_path / "run-v2", metadata)
    call = writer.write_call(_call().model_copy(update={"source_spans": [span]}))
    payload = _phase11_trial(call.call_id).model_dump(mode="json")
    payload.update(
        {
            "target_set_id": "natural-target-v1",
            "memory_before": [
                {
                    "entry_id": "clean-seed",
                    "content": "Clean seed.",
                    "memory_type": "strategy",
                    "clean_or_contaminated": "clean",
                    "metadata": {
                        "contamination_class": "clean",
                        "lineage_status": "exact",
                        "lineage_basis": "seed",
                        "direct_parent_ids": [],
                        "injected_root_ids": [],
                    },
                },
                {
                    "entry_id": "natural-1",
                    "content": "Natural error.",
                    "memory_type": "full_history_transcript",
                    "clean_or_contaminated": "contaminated",
                    "source_trial_id": "prior-trial",
                    "metadata": {
                        "direct_parent_ids": ["clean-seed"],
                        "memory_error_status": "satisfied",
                    },
                },
            ],
            "method_calls": [{**payload["method_calls"][0], "source_spans": [span.model_dump()]}],
            "contamination_exposure": {
                "condition": "clean",
                "status": "not_applicable",
                "is_exposed": None,
                "answer_call_id": call.call_id,
                "target_entry_ids": ["natural-1"],
                "source_entry_ids": ["natural-1"],
                "exposed_source_ids": [],
                "exposure_mode": "clean",
                "reason": "clean arm has no controlled target exposure",
                "target_set_id": "natural-target-v1",
                "exposed_entry_ids": [],
                "exposed_injected_root_ids": [],
                "evidence_lineage_status": None,
            },
        }
    )

    writer.write_trial(TrialLog.model_validate(payload))
    writer.finalize()

    from memcontam.evaluation.aggregate import aggregate_run

    assert aggregate_run(tmp_path / "run-v2", stage="replay", contract="phase11")["n_trials"] == 1
