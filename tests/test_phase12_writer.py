from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from memcontam.logging.schema_v3 import (
    AdmissionEvent,
    CheckpointEvent,
    ContextEvent,
    EligibilityEvent,
    FailureEventV3,
    InterventionEvent,
    RetrievalEvent,
    ToolEvent,
    parse_log_record_v3,
)


writer_v3 = importlib.import_module("memcontam.logging.writer_v3")
Phase12RunWriter = writer_v3.Phase12RunWriter
Phase12WriteError = writer_v3.Phase12WriteError


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-SCHEMA-001.json"
PUBLIC_STREAMS = (
    "trials.jsonl",
    "calls.jsonl",
    "tool_events.jsonl",
    "retrieval_events.jsonl",
    "context_events.jsonl",
    "failures.jsonl",
    "memory_events.jsonl",
    "admission_events.jsonl",
    "intervention_events.jsonl",
    "checkpoint_events.jsonl",
    "eligibility_events.jsonl",
)


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _metadata() -> Any:
    return parse_log_record_v3(_fixture()["valid_run_metadata"][0])


def _trial() -> Any:
    return parse_log_record_v3(_fixture()["valid_trials"][0])


def _events(run_id: str, trial_id: str) -> list[Any]:
    common = {"run_id": run_id, "trial_id": trial_id, "event_seq": 0}
    return [
        ToolEvent(
            record_type="tool_event",
            event_id="tool-1",
            tool_mode="python_sandbox",
            action="execute_python",
            code_hash="sha256:code",
            output="output",
            stderr="",
            exit_code=0,
            status="completed",
            duration_ms=0,
            executor_identity="test-executor",
            parent_call_id="call-1",
            continuation_call_id="call-2",
            **common,
        ),
        RetrievalEvent(
            record_type="retrieval_event",
            event_id="retrieval-1",
            retrieval_id="retrieval-1",
            query_hash="sha256:query",
            retrieved_entry_ids=["entry-1"],
            **common,
        ),
        ContextEvent(
            record_type="context_event",
            event_id="context-1",
            context_id="context-1",
            final_entry_ids=["entry-1"],
            **common,
        ),
        AdmissionEvent(
            record_type="admission_event",
            event_id="admission-1",
            admission_id="admission-1",
            decision="admit",
            **common,
        ),
        InterventionEvent(
            record_type="intervention_event",
            event_id="intervention-1",
            intervention_id="intervention-1",
            arm="contam",
            candidate_triplet_id="triplet-1",
            native_render_id="render-1",
            **common,
        ),
        CheckpointEvent(
            record_type="checkpoint_event",
            event_id="checkpoint-1",
            checkpoint_id="checkpoint-1",
            checkpoint_index=0,
            memory_hash="sha256:memory",
            **common,
        ),
        EligibilityEvent(
            record_type="eligibility_event",
            event_id="eligibility-1",
            eligibility_id="eligibility-1",
            eligible=True,
            **common,
        ),
        FailureEventV3(
            record_type="failure_event", event_id="failure-1", failure_class="provider", **common
        ),
    ]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_writes_and_reopens_complete_v3_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-v3"
    writer = Phase12RunWriter.open(run_dir, _metadata())
    writer.append_trial("trial-1", _trial())
    for event in _events(run_dir.name, "trial-1"):
        writer.append_event(event)
    writer.append_audit_label({"event_id": "admission-1", "audit_label": "private"})

    manifest = writer.finalize()

    assert manifest.status == "completed"
    assert set(manifest.artifacts) == {"run.json", *PUBLIC_STREAMS}
    assert Phase12RunWriter.read_manifest(run_dir)["status"] == "completed"
    assert _jsonl(run_dir / "audit" / "audit_labels.jsonl") == [
        {"audit_label": "private", "event_id": "admission-1"}
    ]
    assert not (run_dir.parent / f"{run_dir.name}.tmp-").exists()

    event_rows = [
        row
        for filename in PUBLIC_STREAMS
        if filename != "trials.jsonl"
        for row in _jsonl(run_dir / filename)
    ]
    assert sorted(row["event_seq"] for row in event_rows) == list(range(1, len(event_rows) + 1))
    assert {row["trial_id"] for row in event_rows} == {"trial-1"}
    for filename, artifact in manifest.artifacts.items():
        payload = (run_dir / filename).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
        assert artifact["count"] == (1 if filename == "run.json" else len(_jsonl(run_dir / filename)))
        if filename.endswith(".jsonl"):
            assert all(
                line == json.dumps(json.loads(line), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                for line in (run_dir / filename).read_text(encoding="utf-8").splitlines()
            )
    public_payload = "".join(
        (run_dir / filename).read_text(encoding="utf-8") for filename in (*PUBLIC_STREAMS, "run.json")
    )
    assert "audit_label" not in public_payload
    assert "private" not in public_payload


def test_rejects_invalid_ids_audit_leakage_and_nonatomic_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = Phase12RunWriter.open(tmp_path / "invalid", _metadata())
    writer.append_trial("trial-1", _trial())
    tool_event = _events("invalid", "trial-1")[0]
    writer.append_event(tool_event)
    with pytest.raises(Phase12WriteError, match="DUPLICATE_EVENT_ID"):
        writer.append_event(tool_event)
    with pytest.raises(Phase12WriteError, match="UNKNOWN_TRIAL_ID"):
        writer.append_event(_events("invalid", "missing")[0].model_copy(update={"event_id": "tool-2"}))
    leaked_admission = _events("invalid", "trial-1")[3].model_copy(
        update={"audit_label": "private"}
    )
    with pytest.raises(Phase12WriteError, match="AUDIT_FIELD_IN_PUBLIC_STREAM"):
        writer.append_event(leaked_admission)

    original_write_line = writer._write_line

    def crash_after_write(filename: str, payload: dict[str, Any]) -> None:
        original_write_line(filename, payload)
        if filename == "retrieval_events.jsonl":
            raise OSError("injected stream crash")

    monkeypatch.setattr(writer, "_write_line", crash_after_write)
    with pytest.raises(OSError, match="injected stream crash"):
        writer.append_event(_events("invalid", "trial-1")[1])
    assert Phase12RunWriter.read_manifest(writer.temp_dir)["status"] == "failed"
    assert not writer.run_dir.exists()

    final_writer = Phase12RunWriter.open(tmp_path / "non-atomic", _metadata())
    final_writer.append_trial("trial-1", _trial())
    original_rename = Path.rename

    def fail_rename(path: Path, target: Path) -> Path:
        if path == final_writer.temp_dir:
            raise OSError("injected rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(Phase12WriteError, match="NONATOMIC_FINALIZE"):
        final_writer.finalize()
    assert Phase12RunWriter.read_manifest(final_writer.temp_dir)["status"] == "failed"
    assert not final_writer.run_dir.exists()
