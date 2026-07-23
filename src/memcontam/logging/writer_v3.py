from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel

from memcontam.logging.schema_v3 import (
    LOGGING_V3,
    AdmissionEvent,
    CheckpointEvent,
    ContextEvent,
    EligibilityEvent,
    FailureEventV3,
    InterventionEvent,
    RetrievalEvent,
    RunMetadataV3,
    ToolEvent,
)


class Phase12WriteError(ValueError):
    pass


@dataclass(frozen=True)
class PublicArtifactManifest:
    status: Literal["completed", "failed"]
    artifacts: dict[str, dict[str, str | int]]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "artifacts": self.artifacts}


class Phase12RunWriter:
    _STREAMS = {
        "trials.jsonl": "trials",
        "calls.jsonl": "calls",
        "tool_events.jsonl": "tool_events",
        "retrieval_events.jsonl": "retrieval_events",
        "context_events.jsonl": "context_events",
        "failures.jsonl": "failures",
        "memory_events.jsonl": "memory_events",
        "admission_events.jsonl": "admission_events",
        "intervention_events.jsonl": "intervention_events",
        "checkpoint_events.jsonl": "checkpoint_events",
        "eligibility_events.jsonl": "eligibility_events",
    }
    _EVENT_STREAMS = {
        ToolEvent: "tool_events.jsonl",
        RetrievalEvent: "retrieval_events.jsonl",
        ContextEvent: "context_events.jsonl",
        AdmissionEvent: "admission_events.jsonl",
        InterventionEvent: "intervention_events.jsonl",
        CheckpointEvent: "checkpoint_events.jsonl",
        EligibilityEvent: "eligibility_events.jsonl",
        FailureEventV3: "failures.jsonl",
    }

    def __init__(self, run_root: Path, metadata: RunMetadataV3, tmp_dir: Path | None = None) -> None:
        self.run_dir = run_root
        if os.path.lexists(self.run_dir):
            raise FileExistsError(f"final run path already exists: {self.run_dir}")
        if not isinstance(metadata, BaseModel) or metadata.schema_version != LOGGING_V3:
            raise TypeError("expected RunMetadataV3")

        self.metadata = metadata.model_copy(deep=True)
        self._state = "running"
        self._event_seq = 0
        self._event_ids: set[str] = set()
        self._trial_ids: set[str] = set()
        self._counts = {count_name: 0 for count_name in self._STREAMS.values()}
        self._streams: dict[str, TextIO] = {}
        self._audit_stream: TextIO | None = None
        temp_root = tmp_dir or self.run_dir.parent
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.run_dir.name}.tmp-", dir=temp_root))
        self._manifest = {
            "run_id": self.run_dir.name,
            "run_metadata": self.metadata.model_dump(mode="json"),
            "status": "running",
            "started_at": _timestamp(),
            "ended_at": None,
            "counts": self._counts,
        }

        try:
            self._write_run_manifest()
            for filename in self._STREAMS:
                self._streams[filename] = (self.temp_dir / filename).open("w", encoding="utf-8")
            audit_dir = self.temp_dir / "audit"
            audit_dir.mkdir()
            self._audit_stream = (audit_dir / "audit_labels.jsonl").open("w", encoding="utf-8")
        except BaseException:
            self._mark_failed()
            raise

    @classmethod
    def open(cls, run_root: Path, metadata: RunMetadataV3) -> Phase12RunWriter:
        return cls(run_root, metadata)

    def append_trial(self, trial_id: str, trial: BaseModel) -> dict[str, Any]:
        self._ensure_running()
        self._validate_trial(trial_id, trial)
        payload = trial.model_dump(mode="json")
        payload["trial_id"] = trial_id
        try:
            self._write_line("trials.jsonl", payload)
            self._trial_ids.add(trial_id)
            self._counts["trials"] += 1
            self._flush_streams()
        except BaseException:
            self._mark_failed()
            raise
        return payload

    def append_call(self, call: object) -> None:
        del call
        raise Phase12WriteError("CALL_RECORD_UNSUPPORTED")

    def append_event(
        self,
        event: ToolEvent
        | RetrievalEvent
        | ContextEvent
        | AdmissionEvent
        | InterventionEvent
        | CheckpointEvent
        | EligibilityEvent
        | FailureEventV3,
    ) -> dict[str, Any]:
        self._ensure_running()
        self._validate_event(event)
        payload = event.model_dump(mode="json")
        payload["event_seq"] = self._next_event_seq()
        filename = self._EVENT_STREAMS[type(event)]
        try:
            self._write_line(filename, payload)
            self._event_ids.add(event.event_id)
            self._counts[self._STREAMS[filename]] += 1
            self._flush_streams()
        except BaseException:
            self._mark_failed()
            raise
        return payload

    def append_audit_label(self, label: dict[str, Any]) -> None:
        self._ensure_running()
        if not isinstance(label, dict):
            raise TypeError("expected audit label mapping")
        try:
            self._write_audit_line(label)
            self._flush_streams()
        except BaseException:
            self._mark_failed()
            raise

    def finalize(self, status: Literal["completed", "failed"] = "completed") -> PublicArtifactManifest:
        if status not in {"completed", "failed"}:
            raise ValueError("final status must be completed or failed")
        if status == "failed":
            if self._state == "running":
                self._mark_failed()
            return self._write_public_artifact_manifest("failed")
        self._ensure_running()
        try:
            self._manifest["status"] = "completed"
            self._manifest["ended_at"] = _timestamp()
            self._write_run_manifest()
            self._fsync_close_streams()
            public_manifest = self._write_public_artifact_manifest("completed")
            if os.path.lexists(self.run_dir):
                raise FileExistsError(f"final run path already exists: {self.run_dir}")
            self.temp_dir.rename(self.run_dir)
            _fsync_directory(self.run_dir.parent)
            self._state = "completed"
            return public_manifest
        except BaseException as error:
            self._mark_failed()
            raise Phase12WriteError("NONATOMIC_FINALIZE") from error

    @classmethod
    def read_manifest(cls, run_dir: Path | str) -> dict[str, Any]:
        with (Path(run_dir) / "run.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def read_jsonl(cls, run_dir: Path | str, filename: str) -> list[dict[str, Any]]:
        with (Path(run_dir) / filename).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _validate_trial(self, trial_id: str, trial: BaseModel) -> None:
        if not trial_id or trial_id in self._trial_ids:
            raise Phase12WriteError("DUPLICATE_EVENT_ID")
        if not isinstance(trial, BaseModel) or getattr(trial, "schema_version", None) != LOGGING_V3:
            raise TypeError("expected TrialLogV3")
        if getattr(trial, "contract_level", None) != "phase12" or not hasattr(trial, "trial_kind"):
            raise TypeError("expected TrialLogV3")
        self._reject_audit_fields(trial.model_dump(mode="json"), trial.__dict__)

    def _validate_event(
        self,
        event: ToolEvent
        | RetrievalEvent
        | ContextEvent
        | AdmissionEvent
        | InterventionEvent
        | CheckpointEvent
        | EligibilityEvent
        | FailureEventV3,
    ) -> None:
        if type(event) not in self._EVENT_STREAMS:
            raise TypeError("expected V3 event")
        if event.event_id in self._event_ids:
            raise Phase12WriteError("DUPLICATE_EVENT_ID")
        if event.run_id != self.run_dir.name:
            raise Phase12WriteError("UNKNOWN_TRIAL_ID")
        if event.trial_id is None or event.trial_id not in self._trial_ids:
            raise Phase12WriteError("UNKNOWN_TRIAL_ID")
        self._reject_audit_fields(event.model_dump(mode="json"), event.__dict__)

    @staticmethod
    def _reject_audit_fields(*values: Any) -> None:
        if any(_contains_audit_field(value) for value in values):
            raise Phase12WriteError("AUDIT_FIELD_IN_PUBLIC_STREAM")

    def _write_line(self, filename: str, payload: dict[str, Any]) -> None:
        self._streams[filename].write(_canonical_json(payload))

    def _write_audit_line(self, label: dict[str, Any]) -> None:
        if self._audit_stream is None:
            raise RuntimeError("audit stream is unavailable")
        self._audit_stream.write(_canonical_json(label))

    def _write_run_manifest(self) -> None:
        manifest_tmp = self.temp_dir / "run.json.tmp"
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            handle.write(_canonical_json(self._manifest))
            handle.flush()
            os.fsync(handle.fileno())
        manifest_tmp.replace(self.temp_dir / "run.json")
        _fsync_directory(self.temp_dir)

    def _write_public_artifact_manifest(
        self, status: Literal["completed", "failed"]
    ) -> PublicArtifactManifest:
        artifacts = {
            filename: {"sha256": _sha256(self.temp_dir / filename), "count": count}
            for filename, count in (("run.json", 1), *self._stream_counts())
        }
        manifest = PublicArtifactManifest(status=status, artifacts=artifacts)
        manifest_tmp = self.temp_dir / "public_artifact_manifest.json.tmp"
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            handle.write(_canonical_json(manifest.to_dict()))
            handle.flush()
            os.fsync(handle.fileno())
        manifest_tmp.replace(self.temp_dir / "public_artifact_manifest.json")
        _fsync_directory(self.temp_dir)
        return manifest

    def _stream_counts(self) -> list[tuple[str, int]]:
        return [(filename, self._counts[count_name]) for filename, count_name in self._STREAMS.items()]

    def _flush_streams(self) -> None:
        for stream in self._streams.values():
            stream.flush()
        if self._audit_stream is not None:
            self._audit_stream.flush()

    def _fsync_close_streams(self) -> None:
        for stream in [*self._streams.values(), self._audit_stream]:
            if stream is not None and not stream.closed:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()

    def _mark_failed(self) -> None:
        if self._state != "running":
            return
        self._state = "failed"
        self._manifest["status"] = "failed"
        self._manifest["ended_at"] = _timestamp()
        try:
            self._write_run_manifest()
        finally:
            for stream in [*self._streams.values(), self._audit_stream]:
                if stream is None or stream.closed:
                    continue
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                except OSError:
                    pass
                finally:
                    stream.close()

    def _ensure_running(self) -> None:
        if self._state != "running":
            raise RuntimeError(f"writer is {self._state}")

    def _next_event_seq(self) -> int:
        self._event_seq += 1
        return self._event_seq


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def _contains_audit_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any("audit" in str(key).lower() or _contains_audit_field(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_audit_field(item) for item in value)
    return False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
