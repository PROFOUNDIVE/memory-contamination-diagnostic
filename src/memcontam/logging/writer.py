from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO, TypeVar

from memcontam.logging.schema import (
    LOGGING_V2,
    CallEvent,
    FailureEvent,
    FilterEvent,
    MemoryEvent,
    RunMetadata,
    TrialLog,
    _v2_target_entry_ids,
)


EventT = TypeVar("EventT", CallEvent, FailureEvent, FilterEvent, MemoryEvent)


class RunLogWriter:
    """Own the ordered, crash-safe artifacts for one strict run."""

    _STREAMS = {
        "trials.jsonl": "trials",
        "calls.jsonl": "calls",
        "failures.jsonl": "failures",
        "filter_events.jsonl": "filter_events",
        "memory_events.jsonl": "memory_events",
    }

    def __init__(
        self,
        run_dir: Path | str,
        run_metadata: RunMetadata,
        tmp_dir: Path | str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        if os.path.lexists(self.run_dir):
            raise FileExistsError(f"final run path already exists: {self.run_dir}")

        self.run_metadata = run_metadata.model_copy(deep=True)
        self._frozen_checkpoint: dict[str, Any] | None = None
        temp_root = Path(tmp_dir) if tmp_dir is not None else self.run_dir.parent
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.run_dir.name}.tmp-", dir=temp_root))
        self._streams: dict[str, TextIO] = {}
        self._event_seq = 0
        self._event_indices: dict[tuple[str, str], int] = {}
        self._state = "running"
        self._counts = {count_name: 0 for count_name in self._STREAMS.values()}
        self._manifest = {
            "run_metadata": self.run_metadata.model_dump(mode="json"),
            "status": "running",
            "started_at": _timestamp(),
            "ended_at": None,
            "counts": self._counts,
        }

        try:
            self._write_manifest()
            for filename in self._STREAMS:
                self._streams[filename] = (self.temp_dir / filename).open("w", encoding="utf-8")
        except BaseException:
            self._mark_failed()
            raise

    def write_call(self, call_event: CallEvent) -> CallEvent:
        self._require_type(call_event, CallEvent)
        return self._write_event(call_event, "call", "call_id", "calls.jsonl")

    def write_failure(self, failure_event: FailureEvent) -> FailureEvent:
        self._require_type(failure_event, FailureEvent)
        return self._write_event(failure_event, "failure", "failure_id", "failures.jsonl")

    def write_filter(self, filter_event: FilterEvent) -> FilterEvent:
        self._require_type(filter_event, FilterEvent)
        return self._write_event(filter_event, "filter", "filter_id", "filter_events.jsonl")

    def write_memory(self, memory_event: MemoryEvent) -> MemoryEvent:
        self._require_type(memory_event, MemoryEvent)
        return self._write_event(memory_event, "memory", "memory_id", "memory_events.jsonl")

    def write_trial(self, trial: TrialLog) -> TrialLog:
        self._require_type(trial, TrialLog)
        self._ensure_running()
        self._validate_trial_context(trial)
        enriched = trial.model_copy(update={"event_seq": self._next_event_seq()})
        try:
            self._write_line("trials.jsonl", enriched)
            self._counts["trials"] += 1
            self._flush_streams()
        except BaseException:
            self._mark_failed()
            raise
        return enriched

    def finalize(self, status: str = "completed") -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("final status must be completed or failed")
        if status == "failed":
            if self._state == "running":
                self._mark_failed()
            return
        self._ensure_running()

        try:
            self._manifest["status"] = "completed"
            self._manifest["ended_at"] = _timestamp()
            self._write_manifest()
            self._fsync_close_streams()
            if os.path.lexists(self.run_dir):
                raise FileExistsError(f"final run path already exists: {self.run_dir}")
            self.temp_dir.rename(self.run_dir)
            _fsync_directory(self.run_dir.parent)
            self._state = "completed"
        except BaseException:
            self._mark_failed()
            raise

    @classmethod
    def read_manifest(cls, run_dir: Path | str) -> dict[str, Any]:
        with (Path(run_dir) / "run.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def read_jsonl(cls, run_dir: Path | str, filename: str) -> list[dict[str, Any]]:
        with (Path(run_dir) / filename).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _write_event(
        self,
        event: EventT,
        kind: str,
        identifier_field: str,
        filename: str,
    ) -> EventT:
        self._ensure_running()
        self._validate_event_context(event)
        event_index = self._next_event_index(event.trial_id, kind)
        identifier = getattr(event, identifier_field)
        if kind != "call" or not identifier.startswith(f"{event.trial_id}:call:"):
            identifier = f"{event.trial_id}:{kind}:{event_index}"
        enriched = event.model_copy(
            update={
                identifier_field: identifier,
                "event_seq": self._next_event_seq(),
            }
        )
        try:
            self._write_line(filename, enriched)
            self._counts[self._STREAMS[filename]] += 1
        except BaseException:
            self._mark_failed()
            raise
        return enriched

    def _write_line(
        self, filename: str, model: CallEvent | FailureEvent | FilterEvent | MemoryEvent | TrialLog
    ) -> None:
        payload = json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._streams[filename].write(f"{payload}\n")

    def _write_manifest(self) -> None:
        manifest_tmp = self.temp_dir / "run.json.tmp"
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                self._manifest, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest_tmp.replace(self.temp_dir / "run.json")
        _fsync_directory(self.temp_dir)

    def _flush_streams(self) -> None:
        for stream in self._streams.values():
            stream.flush()

    def _fsync_close_streams(self) -> None:
        for stream in self._streams.values():
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
            self._write_manifest()
        finally:
            for stream in self._streams.values():
                try:
                    if not stream.closed:
                        stream.flush()
                        os.fsync(stream.fileno())
                        stream.close()
                except OSError:
                    stream.close()

    def _ensure_running(self) -> None:
        if self._state != "running":
            raise RuntimeError(f"writer is {self._state}")

    def _validate_event_context(self, event: EventT) -> None:
        if (
            event.run_metadata_id != self.run_metadata.run_metadata_id
            or event.run_id != self.run_metadata.run_id
        ):
            raise ValueError("event run context does not match writer metadata")
        if event.stage != self.run_metadata.stage:
            raise ValueError("event stage does not match writer metadata")
        if self.run_metadata.schema_version == LOGGING_V2 and isinstance(event, MemoryEvent):
            evaluation_law = self.run_metadata.evaluation_law
            if evaluation_law is not None and evaluation_law.regime == "frozen":
                raise ValueError("frozen phase11 runs must not write memory events")

    def _validate_trial_context(self, trial: TrialLog) -> None:
        if (
            trial.run_metadata_id != self.run_metadata.run_metadata_id
            or trial.run_id != self.run_metadata.run_id
        ):
            raise ValueError("trial run context does not match writer metadata")
        if trial.stage != self.run_metadata.stage:
            raise ValueError("trial stage does not match writer metadata")
        if trial.schema_version != self.run_metadata.schema_version:
            raise ValueError("trial schema_version does not match writer metadata")
        if self.run_metadata.schema_version != LOGGING_V2:
            return

        evaluation_law = self.run_metadata.evaluation_law
        target_set = self.run_metadata.target_contamination_set
        if evaluation_law is None or target_set is None:
            raise ValueError("logging_v2 writer metadata requires phase11 contract context")
        if trial.evaluation_law_id != evaluation_law.evaluation_law_id:
            raise ValueError("trial evaluation_law_id does not match writer metadata")
        if trial.target_set_id != target_set.target_set_id:
            raise ValueError("trial target_set_id does not match writer metadata")
        answer_call = next(
            (call for call in trial.method_calls if call.call_id == trial.answer_call_id), None
        )
        if answer_call is None:
            raise ValueError("logging_v2 trial answer_call_id must identify a method call")
        for span in answer_call.source_spans:
            expected_target = span.contamination_class in target_set.included_classes and (
                not target_set.require_exact_lineage or span.lineage_status == "exact"
            )
            if span.is_target_contamination != expected_target:
                raise ValueError("source span target membership does not match writer metadata")
        target_entry_ids = _v2_target_entry_ids(trial.memory_before, target_set)
        exposure = trial.contamination_exposure
        if exposure.target_entry_ids != target_entry_ids:
            raise ValueError("target_entry_ids do not match writer target set")
        exposed_entry_ids = [
            span.entry_id
            for span in answer_call.source_spans
            if span.entry_id in target_entry_ids and span.is_target_contamination
        ]
        if exposure.status == "supported":
            if exposure.is_exposed:
                if (
                    exposure.exposed_entry_ids != exposed_entry_ids
                    or exposure.exposed_source_ids != exposed_entry_ids
                ):
                    raise ValueError("exposed IDs do not match writer target set")
            elif exposure.exposed_entry_ids or exposure.exposed_source_ids:
                raise ValueError("negative exposure must not report exposed IDs")
        expected_pair_id = ":".join(
            [trial.trajectory_pair_id or "", str(trial.checkpoint_index), trial.sample_id]
        )
        if trial.pair_id != expected_pair_id:
            raise ValueError(
                f"trial pair_id does not match trajectory/checkpoint/sample: {trial.pair_id}"
            )
        if evaluation_law.regime == "online":
            if (
                trial.memory_update_mode not in {"enabled", "not_applicable"}
                or trial.checkpoint_ref
            ):
                raise ValueError("online phase11 trial has frozen checkpoint/update context")
            return
        if trial.memory_update_mode not in {"disabled", "not_applicable"}:
            raise ValueError("frozen phase11 trial has enabled memory updates")
        if trial.memory_update_mode == "not_applicable":
            if trial.checkpoint_ref is not None:
                raise ValueError("frozen not_applicable trial must not include checkpoint_ref")
            return
        if trial.checkpoint_ref is None:
            raise ValueError("frozen phase11 trial requires checkpoint_ref")
        checkpoint = trial.checkpoint_ref.model_dump(mode="json")
        if self._frozen_checkpoint is None:
            self._frozen_checkpoint = checkpoint
        elif checkpoint != self._frozen_checkpoint:
            raise ValueError("frozen phase11 checkpoint_ref does not match writer metadata")

    def _next_event_seq(self) -> int:
        self._event_seq += 1
        return self._event_seq

    def _next_event_index(self, trial_id: str, kind: str) -> int:
        key = (trial_id, kind)
        self._event_indices[key] = self._event_indices.get(key, 0) + 1
        return self._event_indices[key]

    @staticmethod
    def _require_type(value: object, expected_type: type[object]) -> None:
        if not isinstance(value, expected_type):
            raise TypeError(f"expected {expected_type.__name__}")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
