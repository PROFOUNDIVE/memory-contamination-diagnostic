from __future__ import annotations

import hashlib
import json
from pathlib import Path

from memcontam.manifests.phase13_archive_authority import ArchiveProjection, StreamProjection
from memcontam.manifests.phase13_archive_models import Phase13Archive, SourceAttemptRow


class SourceValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def validate_archive_sources(
    archive: Phase13Archive, projection: ArchiveProjection
) -> dict[str, SourceAttemptRow]:
    attempts: dict[str, SourceAttemptRow] = {}
    completed: dict[str, SourceAttemptRow] = {}
    source_ids: set[str] = set()
    for row in archive.source_attempts:
        if row.attempt_id in attempts:
            raise SourceValidationError("DUPLICATE_ATTEMPT_ID")
        if row.source_run_id in source_ids:
            raise SourceValidationError("DUPLICATE_SOURCE_RUN_ID")
        source_ids.add(row.source_run_id)
        if row.rerun_parent_id is not None and row.rerun_parent_id not in attempts:
            raise SourceValidationError("RERUN_PARENT_MISMATCH")
        if row.status == "invalidated":
            selected = _validate_raw(row)
            if not row.invalidated_reason or row.rerun_parent_id is not None:
                raise SourceValidationError("INVALIDATED_ATTEMPT_EVIDENCE_REQUIRED")
            if hashlib.sha256(selected).hexdigest() != row.raw_evidence_sha256:
                raise SourceValidationError("INVALIDATED_RAW_EVIDENCE_MISMATCH")
        else:
            if row.invalidated_reason is not None or row.rerun_parent_id is None:
                raise SourceValidationError("INVALIDATED_ATTEMPT_EVIDENCE_REQUIRED")
            stream = projection.streams.get(row.source_run_id)
            if stream is None:
                raise SourceValidationError("SOURCE_RUN_UNREGISTERED")
            _validate_completed(row, stream, projection)
            _validate_raw(row)
            completed[row.source_run_id] = row
        attempts[row.attempt_id] = row
    if not completed:
        raise SourceValidationError("COMPLETED_SOURCE_REQUIRED")
    if len(completed) != 1:
        raise SourceValidationError("SOURCE_EXECUTION_MULTIPLICITY_INVALID")
    return completed


def _validate_raw(row: SourceAttemptRow) -> bytes:
    try:
        lines = Path(row.source_raw_path).read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise SourceValidationError("SOURCE_RAW_HASH_MISMATCH") from error
    if hashlib.sha256(b"".join(lines)).hexdigest() != row.source_raw_sha256:
        raise SourceValidationError("SOURCE_RAW_HASH_MISMATCH")
    start, end = row.raw_record_range
    if start < 0 or end < start or end >= len(lines) or end - start + 1 != len(row.events):
        raise SourceValidationError("SOURCE_RAW_RANGE_MISMATCH")
    selected = b"".join(lines[start : end + 1])
    try:
        raw_rows = tuple(json.loads(line) for line in lines[start : end + 1])
    except json.JSONDecodeError as error:
        raise SourceValidationError("SOURCE_RAW_RANGE_MISMATCH") from error
    if tuple(event.model_dump(mode="json") for event in row.events) != raw_rows:
        raise SourceValidationError("SOURCE_RAW_RANGE_MISMATCH")
    return selected


def _validate_completed(
    row: SourceAttemptRow, stream: StreamProjection, projection: ArchiveProjection
) -> None:
    execution = projection.execution
    if row.source_manifest_id != stream.stream_id or row.source_ordered_stream_sha256 != stream.ordered_stream_sha256:
        raise SourceValidationError("SOURCE_MANIFEST_IDENTITY_MISMATCH")
    if row.execution_contract_id != execution.execution_contract_id:
        raise SourceValidationError("SOURCE_EXECUTION_CONTRACT_MISMATCH")
    if len(row.events) != 10 or tuple(event.event_time for event in row.events) != tuple(range(10)):
        raise SourceValidationError("SOURCE_H10_RANGE_INVALID")
    if tuple(event.absolute_trial_index for event in row.events) != tuple(range(2, 12)):
        raise SourceValidationError("SOURCE_H10_RANGE_INVALID")
    calls: set[str] = set()
    event_ids: set[str] = set()
    for index, event in enumerate(row.events):
        _validate_event_identity(event, row, stream, projection)
        if event.event_id in event_ids:
            raise SourceValidationError("DUPLICATE_EVENT_ID")
        event_ids.add(event.event_id)
        if event.semantic_call_id in calls:
            raise SourceValidationError("DUPLICATE_SEMANTIC_CALL_ID")
        calls.add(event.semantic_call_id)
        parents = () if index == 0 else (row.events[index - 1].event_id,)
        if event.lineage_parent_ids != parents:
            raise SourceValidationError("EVENT_LINEAGE_MISMATCH")
        if index == 0 and event.state_before_sha256 != stream.checkpoint_hashes[event.baseline]:
            raise SourceValidationError("EVENT_INITIAL_STATE_MISMATCH")
        if index and row.events[index - 1].state_after_sha256 != event.state_before_sha256:
            raise SourceValidationError("EVENT_STATE_CHAIN_MISMATCH")


def _validate_event_identity(event, row, stream, projection) -> None:  # noqa: ANN001
    identities = projection.execution.identities
    if event.task != stream.task:
        raise SourceValidationError("SOURCE_TASK_MISMATCH")
    if event.model != identities.model_snapshot_id:
        raise SourceValidationError("SOURCE_MODEL_MISMATCH")
    if event.session_id != f"session-{stream.seed_id}":
        raise SourceValidationError("SOURCE_SESSION_MISMATCH")
    if event.native_state_id != identities.native_capacity_registry_id:
        raise SourceValidationError("NATIVE_STATE_MISMATCH")
    if event.baseline not in stream.checkpoints or event.source_checkpoint_id != stream.checkpoints[event.baseline]:
        raise SourceValidationError("SOURCE_CHECKPOINT_UNREGISTERED")
    templates = {
        (row.task, row.baseline, row.arm_key.lower())
        for row in projection.execution.execution_templates
    }
    if (event.task, event.baseline, event.arm) not in templates:
        raise SourceValidationError("SOURCE_EXECUTION_TEMPLATE_MISMATCH")
    if (event.arm == "clean") != (event.intervention_id is None):
        raise SourceValidationError("SOURCE_INTERVENTION_MISMATCH")
    if event.call_owner_id != projection.execution.execution_owner_id:
        raise SourceValidationError("EVENT_CALL_OWNER_MISMATCH")
    if event.write_event_ids != (f"write-{event.event_time}",):
        raise SourceValidationError("SOURCE_WRITE_IDENTITY_MISMATCH")
    if event.retention_event_ids != (f"retain-{event.event_time}",):
        raise SourceValidationError("SOURCE_RETENTION_IDENTITY_MISMATCH")
    if event.eviction_event_ids:
        raise SourceValidationError("SOURCE_EVICTION_IDENTITY_MISMATCH")


__all__ = ("SourceValidationError", "validate_archive_sources")
