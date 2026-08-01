from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from memcontam.experiment.phase12.filter_challenge.bct_archive_models import (
    ArchiveValidation,
    LedgerError,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_storage import (
    _LockedJsonl,
    _ZERO_HASH,
    _canonical_json,
    _fsync_parent,
    _hash,
    _record_value,
    _string_value,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import EvidenceBuildError


def append_archive_record(root: Path, stream: Literal["public", "audit"], payload: dict[str, object]) -> None:
    if stream == "public" and _contains_hidden_label(payload):
        raise LedgerError("AUDIT_FIELD_IN_PUBLIC_STREAM")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stream}.jsonl"
    with _LockedJsonl(path) as output:
        records = _read_archive_records(output)
        record = _archive_record(records, output.tell(), payload)
        output.seek(0, os.SEEK_END)
        output.write(_canonical_json(record) + "\n")
        output.flush()
        os.fsync(output.fileno())
        _fsync_parent(path)


def validate_live_archive(root: Path) -> ArchiveValidation:
    try:
        public = _read_archive_path(root / "public.jsonl")
        audit = _read_archive_path(root / "audit.jsonl")
        if not public or not audit or any(_contains_hidden_label(row) for row in public):
            raise LedgerError("LIVE_ARCHIVE_INVALID")
        public_runs = [_string_value(row["run_id"]) for row in public]
        audit_runs = [_string_value(row["run_id"]) for row in audit]
        if len(public_runs) != len(set(public_runs)) or len(audit_runs) != len(set(audit_runs)):
            raise LedgerError("LIVE_ARCHIVE_REUSED_RUN_ID")
        if set(public_runs) != set(audit_runs):
            raise LedgerError("LIVE_ARCHIVE_RECONCILIATION_MISSING")
        for public_row, audit_row in zip(public, audit, strict=True):
            if _archive_identity(public_row) != _archive_identity(audit_row):
                raise LedgerError("LIVE_ARCHIVE_RECONCILIATION_MISSING")
    except (EvidenceBuildError, LedgerError, OSError, UnicodeError, json.JSONDecodeError) as error:
        return ArchiveValidation(False, error.code if isinstance(error, LedgerError) else "LIVE_ARCHIVE_INVALID")
    return ArchiveValidation(True)


def _contains_hidden_label(payload: object) -> bool:
    if isinstance(payload, dict):
        return bool({"candidate_role", "correctness_label", "hidden_label"} & payload.keys()) or any(
            _contains_hidden_label(value) for value in payload.values()
        )
    if isinstance(payload, list):
        return any(_contains_hidden_label(value) for value in payload)
    return False


def _archive_record(
    records: list[dict[str, object]], start: int, payload: dict[str, object]
) -> dict[str, object]:
    run_id = _string_value(payload.get("run_id"))
    status = _string_value(payload.get("status"))
    if status not in {"planned", "completed", "invalidated", "not_issued"}:
        raise LedgerError("LIVE_ARCHIVE_STATUS_INVALID")
    failure_code = payload.get("failure_code")
    if failure_code is not None:
        _string_value(failure_code)
    inclusion_state = payload.get("inclusion_state", "included")
    if inclusion_state not in {"included", "excluded"}:
        raise LedgerError("LIVE_ARCHIVE_INCLUSION_INVALID")
    record = {
        **payload,
        "run_id": run_id,
        "status": status,
        "failure_code": failure_code,
        "inclusion_state": inclusion_state,
        "sequence": len(records) + 1,
        "previous_hash": records[-1]["record_hash"] if records else _ZERO_HASH,
        "raw_byte_start": start,
        "raw_byte_end": start,
    }
    for _ in range(4):
        record["record_hash"] = _hash({key: value for key, value in record.items() if key != "record_hash"})
        end = start + len((_canonical_json(record) + "\n").encode("utf-8"))
        if record["raw_byte_end"] == end:
            return record
        record["raw_byte_end"] = end
    raise LedgerError("LIVE_ARCHIVE_RANGE_INVALID")


def _read_archive_records(stream) -> list[dict[str, object]]:
    stream.seek(0)
    offset = 0
    records: list[dict[str, object]] = []
    for line in stream:
        encoded = line.encode("utf-8")
        if line.strip():
            record = _record_value(json.loads(line))
            if record.get("raw_byte_start") != offset or record.get("raw_byte_end") != offset + len(encoded):
                raise LedgerError("LIVE_ARCHIVE_RANGE_INVALID")
            records.append(record)
        offset += len(encoded)
    previous = _ZERO_HASH
    for sequence, record in enumerate(records, start=1):
        if record.get("sequence") != sequence or record.get("previous_hash") != previous:
            raise LedgerError("LIVE_ARCHIVE_CHAIN_INVALID")
        unsigned = {key: value for key, value in record.items() if key != "record_hash"}
        if record.get("record_hash") != _hash(unsigned):
            raise LedgerError("LIVE_ARCHIVE_CHAIN_INVALID")
        previous = _string_value(record["record_hash"])
    return records


def _read_archive_path(path: Path) -> list[dict[str, object]]:
    with _LockedJsonl(path) as stream:
        return _read_archive_records(stream)


def _archive_identity(row: dict[str, object]) -> tuple[str, str, str | None, str]:
    failure = row["failure_code"]
    status = _string_value(row["status"])
    inclusion = _string_value(row["inclusion_state"])
    if status not in {"planned", "completed", "invalidated", "not_issued"}:
        raise LedgerError("LIVE_ARCHIVE_STATUS_INVALID")
    if inclusion not in {"included", "excluded"}:
        raise LedgerError("LIVE_ARCHIVE_INCLUSION_INVALID")
    return (_string_value(row["run_id"]), status, None if failure is None else _string_value(failure), inclusion)
