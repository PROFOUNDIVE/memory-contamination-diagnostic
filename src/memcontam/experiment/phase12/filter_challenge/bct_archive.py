from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult


SHARED_WALL_SECONDS: Final = 10_800
PROCESS_WALL_SECONDS: Final = {"screening": 3_600, "bct": 7_200}


class LedgerError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ProcessReservation:
    reservation_id: str
    reserved_wall_seconds: int


@dataclass(frozen=True, slots=True)
class ArchiveValidation:
    valid: bool
    reason_code: str | None = None


class BudgetLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def remaining_wall_seconds(self) -> int:
        return SHARED_WALL_SECONDS - _reserved_wall_seconds(self._records())

    def reserve_process(self, stage: Literal["screening", "bct"], run_id: str) -> ProcessReservation:
        requested = PROCESS_WALL_SECONDS[stage]
        with self._locked() as stream:
            records = _read_records(stream)
            if SHARED_WALL_SECONDS - _reserved_wall_seconds(records) < requested:
                raise LedgerError("WALL_TIME_CAP_EXCEEDED")
            reservation_id = f"{run_id}:{stage}:wall"
            if reservation_id in {_string_value(row["reservation_id"]) for row in records if row["kind"] == "reserve"}:
                raise LedgerError("RUN_ID_ALREADY_EXISTS")
            _append(stream, records, {"kind": "reserve", "reservation_id": reservation_id, "wall_seconds": requested})
        return ProcessReservation(reservation_id, requested)

    def settle_process(self, reservation: ProcessReservation, elapsed_seconds: int) -> None:
        if elapsed_seconds < 0 or elapsed_seconds > reservation.reserved_wall_seconds:
            raise LedgerError("WALL_TIME_SETTLEMENT_INVALID")
        with self._locked() as stream:
            records = _read_records(stream)
            if reservation.reservation_id not in {row["reservation_id"] for row in records if row["kind"] == "reserve"}:
                raise LedgerError("LEDGER_RESERVATION_UNKNOWN")
            if reservation.reservation_id in {row["reservation_id"] for row in records if row["kind"] == "settle"}:
                raise LedgerError("LEDGER_RESERVATION_SETTLED")
            _append(stream, records, {"kind": "settle", "reservation_id": reservation.reservation_id, "wall_seconds": elapsed_seconds})

    def head(self) -> str:
        records = self._records()
        return _string_value(records[-1]["record_hash"]) if records else "0" * 64

    def _records(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self._locked() as stream:
            return _read_records(stream)

    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return _LedgerLock(self.path)


class _LedgerLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._stream = None

    def __enter__(self):
        self._stream = self._path.open("a+", encoding="utf-8")
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX)
        return self._stream

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self._stream is not None
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()


def append_archive_record(root: Path, stream: Literal["public", "audit"], payload: dict[str, object]) -> None:
    if stream == "public" and _contains_hidden_label(payload):
        raise LedgerError("AUDIT_FIELD_IN_PUBLIC_STREAM")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stream}.jsonl"
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


def validate_live_archive(root: Path) -> ArchiveValidation:
    try:
        public = _jsonl(root / "public.jsonl")
        audit = _jsonl(root / "audit.jsonl")
        if not public or not audit or any(_contains_hidden_label(row) for row in public):
            raise LedgerError("LIVE_ARCHIVE_INVALID")
    except (LedgerError, OSError, UnicodeError, json.JSONDecodeError) as error:
        return ArchiveValidation(False, error.code if isinstance(error, LedgerError) else "LIVE_ARCHIVE_INVALID")
    return ArchiveValidation(True)


def build_evidence_report(bundle: Path, report_id: str, stage_result: Path, plan_digest: str) -> Path:
    stage = CalibrationStageResult.model_validate_json(stage_result.read_text(encoding="utf-8"))
    bundle.mkdir(parents=True, exist_ok=True)
    path = bundle / f"{report_id}_report.json"
    if path.exists():
        raise LedgerError("EVIDENCE_REPORT_EXISTS")
    payload = {
        "schema_version": "phase12_fv5_evidence_report_v1",
        "report_id": report_id,
        "approved_plan_sha256": plan_digest,
        "stage_result_sha256": _sha256(stage_result),
        "stage_result_path": str(stage_result),
        "stage_disposition": stage.disposition,
        "provider_calls_issued": stage.provider_calls_issued,
    }
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def validate_evidence_bundle(bundle: Path, plan_digest: str, through: str = "screening") -> ArchiveValidation:
    try:
        required = {
            "authority-methods": ("authority-transition", "methods-lock"),
            "freeze-a": ("authority-transition", "methods-lock", "freeze-a"),
            "screening": ("authority-transition", "methods-lock", "freeze-a", "screening"),
            "freeze-b": ("authority-transition", "methods-lock", "freeze-a", "screening", "freeze-b-search-config"),
            "bct": ("authority-transition", "methods-lock", "freeze-a", "screening", "freeze-b-search-config", "bct-execution", "archive-validation", "claim-scope"),
            "readiness": ("authority-transition", "methods-lock", "freeze-a", "screening", "freeze-b-search-config", "bct-execution", "archive-validation", "claim-scope", "pilot-b-readiness"),
        }.get(through)
        if required is None or any(not (bundle / f"{name}_report.json").is_file() for name in required):
            raise LedgerError("EVIDENCE_REPORT_MISSING")
        for path in tuple(bundle.glob("*_report.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            stage_path = payload.get("stage_result_path")
            if payload.get("approved_plan_sha256") != plan_digest or not isinstance(stage_path, str) or payload.get("stage_result_sha256") != _sha256(Path(stage_path)):
                raise LedgerError("EVIDENCE_PLAN_DIGEST_MISMATCH")
    except (LedgerError, OSError, UnicodeError, json.JSONDecodeError) as error:
        return ArchiveValidation(False, error.code if isinstance(error, LedgerError) else "EVIDENCE_REPORT_INVALID")
    return ArchiveValidation(True)


def _read_records(stream) -> list[dict[str, object]]:
    stream.seek(0)
    records = [json.loads(line) for line in stream if line.strip()]
    previous = "0" * 64
    for sequence, record in enumerate(records, start=1):
        if record.get("sequence") != sequence or record.get("previous_hash") != previous:
            raise LedgerError("LEDGER_CHAIN_INVALID")
        payload = {key: value for key, value in record.items() if key != "record_hash"}
        if record.get("record_hash") != _hash(payload):
            raise LedgerError("LEDGER_CHAIN_INVALID")
        previous = str(record["record_hash"])
    return records


def _append(stream, records: list[dict[str, object]], payload: dict[str, object]) -> None:
    record = {**payload, "sequence": len(records) + 1, "previous_hash": records[-1]["record_hash"] if records else "0" * 64}
    record["record_hash"] = _hash(record)
    stream.seek(0, os.SEEK_END)
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _reserved_wall_seconds(records: list[dict[str, object]]) -> int:
    reserved = {
        _string_value(row["reservation_id"]): _integer_value(row["wall_seconds"])
        for row in records
        if row["kind"] == "reserve"
    }
    settled = {
        _string_value(row["reservation_id"]): _integer_value(row["wall_seconds"])
        for row in records
        if row["kind"] == "settle"
    }
    return sum(settled.get(key, value) for key, value in reserved.items())


def _contains_hidden_label(payload: dict[str, object]) -> bool:
    return bool({"candidate_role", "correctness_label", "hidden_label"} & payload.keys())


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerError("LEDGER_CHAIN_INVALID")
    return value


def _string_value(value: object) -> str:
    if not isinstance(value, str):
        raise LedgerError("LEDGER_CHAIN_INVALID")
    return value
