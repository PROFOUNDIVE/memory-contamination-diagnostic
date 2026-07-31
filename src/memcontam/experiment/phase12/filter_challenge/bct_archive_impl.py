from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping

from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EvidenceBuildError,
    read_regular_nofollow,
    sha256_regular_nofollow,
)


SHARED_WALL_SECONDS: Final = 10_800
_ZERO_HASH: Final = "0" * 64
_Stage = Literal["screening", "bct"]
_REPORT_IDS: Final = (
    "authority-transition", "methods-lock", "freeze-a", "screening", "freeze-b-search-config",
    "bct-execution", "archive-validation", "claim-scope", "pilot-b-readiness",
)
_REPORT_SCHEMAS: Final = {
    report_id: f"phase12_fv5_{report_id.replace('-', '_')}_report_v1" for report_id in _REPORT_IDS
}


class LedgerError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    calls: int
    input_tokens: int
    output_tokens: int
    microusd: int
    wall_seconds: int


_SHARED_CAP: Final = ResourceBudget(570, 2_334_720, 364_800, 10_000_000, 10_800)
_STAGE_CAPS: Final = {
    "screening": ResourceBudget(90, 368_640, 57_600, 2_000_000, 3_600),
    "bct": ResourceBudget(480, 1_966_080, 307_200, 8_000_000, 7_200),
}


@dataclass(frozen=True, slots=True)
class ProcessReservation:
    reservation_id: str
    stage: _Stage
    resources: ResourceBudget

    @property
    def reserved_wall_seconds(self) -> int:
        return self.resources.wall_seconds


@dataclass(frozen=True, slots=True)
class ProcessDeadline:
    deadline_monotonic: float

    def clamp_timeout(self, requested_seconds: float, monotonic_now: float | None = None) -> float:
        if requested_seconds < 0:
            raise LedgerError("REQUEST_TIMEOUT_INVALID")
        now = time.monotonic() if monotonic_now is None else monotonic_now
        return min(requested_seconds, max(0.0, self.deadline_monotonic - now))


@dataclass(frozen=True, slots=True)
class ArchiveValidation:
    valid: bool
    reason_code: str | None = None


class BudgetLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def remaining_wall_seconds(self) -> int:
        return self.remaining_resources.wall_seconds

    @property
    def remaining_resources(self) -> ResourceBudget:
        return _subtract(_SHARED_CAP, _consumed_resources(self._records()))

    def reserve_process(self, stage: _Stage, run_id: str) -> ProcessReservation:
        requested = _STAGE_CAPS[stage]
        with self._locked() as stream:
            records = _read_records(stream)
            if not _fits(_subtract(_SHARED_CAP, _consumed_resources(records)), requested):
                raise LedgerError("WALL_TIME_CAP_EXCEEDED")
            reservation_id = f"{run_id}:{stage}:wall"
            if reservation_id in {_string_value(row["reservation_id"]) for row in records if row["kind"] == "reserve"}:
                raise LedgerError("RUN_ID_ALREADY_EXISTS")
            _append(
                stream,
                records,
                {"kind": "reserve", "reservation_id": reservation_id, "stage": stage, "resources": _budget_json(requested)},
            )
        return ProcessReservation(reservation_id, stage, requested)

    def settle_process(self, reservation: ProcessReservation, elapsed_seconds: int) -> None:
        self.settle(reservation, ResourceBudget(0, 0, 0, 0, elapsed_seconds))

    def settle(self, reservation: ProcessReservation, used: ResourceBudget) -> None:
        if not _fits(reservation.resources, used):
            raise LedgerError("RESOURCE_SETTLEMENT_INVALID")
        with self._locked() as stream:
            records = _read_records(stream)
            if reservation.reservation_id not in {row["reservation_id"] for row in records if row["kind"] == "reserve"}:
                raise LedgerError("LEDGER_RESERVATION_UNKNOWN")
            if reservation.reservation_id in {row["reservation_id"] for row in records if row["kind"] == "settle"}:
                raise LedgerError("LEDGER_RESERVATION_SETTLED")
            _append(stream, records, {"kind": "settle", "reservation_id": reservation.reservation_id, "resources": _budget_json(used)})

    def deadline_for(self, reservation: ProcessReservation, started_at: float | None = None) -> ProcessDeadline:
        records = self._records()
        if reservation.reservation_id not in {row["reservation_id"] for row in records if row["kind"] == "reserve"}:
            raise LedgerError("LEDGER_RESERVATION_UNKNOWN")
        if reservation.reservation_id in {row["reservation_id"] for row in records if row["kind"] == "settle"}:
            raise LedgerError("LEDGER_RESERVATION_SETTLED")
        now = time.monotonic() if started_at is None else started_at
        return ProcessDeadline(now + reservation.resources.wall_seconds)

    def invalidate_timeout(self, reservation: ProcessReservation) -> None:
        with self._locked() as stream:
            records = _read_records(stream)
            if reservation.reservation_id not in {row["reservation_id"] for row in records if row["kind"] == "reserve"}:
                raise LedgerError("LEDGER_RESERVATION_UNKNOWN")
            _append(stream, records, {"kind": "invalidate", "reservation_id": reservation.reservation_id})

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
    with _LedgerLock(path) as output:
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


def build_evidence_report(bundle: Path, report_id: str, stage_result: Path | None, plan_digest: str) -> Path:
    if report_id not in _REPORT_SCHEMAS:
        raise LedgerError("EVIDENCE_REPORT_CONTRACT_INVALID")
    stage = None if stage_result is None else CalibrationStageResult.model_validate_json(
        read_regular_nofollow(stage_result, "EVIDENCE_STAGE_DIGEST_MISMATCH")
    )
    bundle.mkdir(parents=True, exist_ok=True)
    path = bundle / f"{report_id.replace('-', '_')}_report.json"
    if path.exists():
        raise LedgerError("EVIDENCE_REPORT_EXISTS")
    payload: dict[str, object] = {
        "schema_version": _REPORT_SCHEMAS[report_id],
        "common_envelope": "phase12_fv5_evidence_report_v1",
        "report_id": report_id,
        "producer_argv": "scripts/build_phase12_filter_v5_bct_evidence.py",
        "producer_version": "phase12-filter-v5-bct-v1",
        "producer_code_commit": "6b415fbf3f27103d7d25726f8ce6447f9830a8e3",
        "approved_plan_sha256": plan_digest,
        "stage_result_sha256": None if stage_result is None else sha256_regular_nofollow(stage_result, "EVIDENCE_STAGE_DIGEST_MISMATCH"),
        "stage_result_path": None if stage_result is None else str(stage_result),
        "stage_disposition": "completed" if stage is None else stage.disposition,
        "provider_calls_issued": 0 if stage is None else stage.provider_calls_issued,
    }
    if report_id == "freeze-a":
        payload["input_digests"] = {
            "source_universe": sha256_regular_nofollow(
                Path.cwd() / "data/phase12/filter_v5_bct_v1/source_universe_v1.json",
                "EVIDENCE_SOURCE_UNIVERSE_INVALID",
            )
        }
    payload["output_seal"] = _hash(payload)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def validate_evidence_bundle(bundle: Path, plan_digest: str, through: str = "screening") -> ArchiveValidation:
    try:
        required = {
            "authority-methods": ("authority_transition", "methods_lock"),
            "freeze-a": ("authority_transition", "methods_lock", "freeze_a"),
            "screening": ("authority_transition", "methods_lock", "freeze_a", "screening"),
            "freeze-b": ("authority_transition", "methods_lock", "freeze_a", "screening", "freeze_b_search_config"),
            "bct": ("authority_transition", "methods_lock", "freeze_a", "screening", "freeze_b_search_config", "bct_execution", "archive_validation", "claim_scope"),
            "readiness": ("authority_transition", "methods_lock", "freeze_a", "screening", "freeze_b_search_config", "bct_execution", "archive_validation", "claim_scope", "pilot_b_readiness"),
        }.get(through)
        if required is None:
            raise LedgerError("EVIDENCE_REPORT_MISSING")
        for name in required:
            path = bundle / f"{name}_report.json"
            payload = json.loads(read_regular_nofollow(path, "EVIDENCE_REPORT_INVALID"))
            if not isinstance(payload, dict):
                raise LedgerError("EVIDENCE_REPORT_INVALID")
            report_id = name.replace("_", "-")
            stage_path = payload.get("stage_result_path")
            if (
                payload.get("schema_version") != _REPORT_SCHEMAS[report_id]
                or payload.get("common_envelope") != "phase12_fv5_evidence_report_v1"
                or payload.get("report_id") != report_id
                or payload.get("approved_plan_sha256") != plan_digest
                or payload.get("provider_calls_issued") != 0
                or payload.get("producer_argv") != "scripts/build_phase12_filter_v5_bct_evidence.py"
                or payload.get("producer_version") != "phase12-filter-v5-bct-v1"
                or not isinstance(payload.get("producer_code_commit"), str)
                or payload.get("all_passed") is not None
                or payload.get("output_seal") != _hash({key: value for key, value in payload.items() if key != "output_seal"})
            ):
                raise LedgerError("EVIDENCE_REPORT_CONTRACT_INVALID")
            if report_id == "freeze-a":
                inputs = payload.get("input_digests")
                source = Path.cwd() / "data/phase12/filter_v5_bct_v1/source_universe_v1.json"
                if not isinstance(inputs, dict) or inputs.get("source_universe") != sha256_regular_nofollow(source, "EVIDENCE_SOURCE_UNIVERSE_INVALID"):
                    raise LedgerError("EVIDENCE_SOURCE_UNIVERSE_INVALID")
            if stage_path is not None and (
                not isinstance(stage_path, str)
                or payload.get("stage_result_sha256") != sha256_regular_nofollow(Path(stage_path), "EVIDENCE_STAGE_DIGEST_MISMATCH")
            ):
                raise LedgerError("EVIDENCE_STAGE_DIGEST_MISMATCH")
            if stage_path is not None:
                stage = CalibrationStageResult.model_validate_json(
                    read_regular_nofollow(Path(stage_path), "EVIDENCE_STAGE_DIGEST_MISMATCH")
                )
                if payload.get("stage_disposition") != stage.disposition:
                    raise LedgerError("EVIDENCE_REPORT_CONTRACT_INVALID")
        if "freeze_b_search_config" in required:
            _validate_freeze_b_waiting_report(bundle, plan_digest)
        if "pilot_b_readiness" in required:
            from memcontam.experiment.phase12.filter_challenge.pilot_b_readiness import (
                validate_readiness_report,
            )

            validate_readiness_report(bundle)
    except (EvidenceBuildError, LedgerError, OSError, UnicodeError, json.JSONDecodeError) as error:
        return ArchiveValidation(False, error.code if isinstance(error, LedgerError) else "EVIDENCE_REPORT_INVALID")
    return ArchiveValidation(True)


def _validate_freeze_b_waiting_report(bundle: Path, plan_digest: str) -> None:
    payload = json.loads(
        read_regular_nofollow(bundle / "freeze_b_search_config_report.json", "EVIDENCE_REPORT_INVALID")
    )
    stage_path = payload.get("stage_result_path")
    if not isinstance(stage_path, str):
        raise LedgerError("EVIDENCE_FREEZE_B_WAITING_INVALID")


    stage = CalibrationStageResult.model_validate_json(
        read_regular_nofollow(Path(stage_path), "EVIDENCE_STAGE_DIGEST_MISMATCH")
    )
    upstream = {
        report_id: _sha256(bundle / f"{report_id.replace('-', '_')}_report.json")
        for report_id in ("authority-transition", "methods-lock", "freeze-a", "screening")
    }
    inputs = payload.get("input_digests")
    if (
        payload.get("approved_plan_sha256") != plan_digest
        or payload.get("stage_disposition") != "blocked_before_stage"
        or payload.get("terminal_status") != "AWAITING_SCREENING_AUTHORIZATION"
        or payload.get("provider_calls_issued") != 0
        or stage.stage != "screening"
        or stage.disposition != "blocked_before_stage"
        or stage.terminal_status != "AWAITING_SCREENING_AUTHORIZATION"
        or stage.provider_calls_issued != 0
        or payload.get("upstream_report_sha256") != upstream
        or not isinstance(inputs, dict)
        or inputs.get("freeze_b") is not None
        or inputs.get("search_config") is not None
        or inputs.get("bct_authorization_request") is not None
    ):
        raise LedgerError("EVIDENCE_FREEZE_B_WAITING_INVALID")


def _read_records(stream) -> list[dict[str, object]]:
    stream.seek(0)
    records = [_record_value(json.loads(line)) for line in stream if line.strip()]
    previous = _ZERO_HASH
    for sequence, record in enumerate(records, start=1):
        if record.get("sequence") != sequence or record.get("previous_hash") != previous:
            raise LedgerError("LEDGER_CHAIN_INVALID")
        payload = {key: value for key, value in record.items() if key != "record_hash"}
        if record.get("record_hash") != _hash(payload):
            raise LedgerError("LEDGER_CHAIN_INVALID")
        previous = str(record["record_hash"])
    _validate_ledger_records(records)
    return records


def _append(stream, records: list[dict[str, object]], payload: dict[str, object]) -> None:
    record = {**payload, "sequence": len(records) + 1, "previous_hash": records[-1]["record_hash"] if records else _ZERO_HASH}
    record["record_hash"] = _hash(record)
    stream.seek(0, os.SEEK_END)
    stream.write(_canonical_json(record) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
    _fsync_parent(Path(stream.name))


def _contains_hidden_label(payload: object) -> bool:
    if isinstance(payload, dict):
        return bool({"candidate_role", "correctness_label", "hidden_label"} & payload.keys()) or any(
            _contains_hidden_label(value) for value in payload.values()
        )
    if isinstance(payload, list):
        return any(_contains_hidden_label(value) for value in payload)
    return False


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


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


def _budget_json(value: ResourceBudget) -> dict[str, int]:
    return {"calls": value.calls, "input_tokens": value.input_tokens, "output_tokens": value.output_tokens, "microusd": value.microusd, "wall_seconds": value.wall_seconds}


def _budget_value(value: object) -> ResourceBudget:
    if not isinstance(value, dict):
        raise LedgerError("LEDGER_CHAIN_INVALID")
    budget = ResourceBudget(*(_integer_value(value.get(name)) for name in ("calls", "input_tokens", "output_tokens", "microusd", "wall_seconds")))
    if min(budget.calls, budget.input_tokens, budget.output_tokens, budget.microusd, budget.wall_seconds) < 0:
        raise LedgerError("LEDGER_CHAIN_INVALID")
    return budget


def _fits(limit: ResourceBudget, requested: ResourceBudget) -> bool:
    return all(a >= b for a, b in zip(_budget_json(limit).values(), _budget_json(requested).values(), strict=True))


def _subtract(limit: ResourceBudget, used: ResourceBudget) -> ResourceBudget:
    if not _fits(limit, used):
        raise LedgerError("WALL_TIME_CAP_EXCEEDED")
    return ResourceBudget(*(a - b for a, b in zip(_budget_json(limit).values(), _budget_json(used).values(), strict=True)))


def _consumed_resources(records: list[dict[str, object]]) -> ResourceBudget:
    reserved = { _string_value(row["reservation_id"]): _budget_value(row["resources"]) for row in records if row["kind"] == "reserve" }
    settled = { _string_value(row["reservation_id"]): _budget_value(row["resources"]) for row in records if row["kind"] == "settle" }
    values = tuple(settled.get(identifier, budget) for identifier, budget in reserved.items())
    return ResourceBudget(
        sum(item.calls for item in values),
        sum(item.input_tokens for item in values),
        sum(item.output_tokens for item in values),
        sum(item.microusd for item in values),
        sum(item.wall_seconds for item in values),
    )


def _validate_ledger_records(records: list[dict[str, object]]) -> None:
    reservations: set[str] = set()
    settled: set[str] = set()
    for row in records:
        kind = _string_value(row["kind"])
        match kind:
            case "reserve":
                identifier = _string_value(row["reservation_id"])
                if identifier in reservations or _string_value(row["stage"]) not in _STAGE_CAPS:
                    raise LedgerError("LEDGER_CHAIN_INVALID")
                resources = _budget_value(row["resources"])
                if resources != _STAGE_CAPS[_string_value(row["stage"])]:
                    raise LedgerError("LEDGER_CHAIN_INVALID")
                reservations.add(identifier)
            case "settle":
                identifier = _string_value(row["reservation_id"])
                if identifier not in reservations or identifier in settled or not _fits(_reservation_resources(records, identifier), _budget_value(row["resources"])):
                    raise LedgerError("LEDGER_CHAIN_INVALID")
                settled.add(identifier)
            case "invalidate":
                if _string_value(row["reservation_id"]) not in reservations:
                    raise LedgerError("LEDGER_CHAIN_INVALID")
            case _:
                raise LedgerError("LEDGER_CHAIN_INVALID")


def _reservation_resources(records: list[dict[str, object]], identifier: str) -> ResourceBudget:
    return next(_budget_value(row["resources"]) for row in records if row["kind"] == "reserve" and row["reservation_id"] == identifier)


def _record_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LedgerError("LEDGER_CHAIN_INVALID")
    return value


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _archive_record(records: list[dict[str, object]], start: int, payload: dict[str, object]) -> dict[str, object]:
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
    record = {**payload, "run_id": run_id, "status": status, "failure_code": failure_code, "inclusion_state": inclusion_state, "sequence": len(records) + 1, "previous_hash": records[-1]["record_hash"] if records else _ZERO_HASH, "raw_byte_start": start, "raw_byte_end": start}
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
    with _LedgerLock(path) as stream:
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
