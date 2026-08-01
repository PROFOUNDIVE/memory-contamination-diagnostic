from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.bct_archive_models import (
    LedgerError,
    ProcessDeadline,
    ProcessReservation,
    ResourceBudget,
    STAGE_CAPS,
    _Stage,
    resource_fits,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_ledger_validation import (
    budget_value,
    validate_ledger_records,
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


SHARED_WALL_SECONDS: Final = 10_800
_SHARED_CAP: Final = ResourceBudget(570, 2_334_720, 364_800, 10_000_000, SHARED_WALL_SECONDS)


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
        requested = STAGE_CAPS[stage]
        with self._locked() as stream:
            records = _read_records(stream)
            if not resource_fits(_subtract(_SHARED_CAP, _consumed_resources(records)), requested):
                raise LedgerError("WALL_TIME_CAP_EXCEEDED")
            reservation_id = f"{run_id}:{stage}:wall"
            if reservation_id in {
                _string_value(row["reservation_id"])
                for row in records
                if row["kind"] == "reserve"
            }:
                raise LedgerError("RUN_ID_ALREADY_EXISTS")
            _append(
                stream,
                records,
                {
                    "kind": "reserve",
                    "reservation_id": reservation_id,
                    "stage": stage,
                    "resources": _budget_json(requested),
                },
            )
        return ProcessReservation(reservation_id, stage, requested)

    def settle_process(self, reservation: ProcessReservation, elapsed_seconds: int) -> None:
        self.settle(reservation, ResourceBudget(0, 0, 0, 0, elapsed_seconds))

    def settle(self, reservation: ProcessReservation, used: ResourceBudget) -> None:
        if not resource_fits(reservation.resources, used):
            raise LedgerError("RESOURCE_SETTLEMENT_INVALID")
        with self._locked() as stream:
            records = _read_records(stream)
            if reservation.reservation_id not in {
                row["reservation_id"] for row in records if row["kind"] == "reserve"
            }:
                raise LedgerError("LEDGER_RESERVATION_UNKNOWN")
            if reservation.reservation_id in {
                row["reservation_id"] for row in records if row["kind"] == "settle"
            }:
                raise LedgerError("LEDGER_RESERVATION_SETTLED")
            _append(
                stream,
                records,
                {"kind": "settle", "reservation_id": reservation.reservation_id, "resources": _budget_json(used)},
            )

    def deadline_for(
        self, reservation: ProcessReservation, started_at: float | None = None
    ) -> ProcessDeadline:
        records = self._records()
        if reservation.reservation_id not in {
            row["reservation_id"] for row in records if row["kind"] == "reserve"
        }:
            raise LedgerError("LEDGER_RESERVATION_UNKNOWN")
        if reservation.reservation_id in {
            row["reservation_id"] for row in records if row["kind"] == "settle"
        }:
            raise LedgerError("LEDGER_RESERVATION_SETTLED")
        now = time.monotonic() if started_at is None else started_at
        return ProcessDeadline(now + reservation.resources.wall_seconds)

    def invalidate_timeout(self, reservation: ProcessReservation) -> None:
        with self._locked() as stream:
            records = _read_records(stream)
            if reservation.reservation_id not in {
                row["reservation_id"] for row in records if row["kind"] == "reserve"
            }:
                raise LedgerError("LEDGER_RESERVATION_UNKNOWN")
            _append(stream, records, {"kind": "invalidate", "reservation_id": reservation.reservation_id})

    def head(self) -> str:
        records = self._records()
        return _string_value(records[-1]["record_hash"]) if records else _ZERO_HASH

    def _records(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self._locked() as stream:
            return _read_records(stream)

    def _locked(self) -> _LockedJsonl:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return _LockedJsonl(self.path)


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
    validate_ledger_records(records, STAGE_CAPS)
    return records


def _append(stream, records: list[dict[str, object]], payload: dict[str, object]) -> None:
    record = {
        **payload,
        "sequence": len(records) + 1,
        "previous_hash": records[-1]["record_hash"] if records else _ZERO_HASH,
    }
    record["record_hash"] = _hash(record)
    stream.seek(0, os.SEEK_END)
    stream.write(_canonical_json(record) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
    _fsync_parent(Path(stream.name))


def _budget_json(value: ResourceBudget) -> dict[str, int]:
    return {
        "calls": value.calls,
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "microusd": value.microusd,
        "wall_seconds": value.wall_seconds,
    }


def _subtract(limit: ResourceBudget, used: ResourceBudget) -> ResourceBudget:
    if not resource_fits(limit, used):
        raise LedgerError("WALL_TIME_CAP_EXCEEDED")
    return ResourceBudget(
        *(available - consumed for available, consumed in zip(_budget_json(limit).values(), _budget_json(used).values(), strict=True))
    )


def _consumed_resources(records: list[dict[str, object]]) -> ResourceBudget:
    reserved = {
        _string_value(row["reservation_id"]): budget_value(row["resources"])
        for row in records
        if row["kind"] == "reserve"
    }
    settled = {
        _string_value(row["reservation_id"]): budget_value(row["resources"])
        for row in records
        if row["kind"] == "settle"
    }
    values = tuple(settled.get(identifier, budget) for identifier, budget in reserved.items())
    return ResourceBudget(
        sum(item.calls for item in values),
        sum(item.input_tokens for item in values),
        sum(item.output_tokens for item in values),
        sum(item.microusd for item in values),
        sum(item.wall_seconds for item in values),
    )
