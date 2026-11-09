from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .phase13_live_diagnostic_models import LiveDiagnosticBinding

from memcontam.readiness.phase13_main_runner_models import (
    LEDGER_SCHEMA_ID,
    ExecutionUnit,
    MainRunBinding,
    MainRunError,
    SessionState,
    units_sha256,
)


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        yield connection
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        connection.close()


def canonical(value: list) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def ledger_genesis(connection: sqlite3.Connection) -> str:
    metadata = sorted(connection.execute("SELECT key, value FROM metadata").fetchall())
    domain = ("phase13-live-diagnostic-event-genesis-v1" if
              dict(metadata).get("ledger_schema_id") == "phase13-live-diagnostic-ledger-v1"
              else "phase13-main-a-event-genesis-v1")
    return hashlib.sha256(canonical([domain, metadata])).hexdigest()


def _full_units_identity_sha256(units: tuple[ExecutionUnit, ...]) -> str:
    return hashlib.sha256(canonical([asdict(unit) for unit in units])).hexdigest()


def expected_metadata(
    binding: MainRunBinding | LiveDiagnosticBinding,
    units: tuple[ExecutionUnit, ...],
) -> dict[str, str]:
    if isinstance(binding, LiveDiagnosticBinding):
        return {"ledger_schema_id": "phase13-live-diagnostic-ledger-v1", "currency": "USD",
                "binding": binding.model_dump_json(), "binding_sha256": binding.sha256(),
                "full_units_identity_sha256": _full_units_identity_sha256(units)}
    return {
        "ledger_schema_id": LEDGER_SCHEMA_ID,
        "units_sha256": units_sha256(units),
        "full_units_identity_sha256": _full_units_identity_sha256(units),
        **{key: str(value) for key, value in asdict(binding).items()},
    }


def create_ledger_file(
    path: Path,
    binding: MainRunBinding | LiveDiagnosticBinding,
    units: tuple[ExecutionUnit, ...],
) -> None:
    if path.exists():
        raise MainRunError("MAIN_RUN_LEDGER_ALREADY_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _initialize_ledger(temporary, binding, units)
        with temporary.open("rb") as ledger_file:
            os.fsync(ledger_file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise MainRunError("MAIN_RUN_LEDGER_ALREADY_EXISTS") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}-wal").unlink(missing_ok=True)
        Path(f"{temporary}-shm").unlink(missing_ok=True)


def _initialize_ledger(
    path: Path,
    binding: MainRunBinding | LiveDiagnosticBinding,
    units: tuple[ExecutionUnit, ...],
) -> None:
    with connect(path) as connection:
        cost_column = ("realized_cost_usd_micros INTEGER" if isinstance(binding, LiveDiagnosticBinding)
                       else "realized_cost_krw INTEGER NOT NULL")
        connection.executescript(
            f"""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE execution_units (
                sequence INTEGER PRIMARY KEY, unit_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL, seed INTEGER NOT NULL, task TEXT NOT NULL,
                memory_baseline TEXT, arm TEXT NOT NULL, state TEXT NOT NULL
            );
            CREATE TABLE events (
                event_sequence INTEGER PRIMARY KEY, unit_id TEXT NOT NULL,
                kind TEXT NOT NULL, evidence_sha256 TEXT, reason TEXT,
                {cost_column}, previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            """
        )
        metadata = expected_metadata(binding, units)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.executemany(
            "INSERT INTO execution_units VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING')",
            (
                (
                    unit.sequence,
                    unit.unit_id,
                    unit.kind,
                    unit.seed,
                    unit.task,
                    unit.memory_baseline,
                    unit.arm,
                )
                for unit in units
            ),
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def transition(current: str, kind: str) -> str:
    transitions = {
        ("PENDING", "INTENT"): "DISPATCH_INTENT_PERSISTED",
        ("PENDING", "PAUSED"): "PENDING",
        ("DISPATCH_INTENT_PERSISTED", "COMPLETED"): "COMPLETED",
        ("DISPATCH_INTENT_PERSISTED", "TERMINAL"): "TERMINAL_TECHNICAL_MISSING",
        ("PENDING", "DEPENDENCY_TERMINAL"): "TERMINAL_TECHNICAL_MISSING",
        ("DISPATCH_INTENT_PERSISTED", "NO_REQUEST"): "PENDING",
        ("DISPATCH_INTENT_PERSISTED", "REQUEST_INTENT"): "DISPATCH_INTENT_PERSISTED",
        ("DISPATCH_INTENT_PERSISTED", "REQUEST_RECEIPT"): "DISPATCH_INTENT_PERSISTED",
    }
    try:
        return transitions[(current, kind)]
    except KeyError as error:
        raise MainRunError("MAIN_RUN_STATE_TRANSITION_INVALID") from error


def session_state(
    last_kind: str | None,
    pending: int,
    in_flight: int,
    completed: int,
    terminal: int,
) -> SessionState:
    if in_flight:
        return "BLOCKED_IN_FLIGHT"
    if pending == 0:
        return "COMPLETED"
    if last_kind == "PAUSED":
        return "PAUSED_BEFORE_DISPATCH"
    if last_kind == "TERMINAL":
        return "STOPPED_TERMINAL_TECHNICAL_MISSING"
    if completed + terminal == 0:
        return "NOT_STARTED"
    return "READY"


def validate_integrity(
    connection: sqlite3.Connection,
    units: tuple[ExecutionUnit, ...],
) -> tuple[list[str], str | None, int]:
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if "full_units_identity_sha256" not in metadata:
        raise MainRunError("MAIN_RUN_UNIT_IDENTITY_BINDING_MISSING")
    if metadata["full_units_identity_sha256"] != _full_units_identity_sha256(units):
        raise MainRunError("MAIN_RUN_BINDING_MISMATCH")
    rows = connection.execute(
        "SELECT sequence, unit_id, kind, seed, task, memory_baseline, arm, state "
        "FROM execution_units ORDER BY sequence"
    ).fetchall()
    if len(rows) != len(units):
        raise MainRunError("MAIN_RUN_LEDGER_INTEGRITY_INVALID")
    projected = {unit.unit_id: "PENDING" for unit in units}
    for row, unit in zip(rows, units, strict=True):
        if tuple(row[:7]) != (
            unit.sequence,
            unit.unit_id,
            unit.kind,
            unit.seed,
            unit.task,
            unit.memory_baseline,
            unit.arm,
        ):
            raise MainRunError("MAIN_RUN_LEDGER_INTEGRITY_INVALID")
    previous_hash = ledger_genesis(connection)
    last_kind: str | None = None
    realized_cost = 0
    events = connection.execute("SELECT * FROM events ORDER BY event_sequence").fetchall()
    for expected_sequence, event in enumerate(events):
        values = list(event[:7])
        if event[0] != expected_sequence or event[6] != previous_hash:
            raise MainRunError("MAIN_RUN_LEDGER_INTEGRITY_INVALID")
        if hashlib.sha256(canonical(values)).hexdigest() != event[7]:
            raise MainRunError("MAIN_RUN_LEDGER_INTEGRITY_INVALID")
        unit_id, kind = str(event[1]), str(event[2])
        if unit_id not in projected:
            raise MainRunError("MAIN_RUN_LEDGER_INTEGRITY_INVALID")
        projected[unit_id] = transition(projected[unit_id], kind)
        if event[5] is None and metadata.get("currency") != "USD":
            raise MainRunError("MAIN_RUN_LEDGER_INTEGRITY_INVALID")
        realized_cost += 0 if event[5] is None else int(event[5])
        previous_hash = str(event[7])
        last_kind = kind
    observed = {str(row[1]): str(row[7]) for row in rows}
    if projected != observed:
        raise MainRunError("MAIN_RUN_LEDGER_INTEGRITY_INVALID")
    return list(projected.values()), last_kind, realized_cost


def append_event(connection: sqlite3.Connection, unit_id: str, kind: str,
                 evidence_sha256: str | None = None, reason: str | None = None,
                 realized_cost: int | None = 0) -> None:
    current_row = connection.execute(
        "SELECT state FROM execution_units WHERE unit_id = ?", (unit_id,)
    ).fetchone()
    if current_row is None:
        raise MainRunError("MAIN_RUN_UNIT_UNKNOWN")
    target = transition(str(current_row[0]), kind)
    previous = connection.execute(
        "SELECT event_hash FROM events ORDER BY event_sequence DESC LIMIT 1"
    ).fetchone()
    previous_hash = ledger_genesis(connection) if previous is None else str(previous[0])
    sequence = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    values = [sequence, unit_id, kind, evidence_sha256, reason, realized_cost, previous_hash]
    event_hash = hashlib.sha256(canonical(values)).hexdigest()
    connection.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (*values, event_hash))
    connection.execute("UPDATE execution_units SET state = ? WHERE unit_id = ?", (target, unit_id))
