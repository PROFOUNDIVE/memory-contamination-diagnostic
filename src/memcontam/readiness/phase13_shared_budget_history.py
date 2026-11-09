"""Transactional financial history; detects projection corruption, not wholesale artifact forgery."""
from __future__ import annotations

import hashlib
import sqlite3
from typing import NamedTuple

from pydantic import TypeAdapter

from .phase13_main_runner_store import canonical
from .phase13_shared_budget_models import BudgetError


class ReservationRow(NamedTuple):
    reservation_id: str
    operation: str
    run_id: str
    request_id: str
    attempt_id: str
    state: str
    known: int
    held: int
    proof: str | None


def create_history(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE history (
            sequence INTEGER PRIMARY KEY, snapshot TEXT NOT NULL,
            previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE);
        CREATE TRIGGER history_no_update BEFORE UPDATE ON history
            BEGIN SELECT RAISE(ABORT, 'BUDGET_HISTORY_APPEND_ONLY'); END;
        CREATE TRIGGER history_no_delete BEFORE DELETE ON history
            BEGIN SELECT RAISE(ABORT, 'BUDGET_HISTORY_APPEND_ONLY'); END;
    """)


def genesis(binding_sha256: str) -> str:
    return hashlib.sha256(canonical(["phase13-shared-budget-history-v1", binding_sha256])).hexdigest()


def append_history(connection: sqlite3.Connection, reservation_id: str, binding_sha256: str) -> None:
    row = connection.execute("SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)).fetchone()
    snapshot = canonical(list(row)).decode()
    previous = connection.execute("SELECT sequence,event_hash FROM history ORDER BY sequence DESC LIMIT 1").fetchone()
    sequence, previous_hash = (0, genesis(binding_sha256)) if previous is None else (previous[0] + 1, previous[1])
    values = [sequence, snapshot, previous_hash]
    connection.execute("INSERT INTO history VALUES (?,?,?,?)", (*values, hashlib.sha256(canonical(values)).hexdigest()))


def validate_history(connection: sqlite3.Connection, binding_sha256: str) -> list[ReservationRow]:
    previous_hash = genesis(binding_sha256)
    projected: dict[str, ReservationRow] = {}
    history: list[ReservationRow] = []
    requests: set[tuple[str, str]] = set()
    attempts: set[str] = set()
    parser = TypeAdapter(ReservationRow)
    for expected, (sequence, snapshot, previous, digest) in enumerate(connection.execute(
        "SELECT sequence,snapshot,previous_hash,event_hash FROM history ORDER BY sequence")):
        if (sequence != expected or previous != previous_hash
            or digest != hashlib.sha256(canonical([sequence, snapshot, previous])).hexdigest()):
            raise BudgetError("BUDGET_HISTORY_INTEGRITY_INVALID")
        row = parser.validate_json(snapshot, strict=True)
        prior = projected.get(row.reservation_id)
        if prior is None:
            request = (row.run_id, row.request_id)
            if row.state != "reserved" or request in requests or row.attempt_id in attempts:
                raise BudgetError("BUDGET_HISTORY_MEMBERSHIP_INVALID")
            requests.add(request)
            attempts.add(row.attempt_id)
        elif (row[:5] != prior[:5] or row.known < prior.known
              or (row.state == "intent" and (prior.state != "reserved" or row[6:] != prior[6:]))
              or (row.state != "intent" and (row.state not in {"ambiguous", "charge", "no_charge", "no_request"}
                  or prior.state not in {"reserved", "intent", "ambiguous"} or row.proof is None))):
            raise BudgetError("BUDGET_HISTORY_TRANSITION_INVALID")
        projected[row.reservation_id] = row
        history.append(row)
        previous_hash = digest
    observed = {row[0]: ReservationRow(*row) for row in connection.execute("SELECT * FROM reservations")}
    if observed != projected:
        raise BudgetError("BUDGET_HISTORY_PROJECTION_MISMATCH")
    return history
