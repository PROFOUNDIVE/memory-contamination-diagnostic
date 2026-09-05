from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from pydantic import JsonValue

from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_cost_models import ProviderCostEvidence, canonical_bytes, digest
from .phase13_v3_terminal_models import (
    AmbiguousAttemptV3, EvidenceState, EventV3, LedgerBindingV3,
    NoRequestV3, OverflowV3, TerminalEvidenceError, advance, parse_event,
)


@contextmanager
def connection_to(path: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True)) as connection:
        connection.execute("PRAGMA synchronous = FULL")
        with connection:
            yield connection


@dataclass(frozen=True, slots=True)
class TerminalLedgerV3:
    """Append-only evidence seam; production entrypoint/path policy is a separate gate."""

    path: Path
    binding: LedgerBindingV3

    @classmethod
    def create(cls, path: Path, binding: Mapping[str, JsonValue]) -> TerminalLedgerV3:
        parsed = LedgerBindingV3.model_validate_json(json.dumps(dict(binding)))
        resolved = path.absolute()
        with os.fdopen(os.open(resolved, os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_RDWR, 0o600), "wb"):
            pass
        ledger = cls(resolved, parsed)
        with connection_to(resolved) as connection:
            connection.execute("CREATE TABLE metadata (raw BLOB NOT NULL)")
            connection.execute("CREATE TABLE events (sequence INTEGER PRIMARY KEY, raw BLOB NOT NULL, event_hash TEXT NOT NULL)")
            connection.execute("INSERT INTO metadata VALUES (?)", (canonical_bytes(parsed),))
        ledger._sync()
        return ledger

    @classmethod
    def open(cls, path: Path, binding: LedgerBindingV3) -> TerminalLedgerV3:
        ledger = cls(path.absolute(), binding)
        with connection_to(ledger.path) as connection:
            ledger._replay(connection)
        ledger._sync()
        return ledger

    def rows(self) -> tuple[bytes, ...]:
        with connection_to(self.path) as connection:
            self._replay(connection)
            return tuple(bytes(row[0]) for row in connection.execute("SELECT raw FROM events ORDER BY sequence"))

    def state(self, unit_id: str) -> EvidenceState:
        with connection_to(self.path) as connection:
            states = self._replay(connection)
        try:
            return states[unit_id]
        except KeyError as error:
            raise TerminalEvidenceError() from error

    def append(self, supplied: Mapping[str, JsonValue]) -> None:
        event = parse_event(json.dumps(dict(supplied), allow_nan=False).encode())
        raw = canonical_bytes(event)
        with connection_to(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            states = self._replay(connection)
            if event.unit_id not in states:
                raise TerminalEvidenceError()
            duplicate = connection.execute("SELECT 1 FROM events WHERE raw = ?", (raw,)).fetchone()
            if duplicate is None:
                advance(states[event.unit_id], event)
                self._dispatch_gate(states, event)
                connection.execute("INSERT INTO events(raw, event_hash) VALUES (?, ?)", (raw, digest(event)))
        self._sync()

    def recover(self, unit_id: str, proof_hash: str) -> None:
        state = self.state(unit_id)
        common = {"schema_version": "phase13_main_reconciliation_v3", "unit_id": unit_id,
                  "revision": state.revision + 1, "previous_hash": state.event_hash,
                  "compiled": None if state.compiled is None else state.compiled.model_dump(mode="json")}
        match state.kind:
            case "DISPATCH_INTENT_PERSISTED" | "REQUEST_COMPILED":
                event: EventV3 = NoRequestV3.model_validate({**common, "kind": "NO_REQUEST", "proof_hash": proof_hash})
            case "ATTEMPT_STARTED":
                event = AmbiguousAttemptV3.model_validate({**common, "kind": "AMBIGUOUS_ATTEMPT", "proof_hash": proof_hash,
                    "failure_code": "MAIN_AMBIGUOUS_ATTEMPT", "transport_attempts": 1,
                    "cost": {}, "realized_cost_krw": None})
            case "INPUT_ENVELOPE_OVERFLOW":
                event = OverflowV3.model_validate({**common, "kind": "TERMINAL_TECHNICAL_MISSING",
                    "schema_version": "phase13_main_dispatch_evidence_v3", "transport_attempts": 0,
                    "failure_code": "MAIN_INPUT_ENVELOPE_EXCEEDED", "realized_cost_krw": 0})
            case "PENDING" | "TERMINAL_TECHNICAL_MISSING" | "COMPLETED" | "ATTEMPTED_PROVIDER_FAILURE" | "AMBIGUOUS_ATTEMPT":
                raise TerminalEvidenceError()
            case unreachable:
                assert_never(unreachable)
        self.append(event.model_dump(mode="json"))

    def reconcile_cost(self, unit_id: str, supplied: Mapping[str, JsonValue], proof_hash: str) -> None:
        cost = ProviderCostEvidence.model_validate_json(json.dumps(dict(supplied)))
        realized = reconcile_actual(cost).realized_krw
        state = self.state(unit_id)
        self.append({
            "schema_version": "phase13_main_reconciliation_v3", "kind": "COST_RECONCILED",
            "unit_id": unit_id, "revision": state.revision + 1, "previous_hash": state.event_hash,
            "compiled": None if state.compiled is None else state.compiled.model_dump(mode="json"),
            "transport_attempts": 1, "cost": cost.model_dump(mode="json"),
            "realized_cost_krw": realized, "proof_hash": proof_hash,
        })

    def _replay(self, connection: sqlite3.Connection) -> dict[str, EvidenceState]:
        metadata = connection.execute("SELECT raw FROM metadata").fetchall()
        if metadata != [(canonical_bytes(self.binding),)]:
            raise TerminalEvidenceError()
        states = {unit: EvidenceState("PENDING", 0, hashlib.sha256(
            f"{digest(self.binding)}:{unit}".encode()).hexdigest()) for unit in self.binding.unit_ids}
        for row in connection.execute("SELECT raw, event_hash FROM events ORDER BY sequence"):
            raw = bytes(row[0])
            event = parse_event(raw)
            if raw != canonical_bytes(event) or row[1] != digest(event) or event.unit_id not in states:
                raise TerminalEvidenceError()
            self._dispatch_gate(states, event)
            states[event.unit_id] = advance(states[event.unit_id], event)
        return states

    @staticmethod
    def _dispatch_gate(states: Mapping[str, EvidenceState], event: EventV3) -> None:
        match event.kind:
            case "DISPATCH_INTENT":
                for state in states.values():
                    if state.attempted_cost is not None:
                        reconcile_actual(state.attempted_cost)
                if any(state.kind in ("DISPATCH_INTENT_PERSISTED", "REQUEST_COMPILED",
                                      "ATTEMPT_STARTED", "INPUT_ENVELOPE_OVERFLOW")
                       for state in states.values()):
                    raise TerminalEvidenceError()
            case "REQUEST_COMPILED" | "ATTEMPT_STARTED" | "INPUT_ENVELOPE_OVERFLOW" | "TERMINAL_TECHNICAL_MISSING" | "NO_REQUEST" | "COMPLETED" | "ATTEMPTED_PROVIDER_FAILURE" | "AMBIGUOUS_ATTEMPT" | "COST_RECONCILED":
                return
            case unreachable:
                assert_never(unreachable)

    def _sync(self) -> None:
        with self.path.open("rb") as database:
            os.fsync(database.fileno())
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
