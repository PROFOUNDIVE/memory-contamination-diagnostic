from __future__ import annotations

# noqa: SIZE_OK — append-only writes and incremental replay share one transaction invariant.
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import assert_never

from pydantic import JsonValue

from .phase13_authority_files import read_regular_nofollow
from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_cost_models import CostError, ProviderCostEvidence, canonical_bytes, digest
from .phase13_v3_entrypoint_paths import PrivateLedger
from .phase13_v3_terminal_models import (
    AmbiguousAttemptV3,
    EventV3,
    EvidenceState,
    LedgerBindingV3,
    NoRequestV3,
    OverflowV3,
    TerminalEvidenceError,
    advance,
    parse_event,
)


@contextmanager
def connection_to(path: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True)) as connection:
        connection.execute("PRAGMA synchronous = FULL")
        with connection:
            yield connection


@dataclass(slots=True)
class TerminalLedgerV3:
    """Append-only evidence seam with an incrementally verified session snapshot."""

    path: Path
    binding: LedgerBindingV3
    guard: PrivateLedger | None = None
    _states: dict[str, EvidenceState] = field(default_factory=dict, init=False, repr=False)
    _event_count: int = field(default=0, init=False, repr=False)
    _last_sequence: int = field(default=0, init=False, repr=False)
    _last_event_hash: str | None = field(default=None, init=False, repr=False)
    _in_flight: int = field(default=0, init=False, repr=False)
    _unknown_costs: dict[str, ProviderCostEvidence] = field(default_factory=dict, init=False, repr=False)
    _realized_costs: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _observer: sqlite3.Connection | None = field(default=None, init=False, repr=False)
    _data_version: int = field(default=0, init=False, repr=False)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        manager = connection_to(self.path) if self.guard is None else self.guard.connect()
        with manager as connection:
            yield connection

    @classmethod
    def create_guarded(cls, guard: PrivateLedger, binding: Mapping[str, JsonValue]) -> TerminalLedgerV3:
        parsed = LedgerBindingV3.model_validate_json(json.dumps(dict(binding)))
        ledger = cls(guard.directory / guard.path.name, parsed, guard)
        with ledger.connection() as connection:
            connection.execute("CREATE TABLE metadata (raw BLOB NOT NULL)")
            connection.execute("CREATE TABLE events (sequence INTEGER PRIMARY KEY, raw BLOB NOT NULL, event_hash TEXT NOT NULL)")
            connection.execute("INSERT INTO metadata VALUES (?)", (canonical_bytes(parsed),))
            ledger._replay(connection)
        ledger._sync()
        ledger._start_observer()
        return ledger

    @classmethod
    def open_guarded(cls, guard: PrivateLedger, binding: LedgerBindingV3) -> TerminalLedgerV3:
        ledger = cls(guard.directory / guard.path.name, binding, guard)
        with ledger.connection() as connection:
            ledger._replay(connection)
        ledger._start_observer()
        return ledger

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
            ledger._replay(connection)
        ledger._sync()
        ledger._start_observer()
        return ledger

    @classmethod
    def open(cls, path: Path, binding: LedgerBindingV3) -> TerminalLedgerV3:
        ledger = cls(path.absolute(), binding)
        with connection_to(ledger.path) as connection:
            ledger._replay(connection)
        ledger._sync()
        ledger._start_observer()
        return ledger

    def close(self) -> None:
        if self._observer is not None:
            self._observer.close()
            self._observer = None

    def generation(self) -> int:
        with self.connection() as connection:
            self._snapshot(connection)
        return self._event_count

    def rows(self) -> tuple[bytes, ...]:
        with self.connection() as connection:
            self._snapshot(connection)
            return tuple(bytes(row[0]) for row in connection.execute("SELECT raw FROM events ORDER BY sequence"))

    def state(self, unit_id: str) -> EvidenceState:
        with self.connection() as connection:
            states = self._snapshot(connection)
        try:
            return states[unit_id]
        except KeyError as error:
            raise TerminalEvidenceError() from error

    def states(self) -> dict[str, EvidenceState]:
        with self.connection() as connection:
            return dict(self._snapshot(connection))

    def read_record(self, name: str) -> bytes:
        return read_regular_nofollow(self.path.parent / name) if self.guard is None else self.guard.read_record(name)

    def require_known_costs(self) -> None:
        with self.connection() as connection:
            self._snapshot(connection)
        if self._unknown_costs:
            reconcile_actual(next(iter(self._unknown_costs.values())))

    def realized_cost_krw(self) -> int:
        self.require_known_costs()
        return sum(self._realized_costs.values())

    def append(self, supplied: Mapping[str, JsonValue]) -> None:
        event = parse_event(json.dumps(dict(supplied), allow_nan=False).encode())
        raw = canonical_bytes(event)
        next_state: EvidenceState | None = None
        sequence = 0
        with self.connection() as connection:
            if not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            states = self._snapshot(connection)
            if event.unit_id not in states:
                raise TerminalEvidenceError()
            duplicate = connection.execute("SELECT 1 FROM events WHERE raw = ?", (raw,)).fetchone()
            if duplicate is None:
                next_state = advance(states[event.unit_id], event)
                self._dispatch_gate(states, event)
                cursor = connection.execute("INSERT INTO events(raw, event_hash) VALUES (?, ?)", (raw, digest(event)))
                rowid = cursor.lastrowid
                if rowid is None:
                    raise TerminalEvidenceError()
                sequence = rowid
        if next_state is not None:
            self._replace_state(event.unit_id, next_state)
            self._event_count += 1
            self._last_sequence = sequence
            self._last_event_hash = next_state.event_hash
        self._sync()
        self._data_version = self._observed_version()

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
                    "failure_code": "MAIN_AMBIGUOUS_ATTEMPT", "transport_attempts": (state.attempt_index or 0) + 1,
                    "cost": {}, "realized_cost_krw": None})
            case "INPUT_ENVELOPE_OVERFLOW":
                event = OverflowV3.model_validate({**common, "kind": "TERMINAL_TECHNICAL_MISSING",
                    "schema_version": "phase13_main_dispatch_evidence_v3", "transport_attempts": 0,
                    "failure_code": "MAIN_INPUT_ENVELOPE_EXCEEDED", "realized_cost_krw": 0})
            case "PENDING" | "RETRYABLE_ATTEMPT_FAILURE" | "TERMINAL_TECHNICAL_MISSING" | "COMPLETED" | "ATTEMPTED_PROVIDER_FAILURE" | "AMBIGUOUS_ATTEMPT":
                raise TerminalEvidenceError()
            case unreachable:
                assert_never(unreachable)
        self.append(event.model_dump(mode="json"))

    def reconcile_cost(self, unit_id: str, supplied: Mapping[str, JsonValue], proof_hash: str,
                       *, attempt_index: int = 0) -> None:
        cost = ProviderCostEvidence.model_validate_json(json.dumps(dict(supplied)))
        realized = reconcile_actual(cost).realized_krw
        state = self.state(unit_id)
        self.append({
            "schema_version": "phase13_main_reconciliation_v3", "kind": "COST_RECONCILED",
            "unit_id": unit_id, "revision": state.revision + 1, "previous_hash": state.event_hash,
            "compiled": None if state.compiled is None else state.compiled.model_dump(mode="json"),
            "transport_attempts": len(state.attempt_costs), "attempt_index": attempt_index,
            "cost": cost.model_dump(mode="json"),
            "realized_cost_krw": realized, "proof_hash": proof_hash,
        })

    def _replay(self, connection: sqlite3.Connection) -> dict[str, EvidenceState]:
        metadata = connection.execute("SELECT raw FROM metadata").fetchall()
        if metadata != [(canonical_bytes(self.binding),)]:
            raise TerminalEvidenceError()
        binding_hash = digest(self.binding)
        states = {unit: EvidenceState("PENDING", 0, hashlib.sha256(
            f"{binding_hash}:{unit}".encode()).hexdigest()) for unit in self.binding.unit_ids}
        self._states = states
        self._event_count = self._last_sequence = self._in_flight = 0
        self._last_event_hash = None
        self._unknown_costs.clear()
        self._realized_costs.clear()
        for sequence, raw, event_hash in connection.execute(
            "SELECT sequence, raw, event_hash FROM events ORDER BY sequence"
        ):
            self._apply_event(int(sequence), bytes(raw), str(event_hash))
        return self._states

    def _snapshot(self, connection: sqlite3.Connection) -> dict[str, EvidenceState]:
        if not self._states:
            return self._replay(connection)
        count, last_sequence = connection.execute(
            "SELECT COUNT(*), COALESCE(MAX(sequence), 0) FROM events"
        ).fetchone()
        observed_version = self._observed_version()
        if observed_version != self._data_version:
            if int(count) < self._event_count:
                raise TerminalEvidenceError()
            states = self._replay(connection)
            self._data_version = observed_version
            return states
        tail = connection.execute(
            "SELECT event_hash FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        last_hash = None if tail is None else str(tail[0])
        if int(count) == self._event_count and last_hash == self._last_event_hash:
            return self._states
        if int(count) < self._event_count or int(last_sequence) <= self._last_sequence:
            return self._replay(connection)
        rows = connection.execute(
            "SELECT sequence, raw, event_hash FROM events WHERE sequence > ? ORDER BY sequence",
            (self._last_sequence,),
        ).fetchall()
        if len(rows) != int(count) - self._event_count:
            return self._replay(connection)
        for sequence, raw, event_hash in rows:
            self._apply_event(int(sequence), bytes(raw), str(event_hash))
        return self._states

    def _start_observer(self) -> None:
        self._observer = sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True)
        self._data_version = self._observed_version()

    def _observed_version(self) -> int:
        if self._observer is None:
            raise TerminalEvidenceError()
        return int(self._observer.execute("PRAGMA data_version").fetchone()[0])

    def _apply_event(self, sequence: int, raw: bytes, event_hash: str) -> None:
        event = parse_event(raw)
        if raw != canonical_bytes(event) or event_hash != digest(event) or event.unit_id not in self._states:
            raise TerminalEvidenceError()
        self._dispatch_gate(self._states, event)
        self._replace_state(event.unit_id, advance(self._states[event.unit_id], event))
        self._event_count += 1
        self._last_sequence = sequence
        self._last_event_hash = event_hash

    def _replace_state(self, unit_id: str, state: EvidenceState) -> None:
        previous = self._states[unit_id]
        unresolved = {"DISPATCH_INTENT_PERSISTED", "REQUEST_COMPILED", "ATTEMPT_STARTED", "RETRYABLE_ATTEMPT_FAILURE", "INPUT_ENVELOPE_OVERFLOW"}
        self._in_flight += int(state.kind in unresolved) - int(previous.kind in unresolved)
        for index in range(len(previous.attempt_costs)):
            self._unknown_costs.pop(f"{unit_id}:{index}", None)
            self._realized_costs.pop(f"{unit_id}:{index}", None)
        for index, attempt in enumerate(state.attempt_costs):
            try:
                realized = reconcile_actual(attempt).realized_krw
            except CostError as error:
                if error.code != "MAIN_TERMINAL_COST_UNKNOWN":
                    raise
                self._unknown_costs[f"{unit_id}:{index}"] = attempt
            else:
                self._realized_costs[f"{unit_id}:{index}"] = realized
        self._states[unit_id] = state

    def _dispatch_gate(self, states: Mapping[str, EvidenceState], event: EventV3) -> None:
        match event.kind:
            case "DISPATCH_INTENT":
                if self._unknown_costs:
                    reconcile_actual(next(iter(self._unknown_costs.values())))
                if self._in_flight:
                    raise TerminalEvidenceError()
            case "REQUEST_COMPILED" | "ATTEMPT_STARTED" | "RETRYABLE_ATTEMPT_FAILURE" | "INPUT_ENVELOPE_OVERFLOW" | "TERMINAL_TECHNICAL_MISSING" | "NO_REQUEST" | "COMPLETED" | "ATTEMPTED_PROVIDER_FAILURE" | "AMBIGUOUS_ATTEMPT" | "COST_RECONCILED":
                return
            case unreachable:
                assert_never(unreachable)

    def _sync(self) -> None:
        if self.guard is not None:
            self.guard.sync()
            return
        with self.path.open("rb") as database:
            os.fsync(database.fileno())
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
