from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, assert_never

from memcontam.readiness.phase13_main_runner_models import (
    DispatchCompleted,
    DispatchTechnicalFailure,
    ExecutionUnit,
    InFlightContext,
    InFlightEvidence,
    MainRunBinding,
    MainRunError,
    MainRunStatus,
)
from memcontam.readiness.phase13_main_runner_store import (
    canonical,
    connect,
    create_ledger_file,
    expected_metadata,
    ledger_genesis,
    session_state,
    transition,
    validate_integrity,
)


EventKind = Literal["INTENT", "COMPLETED", "TERMINAL", "NO_REQUEST", "PAUSED"]


class MainRunLedger:
    def __init__(self, path: Path, units: tuple[ExecutionUnit, ...]) -> None:
        self.path = path
        self._units = units

    @classmethod
    def create(
        cls,
        path: Path,
        binding: MainRunBinding,
        units: tuple[ExecutionUnit, ...],
    ) -> MainRunLedger:
        create_ledger_file(path, binding, units)
        return cls(path, units)

    @classmethod
    def open(
        cls,
        path: Path,
        binding: MainRunBinding,
        units: tuple[ExecutionUnit, ...],
    ) -> MainRunLedger:
        if not path.is_file():
            raise MainRunError("MAIN_RUN_LEDGER_NOT_FOUND")
        ledger = cls(path, units)
        with connect(path) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            expected = expected_metadata(binding, units)
            if metadata != expected:
                raise MainRunError("MAIN_RUN_BINDING_MISMATCH")
            validate_integrity(connection, units)
        return ledger

    def next_pending(self) -> ExecutionUnit | None:
        with connect(self.path) as connection:
            validate_integrity(connection, self._units)
            row = connection.execute(
                "SELECT sequence FROM execution_units WHERE state = 'PENDING' ORDER BY sequence LIMIT 1"
            ).fetchone()
        return None if row is None else self._units[int(row[0])]

    def persist_dispatch_intent(self, unit_id: str) -> None:
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            states, _, _ = validate_integrity(connection, self._units)
            self._require_claimable(connection, unit_id, states)
            self._append_locked(connection, unit_id, "INTENT")

    def claim_dispatch(
        self,
        unit_id: str,
        projected_cost_krw: int,
        tranche_ceiling_krw: int,
    ) -> bool:
        if (
            type(projected_cost_krw) is not int
            or projected_cost_krw < 0
            or type(tranche_ceiling_krw) is not int
            or tranche_ceiling_krw < 0
        ):
            raise MainRunError("MAIN_RUN_COST_INVALID")
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            states, _, realized_cost = validate_integrity(connection, self._units)
            self._require_claimable(connection, unit_id, states)
            if realized_cost + projected_cost_krw > tranche_ceiling_krw:
                self._append_locked(
                    connection,
                    unit_id,
                    "PAUSED",
                    reason=str(projected_cost_krw),
                )
                return False
            self._append_locked(connection, unit_id, "INTENT")
            return True

    def persist_completed(self, unit_id: str, completed: DispatchCompleted) -> None:
        self._append(
            unit_id,
            "COMPLETED",
            evidence_sha256=completed.evidence_sha256,
            realized_cost_krw=completed.realized_cost_krw,
        )

    def persist_terminal_missing(
        self,
        unit_id: str,
        failure: DispatchTechnicalFailure,
    ) -> None:
        self._append(
            unit_id,
            "TERMINAL",
            evidence_sha256=failure.evidence_sha256,
            reason=failure.code,
            realized_cost_krw=failure.realized_cost_krw,
        )

    def persist_pause(self, unit_id: str, projected_cost_krw: int) -> None:
        self._append(unit_id, "PAUSED", reason=str(projected_cost_krw))

    def in_flight_context(self, unit_id: str) -> InFlightContext:
        with connect(self.path) as connection:
            validate_integrity(connection, self._units)
            row = connection.execute(
                "SELECT state FROM execution_units WHERE unit_id = ?", (unit_id,)
            ).fetchone()
            event = connection.execute(
                "SELECT event_hash FROM events WHERE unit_id = ? "
                "ORDER BY event_sequence DESC LIMIT 1",
                (unit_id,),
            ).fetchone()
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if row is None or row[0] != "DISPATCH_INTENT_PERSISTED" or event is None:
            raise MainRunError("MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID")
        return InFlightContext(
            unit_id,
            str(event[0]),
            metadata["package_sha256"],
            metadata["authorization_sha256"],
        )

    def reconcile(self, unit_id: str, evidence: InFlightEvidence) -> None:
        if evidence.context != self.in_flight_context(unit_id):
            raise MainRunError("MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID")
        match evidence.disposition:
            case "NO_PROVIDER_REQUEST":
                self._append(unit_id, "NO_REQUEST", evidence_sha256=evidence.evidence_sha256)
            case "COMPLETED":
                self._append(
                    unit_id,
                    "COMPLETED",
                    evidence_sha256=evidence.evidence_sha256,
                    realized_cost_krw=evidence.realized_cost_krw,
                )
            case "TERMINAL_FAILURE":
                self._append(
                    unit_id,
                    "TERMINAL",
                    evidence_sha256=evidence.evidence_sha256,
                    reason="RECONCILED_TERMINAL_TECHNICAL_MISSING",
                    realized_cost_krw=evidence.realized_cost_krw,
                )
            case "AMBIGUOUS":
                raise MainRunError("MAIN_RUN_IN_FLIGHT_AMBIGUOUS")
            case unreachable:
                assert_never(unreachable)

    def status(self) -> MainRunStatus:
        with connect(self.path) as connection:
            states, last_kind, realized_cost = validate_integrity(connection, self._units)
        counts = {state: states.count(state) for state in set(states)}
        in_flight = counts.get("DISPATCH_INTENT_PERSISTED", 0)
        completed = counts.get("COMPLETED", 0)
        terminal = counts.get("TERMINAL_TECHNICAL_MISSING", 0)
        pending = counts.get("PENDING", 0)
        current_session_state = session_state(last_kind, pending, in_flight, completed, terminal)
        return MainRunStatus(
            current_session_state,
            len(states),
            pending,
            in_flight,
            completed,
            terminal,
            realized_cost,
        )

    def _append(
        self,
        unit_id: str,
        kind: EventKind,
        evidence_sha256: str | None = None,
        reason: str | None = None,
        realized_cost_krw: int = 0,
    ) -> None:
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            validate_integrity(connection, self._units)
            self._append_locked(
                connection,
                unit_id,
                kind,
                evidence_sha256,
                reason,
                realized_cost_krw,
            )

    def _append_locked(
        self,
        connection,
        unit_id: str,
        kind: EventKind,
        evidence_sha256: str | None = None,
        reason: str | None = None,
        realized_cost_krw: int = 0,
    ) -> None:
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
        values = [
            sequence,
            unit_id,
            kind,
            evidence_sha256,
            reason,
            realized_cost_krw,
            previous_hash,
        ]
        event_hash = hashlib.sha256(canonical(values)).hexdigest()
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (*values, event_hash),
        )
        connection.execute(
            "UPDATE execution_units SET state = ? WHERE unit_id = ?", (target, unit_id)
        )

    @staticmethod
    def _require_claimable(connection, unit_id: str, states: list[str]) -> None:
        if "DISPATCH_INTENT_PERSISTED" in states:
            raise MainRunError("MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED")
        pending = connection.execute(
            "SELECT unit_id FROM execution_units WHERE state = 'PENDING' "
            "ORDER BY sequence LIMIT 1"
        ).fetchone()
        if pending is None or pending[0] != unit_id:
            raise MainRunError("MAIN_RUN_DISPATCH_ORDER_INVALID")
