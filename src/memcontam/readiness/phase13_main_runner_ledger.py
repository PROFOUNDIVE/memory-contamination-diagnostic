from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Literal, assert_never

from memcontam.readiness.phase13_main_live_evidence import (
    MainEvidenceValidationError,
    load_durable_reconciliation_evidence,
    load_durable_unit_evidence,
)
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


EventKind = Literal[
    "INTENT",
    "COMPLETED",
    "TERMINAL",
    "DEPENDENCY_TERMINAL",
    "NO_REQUEST",
    "PAUSED",
]


class MainRunLedger:
    def __init__(
        self,
        path: Path,
        units: tuple[ExecutionUnit, ...],
        core_authorization_gate_krw: int,
    ) -> None:
        self.path = path
        self._units = units
        self._core_authorization_gate_krw = core_authorization_gate_krw

    @classmethod
    def create(
        cls,
        path: Path,
        binding: MainRunBinding,
        units: tuple[ExecutionUnit, ...],
    ) -> MainRunLedger:
        create_ledger_file(path, binding, units)
        return cls(path, units, binding.core_authorization_gate_krw)

    @classmethod
    def open(
        cls,
        path: Path,
        binding: MainRunBinding,
        units: tuple[ExecutionUnit, ...],
    ) -> MainRunLedger:
        if not path.is_file():
            raise MainRunError("MAIN_RUN_LEDGER_NOT_FOUND")
        ledger = cls(path, units, binding.core_authorization_gate_krw)
        with connect(path) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            expected = expected_metadata(binding, units)
            if metadata != expected:
                raise MainRunError("MAIN_RUN_BINDING_MISMATCH")
            ledger._validate_integrity(connection)
        return ledger

    def next_pending(self) -> ExecutionUnit | None:
        with connect(self.path) as connection:
            self._validate_integrity(connection)
            row = connection.execute(
                "SELECT sequence FROM execution_units WHERE state = 'PENDING' ORDER BY sequence LIMIT 1"
            ).fetchone()
        return None if row is None else self._units[int(row[0])]

    def persist_dispatch_intent(self, unit_id: str) -> None:
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            states, _, _ = self._validate_integrity(connection)
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
            or tranche_ceiling_krw > self._core_authorization_gate_krw
        ):
            raise MainRunError("MAIN_RUN_COST_INVALID")
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            states, _, realized_cost = self._validate_integrity(connection)
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
        unit = next((row for row in self._units if row.unit_id == unit_id), None)
        if unit is None:
            raise MainRunError("MAIN_RUN_UNIT_UNKNOWN")
        path = self.path.parent / "units" / f"{unit.sequence:06d}-{unit.unit_id}.json"
        try:
            load_durable_unit_evidence(
                path,
                unit,
                completed.evidence_sha256,
                completed.realized_cost_krw,
            )
        except MainEvidenceValidationError as error:
            raise MainRunError("MAIN_RUN_COMPLETION_EVIDENCE_INVALID") from error
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
        context = self.in_flight_context(unit_id)
        evidence = InFlightEvidence.terminal_failure(
            context,
            failure.evidence_sha256,
            failure.code,
            failure.realized_cost_krw,
        )
        self._validate_reconciliation_evidence(evidence)
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_integrity(connection)
            self._append_terminal_locked(
                connection,
                unit_id,
                failure.evidence_sha256,
                failure.code,
                failure.realized_cost_krw,
            )

    def persist_pause(self, unit_id: str, projected_cost_krw: int) -> None:
        self._append(unit_id, "PAUSED", reason=str(projected_cost_krw))

    def in_flight_context(self, unit_id: str) -> InFlightContext:
        with connect(self.path) as connection:
            self._validate_integrity(connection)
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

    def completed_evidence_sha256(self, unit_id: str) -> str:
        with connect(self.path) as connection:
            self._validate_integrity(connection)
            row = connection.execute(
                "SELECT state FROM execution_units WHERE unit_id = ?", (unit_id,)
            ).fetchone()
            event = connection.execute(
                "SELECT evidence_sha256 FROM events WHERE unit_id = ? AND kind = 'COMPLETED' "
                "ORDER BY event_sequence DESC LIMIT 1",
                (unit_id,),
            ).fetchone()
        if row is None or row[0] != "COMPLETED" or event is None or event[0] is None:
            raise MainRunError("MAIN_RUN_COMPLETION_EVIDENCE_INVALID")
        return str(event[0])

    def reconcile(self, unit_id: str, evidence: InFlightEvidence) -> None:
        if evidence.context != self.in_flight_context(unit_id):
            raise MainRunError("MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID")
        match evidence.disposition:
            case "NO_PROVIDER_REQUEST":
                self._validate_reconciliation_evidence(evidence)
                self._append(unit_id, "NO_REQUEST", evidence_sha256=evidence.evidence_sha256)
            case "COMPLETED":
                unit = next((row for row in self._units if row.unit_id == unit_id), None)
                if unit is None:
                    raise MainRunError("MAIN_RUN_UNIT_UNKNOWN")
                path = self.path.parent / "units" / f"{unit.sequence:06d}-{unit.unit_id}.json"
                try:
                    load_durable_unit_evidence(
                        path,
                        unit,
                        evidence.evidence_sha256,
                        evidence.realized_cost_krw,
                    )
                except MainEvidenceValidationError as error:
                    raise MainRunError("MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID") from error
                self._append(
                    unit_id,
                    "COMPLETED",
                    evidence_sha256=evidence.evidence_sha256,
                    realized_cost_krw=evidence.realized_cost_krw,
                )
            case "TERMINAL_FAILURE":
                self._validate_reconciliation_evidence(evidence)
                if evidence.failure_code is None:
                    raise MainRunError("MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID")
                with connect(self.path) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._validate_integrity(connection)
                    self._append_terminal_locked(
                        connection,
                        unit_id,
                        evidence.evidence_sha256,
                        evidence.failure_code,
                        evidence.realized_cost_krw,
                    )
            case "AMBIGUOUS":
                raise MainRunError("MAIN_RUN_IN_FLIGHT_AMBIGUOUS")
            case unreachable:
                assert_never(unreachable)

    def _validate_reconciliation_evidence(self, evidence: InFlightEvidence) -> None:
        path = self.path.parent / "reconciliation" / f"{evidence.context.intent_event_hash}.json"
        try:
            load_durable_reconciliation_evidence(path, evidence)
        except MainEvidenceValidationError as error:
            raise MainRunError("MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID") from error

    def status(self) -> MainRunStatus:
        with connect(self.path) as connection:
            states, last_kind, realized_cost = self._validate_integrity(connection)
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
            self._validate_integrity(connection)
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

    def _validate_integrity(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[list[str], str | None, int]:
        result = validate_integrity(connection, self._units)
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        intents: dict[str, str] = {}
        units = {unit.unit_id: unit for unit in self._units}
        events = connection.execute(
            "SELECT unit_id, kind, evidence_sha256, reason, realized_cost_krw, event_hash "
            "FROM events ORDER BY event_sequence"
        ).fetchall()
        for unit_id, kind, evidence_sha256, reason, realized_cost_krw, event_hash in events:
            unit_id = str(unit_id)
            match str(kind):
                case "INTENT":
                    intents[unit_id] = str(event_hash)
                case "COMPLETED":
                    unit = units.get(unit_id)
                    if unit is None or evidence_sha256 is None:
                        raise MainRunError("MAIN_RUN_COMPLETION_EVIDENCE_INVALID")
                    path = self.path.parent / "units" / f"{unit.sequence:06d}-{unit.unit_id}.json"
                    try:
                        load_durable_unit_evidence(
                            path,
                            unit,
                            str(evidence_sha256),
                            int(realized_cost_krw),
                        )
                    except MainEvidenceValidationError as error:
                        raise MainRunError("MAIN_RUN_COMPLETION_EVIDENCE_INVALID") from error
                case "NO_REQUEST" | "TERMINAL" as disposition:
                    intent_event_hash = intents.get(unit_id)
                    if intent_event_hash is None or evidence_sha256 is None:
                        raise MainRunError("MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID")
                    context = InFlightContext(
                        unit_id,
                        intent_event_hash,
                        metadata["package_sha256"],
                        metadata["authorization_sha256"],
                    )
                    evidence = (
                        InFlightEvidence.no_provider_request(context, str(evidence_sha256))
                        if disposition == "NO_REQUEST"
                        else InFlightEvidence.terminal_failure(
                            context,
                            str(evidence_sha256),
                            str(reason),
                            int(realized_cost_krw),
                        )
                    )
                    self._validate_reconciliation_evidence(evidence)
                case "PAUSED" | "DEPENDENCY_TERMINAL":
                    continue
                case _:
                    raise MainRunError("MAIN_RUN_LEDGER_INTEGRITY_INVALID") from None
        return result

    def _append_terminal_locked(
        self,
        connection,
        unit_id: str,
        evidence_sha256: str,
        reason: str,
        realized_cost_krw: int,
    ) -> None:
        self._append_locked(
            connection,
            unit_id,
            "TERMINAL",
            evidence_sha256,
            reason,
            realized_cost_krw,
        )
        unit = next((row for row in self._units if row.unit_id == unit_id), None)
        if unit is not None and unit.kind == "CLEAN_PREFIX":
            for dependent in self._units:
                if dependent.prefix_unit_id == unit_id:
                    self._append_locked(
                        connection,
                        dependent.unit_id,
                        "DEPENDENCY_TERMINAL",
                        evidence_sha256,
                        "SHARED_PREFIX_TERMINAL_TECHNICAL_MISSING",
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
