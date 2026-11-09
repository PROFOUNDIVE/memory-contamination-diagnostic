"""One financial authority; no provider transport or scientific launch permission."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .phase13_live_diagnostic import LiveDiagnosticLedger
from .phase13_live_diagnostic_models import DiagnosticRequestIntent
from .phase13_main_live_dispatch import persist_evidence_bytes
from .phase13_main_runner_store import connect
from .phase13_shared_budget_authority import BudgetAuthority
from memcontam.readiness.phase13_shared_budget_history import append_history, create_history, validate_history
from .phase13_shared_budget_models import (
    CAP, BudgetError, BudgetOperation, BudgetStatus, FilePin,
)


class SharedProviderBudget:
    def __init__(self, binding_path: Path) -> None:
        self.binding_path = binding_path
        self.authority = BudgetAuthority.load(binding_path)
        self.binding = self.authority.binding
        self.path = Path(self.binding.ledger_path)

    @classmethod
    def create(cls, binding_path: Path) -> SharedProviderBudget:
        budget = cls(binding_path)
        if not budget.authority.reconciliation.complete or budget.binding.proof_public_key is None:
            raise BudgetError("BUDGET_RECONCILIATION_INCOMPLETE")
        if budget.path.exists():
            raise BudgetError("BUDGET_AUTHORITY_ALREADY_EXISTS")
        persist_evidence_bytes(budget.path.with_suffix(".created"), budget.binding.sha256().encode(),
                               "BUDGET_AUTHORIZATION_ALREADY_USED")
        temporary = budget.path.with_name(f".{uuid4().hex}.sqlite3")
        try:
            with connect(temporary) as connection:
                connection.executescript("""
                    CREATE TABLE metadata (binding TEXT NOT NULL);
                    CREATE TABLE reservations (
                        reservation_id TEXT PRIMARY KEY, operation TEXT NOT NULL,
                        run_id TEXT NOT NULL, request_id TEXT NOT NULL, attempt_id TEXT NOT NULL UNIQUE,
                        state TEXT NOT NULL, known INTEGER NOT NULL CHECK(known >= 0),
                        held INTEGER NOT NULL CHECK(held >= 0), proof TEXT,
                        UNIQUE(run_id, request_id));
                    CREATE TABLE proofs (sha256 TEXT PRIMARY KEY, pin TEXT NOT NULL);
                """)
                connection.execute("INSERT INTO metadata VALUES (?)", (budget.binding.model_dump_json(),))
                create_history(connection)
            persist_evidence_bytes(budget.path, temporary.read_bytes(), "BUDGET_AUTHORITY_ALREADY_EXISTS")
        finally:
            temporary.unlink(missing_ok=True)
            Path(f"{temporary}-wal").unlink(missing_ok=True)
            Path(f"{temporary}-shm").unlink(missing_ok=True)
        return budget

    @classmethod
    def open(cls, binding_path: Path) -> SharedProviderBudget:
        budget = cls(binding_path)
        budget.status()
        return budget

    def _validate(self, connection: sqlite3.Connection) -> None:
        authority = BudgetAuthority.load(self.binding_path)
        if (authority.binding != self.binding or not authority.reconciliation.complete
            or self.path.with_suffix(".created").read_text() != self.binding.sha256()
            or connection.execute("SELECT binding FROM metadata").fetchall()
                != [(self.binding.model_dump_json(),)]):
            raise BudgetError("BUDGET_AUTHORITY_CHANGED")
        for reservation, raw, run, request, attempt, state, known, held, pointer in validate_history(
            connection, self.binding.sha256()):
            operation = BudgetOperation.model_validate_json(raw)
            authority.check_operation(operation, current=False)
            if (run, request, attempt) != (operation.run_id, operation.request_id, operation.attempt_id):
                raise BudgetError("BUDGET_RESERVATION_CORRUPT")
            expected = (0, operation.bound_usd_micros)
            if pointer is not None:
                pin = FilePin.model_validate_json(pointer)
                proof = authority.proof(pin)
                if proof.intent != operation.intent(reservation) or state != proof.kind:
                    raise BudgetError("BUDGET_PROOF_INTENT_MISMATCH")
                expected = proof.amounts(operation.bound_usd_micros)
                if connection.execute("SELECT pin FROM proofs WHERE sha256=?", (pin.sha256,)).fetchall() != [(pointer,)]:
                    raise BudgetError("BUDGET_PROOF_MISSING")
            elif state not in {"reserved", "intent"}:
                raise BudgetError("BUDGET_STATE_INVALID")
            if (known, held) != expected:
                raise BudgetError("BUDGET_LIABILITY_CORRUPT")

    def _status(self, connection: sqlite3.Connection) -> BudgetStatus:
        reconciliation = self.authority.reconciliation
        prior_known = sum(row.known_usd_micros for row in reconciliation.operations)
        prior_held = sum(row.hold_usd_micros for row in reconciliation.operations)
        known, held, count = connection.execute(
            "SELECT COALESCE(SUM(known),0), COALESCE(SUM(held),0), COUNT(*) FROM reservations").fetchone()
        allowance = reconciliation.remaining_allowance_usd_micros
        if allowance is None:
            raise BudgetError("BUDGET_RECONCILIATION_INCOMPLETE")
        available = min(CAP - prior_known - prior_held - known - held, allowance - known - held)
        blockers = list(self.authority.pricing.blockers)
        if not self.authority.pricing.entries:
            blockers.append("BUDGET_PRICE_UNKNOWN")
        today = datetime.now(timezone.utc).date()
        if any(not price.as_of <= today <= price.valid_until for price in self.authority.pricing.entries):
            blockers.append("BUDGET_PRICE_STALE")
        if available < 0:
            blockers.append("BUDGET_CAP_EXCEEDED")
        breached = connection.execute(
            "SELECT state FROM reservations WHERE known > json_extract(operation, '$.bound_usd_micros')").fetchall()
        if breached:
            blockers.append("BUDGET_CONSERVATIVE_BOUND_BREACHED")
        unbounded = any(state == "ambiguous" for (state,) in breached)
        return BudgetStatus(paid_eligible=not blockers, known_usd_micros=prior_known + known,
            held_usd_micros=None if unbounded else prior_held + held,
            available_usd_micros=None if unbounded else max(0, available),
            reservations=count, blockers=tuple(blockers))

    def status(self) -> BudgetStatus:
        if not self.path.is_file():
            raise BudgetError("BUDGET_LEDGER_MISSING")
        with connect(self.path) as connection:
            connection.execute("BEGIN")
            self._validate(connection)
            return self._status(connection)

    def reserve(self, operation: BudgetOperation) -> DiagnosticRequestIntent:
        operation = BudgetOperation.model_validate_json(operation.model_dump_json())
        if not self.path.is_file():
            raise BudgetError("BUDGET_LEDGER_MISSING")
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate(connection)
            self.authority.check_operation(operation)
            status = self._status(connection)
            if connection.execute("SELECT 1 FROM reservations WHERE (run_id=? AND request_id=?) OR attempt_id=?",
                                  (operation.run_id, operation.request_id, operation.attempt_id)).fetchone():
                raise BudgetError("BUDGET_RESERVATION_REPLAY")
            if (not status.paid_eligible or status.available_usd_micros is None
                or operation.bound_usd_micros > status.available_usd_micros):
                raise BudgetError("BUDGET_CAP_OR_ELIGIBILITY_REFUSED")
            intent = operation.intent(uuid4().hex)
            connection.execute("INSERT INTO reservations VALUES (?,?,?,?,?,'reserved',0,?,NULL)",
                (intent.reservation_id, operation.model_dump_json(), operation.run_id, operation.request_id,
                 operation.attempt_id, operation.bound_usd_micros))
            append_history(connection, intent.reservation_id, self.binding.sha256())
        return intent

    def mark_intent(self, intent: DiagnosticRequestIntent) -> BudgetOperation:
        """Commit once before the executor claims intent; never redispatch on reopen."""
        if not self.path.is_file():
            raise BudgetError("BUDGET_LEDGER_MISSING")
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate(connection)
            row = connection.execute("SELECT operation,state FROM reservations WHERE reservation_id=?",
                                     (intent.reservation_id,)).fetchone()
            if row is None or row[1] != "reserved":
                raise BudgetError("BUDGET_INTENT_REPLAY_OR_UNKNOWN")
            operation = BudgetOperation.model_validate_json(row[0])
            if operation.intent(intent.reservation_id) != intent:
                raise BudgetError("BUDGET_INTENT_MISMATCH")
            self.authority.check_operation(operation)
            if not self._status(connection).paid_eligible:
                raise BudgetError("BUDGET_ELIGIBILITY_REFUSED")
            connection.execute("UPDATE reservations SET state='intent' WHERE reservation_id=?", (intent.reservation_id,))
            append_history(connection, intent.reservation_id, self.binding.sha256())
        return operation

    def claim_diagnostic(self, ledger: LiveDiagnosticLedger, intent: DiagnosticRequestIntent) -> None:
        operation = self.mark_intent(intent)
        unit = next((unit for unit in ledger.units if unit.unit_id == operation.unit_id), None)
        if unit is None or operation.execution_binding_sha256 != ledger.binding.sha256():
            raise BudgetError("BUDGET_EXECUTOR_BINDING_MISMATCH")
        observation = ledger.binding.observations[unit.sequence]
        request = next((row for row in observation.requests if row.request_id == intent.request_id), None)
        if (operation.run_id != observation.run_id or request is None
            or operation.operation != request.operation):
            raise BudgetError("BUDGET_EXECUTOR_REQUEST_MISMATCH")
        ledger.claim_request(unit.unit_id, intent)

    def reconcile(self, pin: FilePin) -> None:
        if not self.path.is_file():
            raise BudgetError("BUDGET_LEDGER_MISSING")
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate(connection)
            proof = self.authority.proof(pin)
            row = connection.execute("SELECT operation,state,known FROM reservations WHERE reservation_id=?",
                                     (proof.intent.reservation_id,)).fetchone()
            if row is None or row[1] not in {"reserved", "intent", "ambiguous"}:
                raise BudgetError("BUDGET_PROOF_REPLAY_OR_UNKNOWN")
            operation = BudgetOperation.model_validate_json(row[0])
            if proof.intent != operation.intent(proof.intent.reservation_id):
                raise BudgetError("BUDGET_PROOF_INTENT_MISMATCH")
            known, held = proof.amounts(operation.bound_usd_micros)
            if known < row[2]:
                raise BudgetError("BUDGET_KNOWN_CHARGE_CANNOT_DISAPPEAR")
            connection.execute("INSERT INTO proofs VALUES (?,?)", (pin.sha256, pin.model_dump_json()))
            connection.execute("UPDATE reservations SET state=?,known=?,held=?,proof=? WHERE reservation_id=?",
                (proof.kind, known, held, pin.model_dump_json(), proof.intent.reservation_id))
            append_history(connection, proof.intent.reservation_id, self.binding.sha256())


def verify(binding_path: Path) -> BudgetStatus:
    budget = SharedProviderBudget(binding_path)
    if budget.authority.reconciliation.complete:
        return budget.status()
    if budget.path.exists():
        raise BudgetError("BUDGET_UNRECONCILED_AUTHORITY_EXISTS")
    return BudgetStatus(paid_eligible=False, known_usd_micros=None, held_usd_micros=None,
        available_usd_micros=None, reservations=0,
        blockers=("BUDGET_RECONCILIATION_INCOMPLETE", *budget.authority.reconciliation.blockers,
                  *budget.authority.pricing.blockers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "status"))
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.binding)
        raw = result.model_dump_json(indent=2).encode()
        if args.output is not None:
            persist_evidence_bytes(args.output, raw, "BUDGET_OUTPUT_ALREADY_EXISTS")
        print(raw.decode())
        return 0
    except (ValueError, OSError, sqlite3.Error) as error:
        print(json.dumps({"verified": False, "paid_eligible": False, "error": str(error)}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
