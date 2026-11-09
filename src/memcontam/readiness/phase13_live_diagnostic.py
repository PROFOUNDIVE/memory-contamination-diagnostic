from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .phase13_live_diagnostic_evidence import (
    call_receipt_sha256, live_call_cost, load_live_evidence, validate_live_output, verify_live_sources,
)
from .phase13_live_diagnostic_models import (
    DiagnosticRequestIntent, DiagnosticRequestReceipt, DiagnosticStatus, LiveDiagnosticBinding,
    LiveDiagnosticError, LiveDiagnosticEvidence, LiveDiagnosticOutput, LiveDiagnosticScope,
)
from .phase13_main_live_dispatch import persist_evidence_bytes
from .phase13_main_production import ProductionObject
from .phase13_main_runner_store import (
    append_event, connect, create_ledger_file, expected_metadata, validate_integrity,
)


class LiveDiagnosticLedger:
    def __init__(self, binding: LiveDiagnosticBinding, units: tuple[ProductionObject, ...]) -> None:
        self.binding = LiveDiagnosticBinding.model_validate_json(binding.model_dump_json())
        self.path = Path(binding.ledger_path)
        self.units = units
        if (self.path != self.path.resolve() or self.path.name != "ledger.sqlite3"
            or len(units) != len(binding.observations)
            or tuple(unit.sequence for unit in units) != tuple(range(len(units)))
            or len({unit.unit_id for unit in units}) != len(units)
            or any(unit.diagnostic_trial_count != 1 or unit.projected_cost_krw != 0 for unit in units)):
            raise LiveDiagnosticError("DIAGNOSTIC_BINDING_MISMATCH")
        verify_live_sources(self.binding, units)

    @classmethod
    def create(cls, binding: LiveDiagnosticBinding, units: tuple[ProductionObject, ...]) -> LiveDiagnosticLedger:
        ledger = cls(binding, units)
        create_ledger_file(ledger.path, binding, units)
        return ledger

    @classmethod
    def open(cls, binding: LiveDiagnosticBinding, units: tuple[ProductionObject, ...]) -> LiveDiagnosticLedger:
        ledger = cls(binding, units)
        if not ledger.path.is_file():
            raise LiveDiagnosticError("DIAGNOSTIC_LEDGER_MISSING")
        with connect(ledger.path) as connection:
            connection.execute("BEGIN")
            ledger._validate(connection)
        return ledger

    def _unit(self, unit_id: str) -> ProductionObject:
        unit = next((unit for unit in self.units if unit.unit_id == unit_id), None)
        if unit is None:
            raise LiveDiagnosticError("DIAGNOSTIC_UNIT_UNKNOWN")
        return unit

    def claim(self, unit_id: str) -> None:
        unit = self._unit(unit_id)
        child = self.binding.observations[unit.sequence].child_execution_id
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            states = self._validate(connection)
            self._require_known_costs(connection)
            local = [row for row in self.units if self.binding.observations[row.sequence].child_execution_id == child]
            first = next((row for row in local if states[row.sequence] != "COMPLETED"), None)
            if first != unit or states[unit.sequence] != "PENDING":
                raise LiveDiagnosticError("DIAGNOSTIC_ORDER_OR_AMBIGUOUS_INTENT")
            append_event(connection, unit_id, "INTENT")

    def claim_request(self, unit_id: str, intent: DiagnosticRequestIntent) -> None:
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate(connection)
            self._require_known_costs(connection)
            append_event(connection, unit_id, "REQUEST_INTENT", reason=intent.model_dump_json())
            self._validate(connection)

    def record_request(self, unit_id: str, receipt: DiagnosticRequestReceipt) -> None:
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate(connection)
            append_event(connection, unit_id, "REQUEST_RECEIPT", reason=receipt.model_dump_json(),
                         realized_cost=receipt.realized_cost_usd_micros)
            self._validate(connection)

    def _intent(self, connection: sqlite3.Connection, unit_id: str) -> str:
        row = connection.execute("SELECT event_hash FROM events WHERE unit_id=? AND kind='INTENT'",
                                 (unit_id,)).fetchone()
        if row is None:
            raise LiveDiagnosticError("DIAGNOSTIC_INTENT_REQUIRED")
        return str(row[0])

    def persist_output(self, unit_id: str, output: LiveDiagnosticOutput) -> str:
        unit = self._unit(unit_id)
        parsed_output = LiveDiagnosticOutput.model_validate_json(output.model_dump_json())
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            states = self._validate(connection)
            if states[unit.sequence] != "DISPATCH_INTENT_PERSISTED":
                raise LiveDiagnosticError("DIAGNOSTIC_INTENT_REQUIRED")
            record = LiveDiagnosticEvidence(binding_sha256=self.binding.sha256(),
                intent_event_hash=self._intent(connection, unit_id), unit_id=unit_id,
                sequence=unit.sequence, output=parsed_output)
            self._validate_record(connection, record)
            return persist_evidence_bytes(self._evidence_path(unit_id),
                record.model_dump_json().encode(), "DIAGNOSTIC_EVIDENCE_ALREADY_EXISTS")

    def reconcile(self, unit_id: str, evidence_sha256: str) -> None:
        with connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate(connection)
            record = load_live_evidence(self._evidence_path(unit_id), evidence_sha256)
            if record.unit_id != unit_id:
                raise LiveDiagnosticError("DIAGNOSTIC_CROSS_SCOPE")
            self._validate_record(connection, record)
            kind = "COMPLETED" if record.output.disposition == "COMPLETED" else "TERMINAL"
            append_event(connection, unit_id, kind, evidence_sha256=evidence_sha256)

    def _evidence_path(self, unit_id: str) -> Path:
        unit = self._unit(unit_id)
        return self.path.parent / "units" / f"{unit.sequence:06d}-{unit_id}.json"

    def _validate_record(self, connection: sqlite3.Connection, record: LiveDiagnosticEvidence) -> None:
        unit = self._unit(record.unit_id)
        if (record.binding_sha256 != self.binding.sha256() or record.sequence != unit.sequence
            or record.intent_event_hash != self._intent(connection, unit.unit_id)):
            raise LiveDiagnosticError("DIAGNOSTIC_BINDING_MISMATCH")
        evidence = validate_live_output(unit, record.output, self.binding)
        scope = evidence.runtime_evidence.diagnostic_scope
        if scope is None or scope.run_id != self.binding.observations[unit.sequence].run_id:
            raise LiveDiagnosticError("DIAGNOSTIC_BINDING_MISMATCH")
        receipts = self._requests(connection, unit)
        plans = self.binding.observations[unit.sequence].requests
        by_call = {plan.native_call_id: receipts.get(plan.request_id) for plan in plans
                    if plan.operation == "generation"}
        from .phase13_diagnostic_exchange import selected_context
        from .phase13_diagnostic_exchange_validate import validate_exchange, validate_exchange_receipts
        context = selected_context(self.binding)
        for call in record.output.provider_calls:
            receipt = by_call.get(call.call_id)
            request_hash = hashlib.sha256(json.dumps(call.messages, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            if context is not None:
                if not isinstance(scope, LiveDiagnosticScope):
                    raise LiveDiagnosticError("DIAGNOSTIC_CROSS_SCOPE")
                exchange = validate_exchange(call, context, scope)
                validate_exchange_receipts(exchange, receipts, context)
                request_hash = exchange.generation.request.sha256
            if (receipt is None or receipt.receipt_sha256 != call_receipt_sha256(call)
                or receipt.intent.request_sha256 != request_hash
                or receipt.realized_cost_usd_micros != live_call_cost(call)
                or receipt.outcome != ("failed" if call.error_type else "completed")):
                raise LiveDiagnosticError("DIAGNOSTIC_REQUEST_JOIN_INVALID")
        if record.output.disposition == "COMPLETED" and (
            len(receipts) != len(plans)
            or any(receipt.outcome != "completed" or receipt.realized_cost_usd_micros is None
                   for receipt in receipts.values())
        ):
            raise LiveDiagnosticError("DIAGNOSTIC_REQUEST_JOIN_INVALID")

    def _requests(self, connection: sqlite3.Connection, unit: ProductionObject) -> dict[str, DiagnosticRequestReceipt]:
        plans = self.binding.observations[unit.sequence].requests
        receipts: dict[str, DiagnosticRequestReceipt] = {}
        active: DiagnosticRequestIntent | None = None
        for kind, raw, amount in connection.execute(
            "SELECT kind, reason, realized_cost_usd_micros FROM events WHERE unit_id=? "
            "AND kind IN ('REQUEST_INTENT','REQUEST_RECEIPT') ORDER BY event_sequence", (unit.unit_id,)):
            if kind == "REQUEST_INTENT":
                intent = DiagnosticRequestIntent.model_validate_json(raw)
                if (active is not None or len(receipts) >= len(plans)
                    or intent.request_id != plans[len(receipts)].request_id or amount != 0
                    or any(row.outcome != "completed" or row.realized_cost_usd_micros is None for row in receipts.values())
                    or intent.attempt_id != hashlib.sha256((self.binding.sha256() + intent.request_id).encode()).hexdigest()):
                    raise LiveDiagnosticError("DIAGNOSTIC_ORDER_OR_AMBIGUOUS_INTENT")
                active = intent
            else:
                receipt = DiagnosticRequestReceipt.model_validate_json(raw)
                if active != receipt.intent or receipt.realized_cost_usd_micros != amount:
                    raise LiveDiagnosticError("DIAGNOSTIC_RECEIPT_MISMATCH")
                receipts[receipt.intent.request_id] = receipt
                active = None
        return receipts

    def _validate(self, connection: sqlite3.Connection) -> list[str]:
        if dict(connection.execute("SELECT key,value FROM metadata")) != expected_metadata(self.binding, self.units):
            raise LiveDiagnosticError("DIAGNOSTIC_BINDING_MISMATCH")
        verify_live_sources(self.binding, self.units)
        states, _, _ = validate_integrity(connection, self.units)
        for unit in self.units:
            self._requests(connection, unit)
        for unit_id, kind, digest, amount in connection.execute(
            "SELECT unit_id,kind,evidence_sha256,realized_cost_usd_micros FROM events ORDER BY event_sequence"):
            if kind in {"COMPLETED", "TERMINAL"}:
                record = load_live_evidence(self._evidence_path(unit_id), digest)
                self._validate_record(connection, record)
                if (record.unit_id != unit_id or amount != 0
                    or (record.output.disposition == "COMPLETED") != (kind == "COMPLETED")):
                    raise LiveDiagnosticError("DIAGNOSTIC_EVIDENCE_INVALID")
            elif kind not in {"INTENT", "REQUEST_INTENT", "REQUEST_RECEIPT"} or (kind == "INTENT" and amount != 0):
                raise LiveDiagnosticError("DIAGNOSTIC_EVENT_INVALID")
        return states

    def _require_known_costs(self, connection: sqlite3.Connection) -> None:
        if connection.execute("SELECT 1 FROM events WHERE kind='REQUEST_RECEIPT' "
                              "AND (realized_cost_usd_micros IS NULL OR json_extract(reason, '$.outcome')='ambiguous') "
                              "LIMIT 1").fetchone():
            raise LiveDiagnosticError("DIAGNOSTIC_UNKNOWN_LIABILITY")

    def status(self) -> DiagnosticStatus:
        with connect(self.path) as connection:
            connection.execute("BEGIN")
            states = self._validate(connection)
            costs = [row[0] for row in connection.execute(
                "SELECT realized_cost_usd_micros FROM events WHERE kind='REQUEST_RECEIPT'")]
            intent_count = int(connection.execute(
                "SELECT COUNT(*) FROM events WHERE kind='REQUEST_INTENT'").fetchone()[0])
        return {"completed": states.count("COMPLETED"), "terminal": states.count("TERMINAL_TECHNICAL_MISSING"),
                "pending": states.count("PENDING"), "in_flight": states.count("DISPATCH_INTENT_PERSISTED"),
                "known_cost_usd_micros": sum(cost for cost in costs if cost is not None),
                "unknown_cost_requests": costs.count(None), "unresolved_requests": intent_count - len(costs)}
