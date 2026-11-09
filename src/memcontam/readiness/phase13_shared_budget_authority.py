"""Read-only authority, pricing and reconciliation checks for shared accounting."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .phase13_shared_budget_models import (
    AccountingProof, BudgetBinding, BudgetError, BudgetOperation, CountDecision, FilePin,
    Pricing, Reconciliation,
)


@dataclass(frozen=True, slots=True)
class BudgetAuthority:
    binding: BudgetBinding
    reconciliation: Reconciliation
    pricing: Pricing

    @classmethod
    def load(cls, path: Path) -> BudgetAuthority:
        binding = BudgetBinding.model_validate_json(path.read_bytes())
        reconciliation = Reconciliation.model_validate_json(binding.reconciliation.read())
        pricing = Pricing.model_validate_json(binding.pricing.read())
        root = Path(reconciliation.accounting_root)
        if (root != root.resolve() or path != root / "binding.json"
            or Path(binding.ledger_path) != root / "aggregate.sqlite3"
            or Path(binding.ledger_path) != Path(binding.ledger_path).resolve()
            or binding.account_id != reconciliation.account_id or binding.account_id != pricing.account_id
            or binding.parent_approval != reconciliation.parent_approval
            or binding.execution_approval != reconciliation.execution_approval
            or set(binding.lanes) != {row.lane_id for row in reconciliation.lanes}
            or len(binding.lanes) != len(set(binding.lanes))):
            raise BudgetError("BUDGET_AUTHORITY_SCOPE_MISMATCH")
        pins = [binding.parent_approval, binding.execution_approval, *binding.source_files]
        pins.extend(row.evidence for row in reconciliation.operations)
        pins.extend(pin for lane in reconciliation.lanes for pin in lane.evidence)
        pins.extend(price.evidence for price in pricing.entries)
        for pin in pins:
            pin.read()
        if len({price.price_id for price in pricing.entries}) != len(pricing.entries):
            raise BudgetError("BUDGET_DUPLICATE_PRICE")
        return cls(binding, reconciliation, pricing)

    def check_operation(self, operation: BudgetOperation, *, current: bool = True) -> None:
        if any((row.run_id, row.request_id) == (operation.run_id, operation.request_id)
               or row.attempt_id == operation.attempt_id for row in self.reconciliation.operations):
            raise BudgetError("BUDGET_RESERVATION_REPLAY")
        if operation.lane_id not in self.binding.lanes or operation.run_id not in self.binding.allowed_runs:
            raise BudgetError("BUDGET_OPERATION_SCOPE_MISMATCH")
        price = next((row for row in self.pricing.entries if row.price_id == operation.price_id), None)
        if price is None or self.pricing.blockers:
            raise BudgetError("BUDGET_PRICE_UNKNOWN")
        today = datetime.now(timezone.utc).date()
        if ((current and not price.as_of <= today <= price.valid_until)
            or (price.operation, price.endpoint, price.model, price.service_tier, price.request_sha256)
                != (operation.operation, operation.endpoint, operation.model, operation.service_tier,
                    operation.request_sha256)
            or operation.bound_usd_micros < price.upper_bound_usd_micros):
            raise BudgetError("BUDGET_PRICE_OR_BOUND_INVALID")
        if price.unknown_billing:
            if operation.operation != "count" or operation.unknown_count_decision is None:
                raise BudgetError("BUDGET_UNKNOWN_COUNT_DECISION_REQUIRED")
            decision = CountDecision.model_validate_json(operation.unknown_count_decision.read())
            self.authenticate(decision)
            if (decision.issuer != self.binding.proof_authority
                or decision.budget_binding_sha256 != self.binding.sha256()
                or (decision.run_id, decision.request_id, decision.attempt_id, decision.request_sha256,
                    decision.price_id, decision.bound_usd_micros)
                != (operation.run_id, operation.request_id, operation.attempt_id, operation.request_sha256,
                    operation.price_id, operation.bound_usd_micros)):
                raise BudgetError("BUDGET_COUNT_DECISION_MISMATCH")
            for pin in decision.evidence:
                pin.read()

    def proof(self, pin: FilePin) -> AccountingProof:
        proof = AccountingProof.model_validate_json(pin.read())
        self.authenticate(proof)
        if (proof.issuer != self.binding.proof_authority or proof.account_id != self.binding.account_id
            or proof.budget_binding_sha256 != self.binding.sha256()):
            raise BudgetError("BUDGET_PROOF_AUTHORITY_MISMATCH")
        for witness in proof.evidence:
            witness.read()
        return proof

    def authenticate(self, record: AccountingProof | CountDecision) -> None:
        if self.binding.proof_public_key is None:
            raise BudgetError("BUDGET_PROOF_AUTHORITY_UNESTABLISHED")
        raw = json.dumps(record.model_dump(mode="json", exclude={"signature"}),
                         sort_keys=True, separators=(",", ":")).encode()
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.binding.proof_public_key))
        try:
            key.verify(bytes.fromhex(record.signature), raw)
        except InvalidSignature as error:
            raise BudgetError("BUDGET_PROOF_SIGNATURE_INVALID") from error
