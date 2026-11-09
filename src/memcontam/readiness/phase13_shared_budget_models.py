"""Exact USD accounting inputs; file identity is not scientific authorization."""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Annotated, Final, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .phase13_cost_policy_models import Sha256
from .phase13_live_diagnostic_models import DiagnosticModel, DiagnosticRequestIntent

CAP: Final = 45_000_000
Micros = Annotated[int, Field(strict=True, ge=0)]
PositiveMicros = Annotated[int, Field(strict=True, gt=0, le=CAP)]
Identity = Annotated[str, Field(min_length=1)]
OperationKind = Literal["generation", "count", "prefix", "setup"]


class BudgetError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class FilePin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    path: Identity
    sha256: Sha256

    def read(self) -> bytes:
        path = Path(self.path)
        if path != path.resolve():
            raise BudgetError("BUDGET_NONCANONICAL_PATH")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise BudgetError("BUDGET_STALE_SOURCE")
        return raw


class PriorOperation(DiagnosticModel):
    lane_id: Identity
    run_id: Identity
    request_id: Identity
    attempt_id: Identity
    known_usd_micros: Micros
    hold_usd_micros: Micros
    state: Literal["settled", "ambiguous"]
    evidence: FilePin

    @model_validator(mode="after")
    def retain_unknown(self) -> PriorOperation:
        if (self.state == "ambiguous") != (self.hold_usd_micros > 0):
            raise BudgetError("BUDGET_PRIOR_UNKNOWN_CANNOT_BE_ZERO")
        return self


class LaneBalance(DiagnosticModel):
    lane_id: Identity
    complete: bool
    funding: Literal["aggregate", "separate", "unresolved"]
    known_spend_usd_micros: Micros | None
    unknown_hold_usd_micros: Micros | None
    evidence: tuple[FilePin, ...] = Field(min_length=1)
    reason: Identity


class Reconciliation(DiagnosticModel):
    schema_version: Literal["phase13_budget_reconciliation_v1"]
    currency: Literal["USD"]
    account_id: Identity | None
    accounting_root: Identity
    parent_approval: FilePin
    execution_approval: FilePin
    complete: bool
    remaining_allowance_usd_micros: Micros | None
    lanes: tuple[LaneBalance, ...] = Field(min_length=1)
    operations: tuple[PriorOperation, ...]
    blockers: tuple[str, ...]

    @model_validator(mode="after")
    def reconcile_lanes(self) -> Reconciliation:
        lanes = [row.lane_id for row in self.lanes]
        attempts = [row.attempt_id for row in self.operations]
        if len(set(lanes)) != len(lanes) or len(set(attempts)) != len(attempts):
            raise BudgetError("BUDGET_DUPLICATE_HISTORY")
        if any(row.lane_id not in lanes for row in self.operations):
            raise BudgetError("BUDGET_FOREIGN_HISTORY")
        if self.complete:
            if self.blockers or self.remaining_allowance_usd_micros is None or self.account_id is None:
                raise BudgetError("BUDGET_RECONCILIATION_INCOMPLETE")
            for lane in self.lanes:
                operations = [row for row in self.operations if row.lane_id == lane.lane_id]
                if (not lane.complete or lane.funding == "unresolved"
                    or lane.known_spend_usd_micros != sum(row.known_usd_micros for row in operations)
                    or lane.unknown_hold_usd_micros != sum(row.hold_usd_micros for row in operations)):
                    raise BudgetError("BUDGET_RECONCILIATION_INCOMPLETE")
            if self.remaining_allowance_usd_micros + self.prior_total() > CAP:
                raise BudgetError("BUDGET_CAP_EXCEEDED")
        return self

    def prior_total(self) -> int:
        return sum(row.known_usd_micros + row.hold_usd_micros for row in self.operations)


class Price(DiagnosticModel):
    price_id: Identity
    operation: OperationKind
    endpoint: Identity
    model: Identity
    service_tier: Identity
    currency: Literal["USD"]
    units: Literal["USD_micros_per_bounded_operation"]
    upper_bound_usd_micros: PositiveMicros
    request_sha256: Sha256
    as_of: date
    valid_until: date
    evidence: FilePin
    unknown_billing: bool


class Pricing(DiagnosticModel):
    schema_version: Literal["phase13_budget_pricing_v1"]
    account_id: Identity | None
    entries: tuple[Price, ...]
    blockers: tuple[str, ...]


class BudgetBinding(DiagnosticModel):
    schema_version: Literal["phase13_shared_provider_budget_v1"]
    currency: Literal["USD"]
    cap_usd_micros: Annotated[int, Field(strict=True, ge=CAP, le=CAP)]
    account_id: Identity | None
    run_id: Identity
    allowed_runs: tuple[Identity, ...]
    ledger_path: Identity
    lanes: tuple[Identity, ...] = Field(min_length=1)
    parent_approval: FilePin
    execution_approval: FilePin
    reconciliation: FilePin
    pricing: FilePin
    source_files: tuple[FilePin, ...] = Field(min_length=1)
    proof_authority: Identity
    proof_public_key: Sha256 | None


class BudgetOperation(DiagnosticModel):
    lane_id: Identity
    run_id: Identity
    execution_binding_sha256: Sha256
    unit_id: Sha256
    request_id: Identity
    attempt_id: Sha256
    request_sha256: Sha256
    operation: OperationKind
    endpoint: Identity
    model: Identity
    service_tier: Identity
    price_id: Identity
    bound_usd_micros: PositiveMicros
    unknown_count_decision: FilePin | None

    def intent(self, reservation_id: str) -> DiagnosticRequestIntent:
        return DiagnosticRequestIntent(request_id=self.request_id, attempt_id=self.attempt_id,
            request_sha256=self.request_sha256, reservation_id=reservation_id)


class CountDecision(DiagnosticModel):
    schema_version: Literal["phase13_unknown_count_decision_v1"]
    issuer: Identity
    budget_binding_sha256: Sha256
    run_id: Identity
    request_id: Identity
    attempt_id: Sha256
    request_sha256: Sha256
    bound_usd_micros: PositiveMicros
    price_id: Identity
    evidence: tuple[FilePin, ...] = Field(min_length=1)
    signature: Annotated[str, Field(pattern=r"^[0-9a-f]{128}$")]


class AccountingProof(DiagnosticModel):
    schema_version: Literal["phase13_budget_proof_v1"]
    budget_binding_sha256: Sha256
    account_id: Identity
    issuer: Identity
    intent: DiagnosticRequestIntent
    kind: Literal["no_request", "no_charge", "charge", "ambiguous"]
    realized_usd_micros: Micros | None
    provider_request_id: Identity | None
    owner_quiescent: bool
    evidence: tuple[FilePin, ...] = Field(min_length=1)
    signature: Annotated[str, Field(pattern=r"^[0-9a-f]{128}$")]

    def amounts(self, bound: int) -> tuple[int, int]:
        amount = self.realized_usd_micros
        match self.kind:
            case "no_request" | "no_charge":
                if amount != 0 or not self.owner_quiescent:
                    raise BudgetError("BUDGET_RELEASE_PROOF_INVALID")
                if self.kind == "no_request" and self.provider_request_id is not None:
                    raise BudgetError("BUDGET_RELEASE_PROOF_INVALID")
                if self.kind == "no_charge" and self.provider_request_id is None:
                    raise BudgetError("BUDGET_RELEASE_PROOF_INVALID")
                return 0, 0
            case "charge":
                if amount is None or amount == 0 or self.provider_request_id is None:
                    raise BudgetError("BUDGET_CHARGE_PROOF_INVALID")
                return amount, 0
            case "ambiguous":
                known = 0 if amount is None else amount
                return known, max(0, bound - known)
            case unreachable:
                assert_never(unreachable)


class BudgetStatus(DiagnosticModel):
    verified: bool = True
    paid_eligible: bool
    currency: Literal["USD"] = "USD"
    cap_usd_micros: int = CAP
    known_usd_micros: int | None
    held_usd_micros: int | None
    available_usd_micros: int | None
    reservations: int
    blockers: tuple[str, ...]
