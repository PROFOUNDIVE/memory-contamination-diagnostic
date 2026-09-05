from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final, Literal, Self, assert_never

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from .phase13_cost_policy_models import Sha256
from .phase13_v3_authority_models import FrozenModel
from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_cost_models import CostError, NonnegativeInt, ProviderCostEvidence, digest


class TerminalEvidenceError(ValueError):
    code: str

    def __init__(self, code: str = "MAIN_TERMINAL_EVIDENCE_CONFLICT") -> None:
        self.code = code
        super().__init__(code)


class LedgerBindingV3(FrozenModel):
    schema_version: Literal["phase13_main_run_ledger_v3"]
    unit_ids: tuple[Sha256, ...] = Field(min_length=1)
    package_sha256: Sha256
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def unique_units(self) -> Self:
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise TerminalEvidenceError()
        return self


class CompiledRequestV3(FrozenModel):
    stage: str = Field(min_length=1)
    token_count: NonnegativeInt
    compiled_request_hash: Sha256
    immutable_input_hash: Sha256
    native_state_hash: Sha256


class EventIdentity(FrozenModel):
    unit_id: Sha256
    revision: int = Field(gt=0)
    previous_hash: Sha256


class DispatchIdentity(EventIdentity):
    schema_version: Literal["phase13_main_dispatch_evidence_v3"]


class DispatchIntentV3(DispatchIdentity):
    kind: Literal["DISPATCH_INTENT"]
    compiled: CompiledRequestV3 | None


class RequestCompiledV3(DispatchIdentity):
    kind: Literal["REQUEST_COMPILED"]
    compiled: CompiledRequestV3


class AttemptStartedV3(DispatchIdentity):
    kind: Literal["ATTEMPT_STARTED"]
    compiled: CompiledRequestV3


class OverflowV3(DispatchIdentity):
    kind: Literal["INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING"]
    compiled: CompiledRequestV3
    failure_code: Literal["MAIN_INPUT_ENVELOPE_EXCEEDED"]
    transport_attempts: Annotated[int, Field(ge=0, le=0)]
    realized_cost_krw: Annotated[int, Field(ge=0, le=0)]


class AttemptedEvidence(EventIdentity):
    compiled: CompiledRequestV3
    transport_attempts: Annotated[int, Field(ge=1, le=1)]
    cost: ProviderCostEvidence
    realized_cost_krw: NonnegativeInt | None

    @model_validator(mode="after")
    def exact_cost(self) -> Self:
        try:
            known = reconcile_actual(self.cost).realized_krw
        except CostError as error:
            if error.code != "MAIN_TERMINAL_COST_UNKNOWN":
                raise
            known = None
        if self.realized_cost_krw != known:
            raise TerminalEvidenceError()
        return self


class CompletedV3(AttemptedEvidence):
    schema_version: Literal["phase13_main_dispatch_evidence_v3"]
    kind: Literal["COMPLETED"]
    result_hash: Sha256


class ProviderFailureV3(AttemptedEvidence):
    schema_version: Literal["phase13_main_dispatch_evidence_v3"]
    kind: Literal["ATTEMPTED_PROVIDER_FAILURE"]
    failure_code: str = Field(min_length=1)
    observation_hash: Sha256


class NoRequestV3(EventIdentity):
    schema_version: Literal["phase13_main_reconciliation_v3"]
    kind: Literal["NO_REQUEST"]
    compiled: CompiledRequestV3 | None
    proof_hash: Sha256


class AmbiguousAttemptV3(AttemptedEvidence):
    schema_version: Literal["phase13_main_reconciliation_v3"]
    kind: Literal["AMBIGUOUS_ATTEMPT"]
    failure_code: Literal["MAIN_AMBIGUOUS_ATTEMPT"]
    proof_hash: Sha256


class CostReconciledV3(AttemptedEvidence):
    schema_version: Literal["phase13_main_reconciliation_v3"]
    kind: Literal["COST_RECONCILED"]
    proof_hash: Sha256

    @model_validator(mode="after")
    def known_cost(self) -> Self:
        if self.realized_cost_krw is None:
            raise TerminalEvidenceError("MAIN_TERMINAL_COST_UNKNOWN")
        return self


EventV3 = Annotated[
    DispatchIntentV3 | RequestCompiledV3 | AttemptStartedV3 | OverflowV3 | CompletedV3
    | ProviderFailureV3 | NoRequestV3 | AmbiguousAttemptV3 | CostReconciledV3, Field(discriminator="kind"),
]
EVENT_ADAPTER: Final[TypeAdapter[EventV3]] = TypeAdapter(EventV3)
StateKind = Literal[
    "PENDING", "DISPATCH_INTENT_PERSISTED", "REQUEST_COMPILED", "ATTEMPT_STARTED",
    "INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING", "COMPLETED",
    "ATTEMPTED_PROVIDER_FAILURE", "AMBIGUOUS_ATTEMPT",
]


@dataclass(frozen=True, slots=True)
class EvidenceState:
    kind: StateKind
    revision: int
    event_hash: str
    compiled: CompiledRequestV3 | None = None
    attempted_cost: ProviderCostEvidence | None = None


def parse_event(raw: bytes) -> EventV3:
    try:
        return EVENT_ADAPTER.validate_json(raw)
    except ValidationError as error:
        raise TerminalEvidenceError() from error


def advance(state: EvidenceState, event: EventV3) -> EvidenceState:
    if event.revision != state.revision + 1 or event.previous_hash != state.event_hash:
        raise TerminalEvidenceError()
    target: StateKind
    allowed: tuple[str, ...]
    cost: ProviderCostEvidence | None = None
    match event:
        case DispatchIntentV3():
            allowed, target = ("PENDING",), "DISPATCH_INTENT_PERSISTED"
        case RequestCompiledV3():
            allowed, target = ("DISPATCH_INTENT_PERSISTED",), "REQUEST_COMPILED"
        case AttemptStartedV3():
            allowed, target = ("REQUEST_COMPILED",), "ATTEMPT_STARTED"
        case OverflowV3():
            match event.kind:
                case "INPUT_ENVELOPE_OVERFLOW":
                    allowed, target = ("REQUEST_COMPILED",), "INPUT_ENVELOPE_OVERFLOW"
                case "TERMINAL_TECHNICAL_MISSING":
                    allowed, target = ("INPUT_ENVELOPE_OVERFLOW",), "TERMINAL_TECHNICAL_MISSING"
                case unreachable_kind:
                    assert_never(unreachable_kind)
        case NoRequestV3():
            allowed, target = ("DISPATCH_INTENT_PERSISTED", "REQUEST_COMPILED"), "PENDING"
        case CompletedV3() | ProviderFailureV3() | AmbiguousAttemptV3():
            allowed, target, cost = ("ATTEMPT_STARTED",), event.kind, event.cost
        case CostReconciledV3():
            allowed = ("COMPLETED", "ATTEMPTED_PROVIDER_FAILURE", "AMBIGUOUS_ATTEMPT")
            target, cost = state.kind, event.cost
            if state.attempted_cost is None:
                raise TerminalEvidenceError()
            try:
                reconcile_actual(state.attempted_cost)
            except CostError as error:
                if error.code != "MAIN_TERMINAL_COST_UNKNOWN":
                    raise
            else:
                raise TerminalEvidenceError()
        case unreachable:
            assert_never(unreachable)
    if state.kind not in allowed:
        raise TerminalEvidenceError()
    match event:
        case RequestCompiledV3():
            if state.compiled is not None and event.compiled != state.compiled:
                raise TerminalEvidenceError()
        case DispatchIntentV3() | AttemptStartedV3() | OverflowV3() | NoRequestV3() | CompletedV3() | ProviderFailureV3() | AmbiguousAttemptV3() | CostReconciledV3():
            if event.compiled != state.compiled:
                raise TerminalEvidenceError()
        case unreachable:
            assert_never(unreachable)
    return EvidenceState(target, event.revision, digest(event), event.compiled, cost)
