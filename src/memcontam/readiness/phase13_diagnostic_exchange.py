from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, Protocol

from openai.types.responses import Response
from pydantic import ConfigDict, Field

from .phase13_diagnostic_runtime_contract import DiagnosticExecutionContext, _read
from .phase13_live_diagnostic_models import (
    DiagnosticModel,
    DiagnosticRequestIntent,
    LiveDiagnosticBinding,
    LiveDiagnosticError,
)
from .phase13_main_live_dispatch import persist_evidence_bytes
from .phase13_shared_budget_models import BudgetOperation, FilePin
from .phase13_v3_request import STAGES, Stage


@dataclass(frozen=True, slots=True)
class DiagnosticWireReply:
    status_code: int
    body: bytes
    provider_request_id: str | None = None


class DiagnosticTransport(Protocol):
    @property
    def provenance(self) -> Literal["QA_IN_PROCESS", "PROVIDER"]: ...

    def send(self, operation: BudgetOperation, raw: bytes, before_egress: Callable[[], None]) -> DiagnosticWireReply: ...


class DiagnosticRates(DiagnosticModel):
    input_micros_per_token: str = Field(pattern=r"^\d+(\.\d+)?$")
    output_micros_per_token: str = Field(pattern=r"^\d+(\.\d+)?$")
    count_micros: int = Field(gt=0)


class DiagnosticCount(DiagnosticModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    object: Literal["response.input_tokens"]
    input_tokens: int = Field(ge=0)


class DiagnosticOperationEvidence(DiagnosticModel):
    provenance: Literal["QA_IN_PROCESS", "PROVIDER"]
    operation: BudgetOperation
    intent: DiagnosticRequestIntent
    request: FilePin
    response: FilePin | None
    status_code: int | None
    provider_request_id: str | None


class DiagnosticExchangeEvidence(DiagnosticModel):
    schema_version: Literal["phase13_diagnostic_exchange_v1"] = "phase13_diagnostic_exchange_v1"
    generation: DiagnosticOperationEvidence
    count: FilePin
    native_state_sha256: str


class DiagnosticCallContract(DiagnosticModel):
    schema_version: Literal["phase13_diagnostic_call_v1"] = "phase13_diagnostic_call_v1"
    exchange: FilePin


class DiagnosticRuntimeHold(DiagnosticModel):
    disposition: Literal["HOLD"] = "HOLD"
    code: str
    generation_attempts: int = Field(ge=0)
    unit_id: str
    method_calls: tuple[FilePin, ...] = ()


class DiagnosticStop(BaseException):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def persist_blob(path: Path, raw: bytes) -> FilePin:
    digest = persist_evidence_bytes(path, raw, "DIAGNOSTIC_EXCHANGE_ALREADY_EXISTS")
    return FilePin(path=str(path.resolve()), sha256=digest)


def selected_context(binding: LiveDiagnosticBinding) -> DiagnosticExecutionContext | None:
    pins = [pin for pin in binding.source_files if pin.sha256 == binding.registration_sha256]
    if not pins:
        return None
    if len(pins) != 1:
        raise LiveDiagnosticError("DIAGNOSTIC_REGISTRATION_AMBIGUOUS")
    return DiagnosticExecutionContext.load(binding, Path(pins[0].path))


def rates(context: DiagnosticExecutionContext) -> DiagnosticRates:
    pin = context.contract.rate_card
    if pin is None:
        raise LiveDiagnosticError("DIAGNOSTIC_RATE_CARD_UNESTABLISHED")
    return DiagnosticRates.model_validate_json(_read(pin))


def parsed_generation(exchange: DiagnosticExchangeEvidence, context: DiagnosticExecutionContext, stage: Stage) -> Response | None:
    generation = exchange.generation
    if generation.response is None:
        return None
    raw = _read(generation.response)
    if generation.status_code != 200:
        return None
    response = Response.model_validate_json(raw)
    counted = DiagnosticOperationEvidence.model_validate_json(_read(exchange.count))
    if counted.response is None or counted.status_code != 200:
        raise LiveDiagnosticError("DIAGNOSTIC_COUNT_MISSING")
    count = DiagnosticCount.model_validate_json(_read(counted.response))
    usage = response.usage
    if (not response.id.strip() or response.model != context.contract.model or response.service_tier != "default" or usage is None
        or usage.input_tokens != count.input_tokens or count.input_tokens > STAGES[stage][0]
        or usage.output_tokens > STAGES[stage][1]
        or usage.total_tokens != usage.input_tokens + usage.output_tokens
        or any(item.type != "message" or any(part.type != "output_text" for part in item.content)
               for item in response.output)):
        raise LiveDiagnosticError("DIAGNOSTIC_RESPONSE_IDENTITY_OR_USAGE")
    return response


def generation_cost(response: Response | None, context: DiagnosticExecutionContext) -> int | None:
    if response is None or response.usage is None:
        return None
    card = rates(context)
    amount = (Decimal(card.input_micros_per_token) * response.usage.input_tokens
              + Decimal(card.output_micros_per_token) * response.usage.output_tokens)
    if amount <= 0 or amount != amount.to_integral_value():
        raise LiveDiagnosticError("DIAGNOSTIC_COST_NOT_REPRESENTABLE")
    return int(amount)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
