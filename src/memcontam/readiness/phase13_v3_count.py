from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import Field, JsonValue

from .phase13_cost_policy_models import Sha256
from .phase13_v3_authority_models import FrozenModel
from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_cost_models import DecimalString, ProviderCostEvidence, canonical_bytes
from .phase13_v3_request import STAGES, CompiledProviderRequestV3, PackageBindingV3, RequestKeyV3, compile_request_bytes
from .phase13_v3_terminal_models import CompiledRequestV3, TerminalEvidenceError

if TYPE_CHECKING:
    from .phase13_v3_terminal_ledger import TerminalLedgerV3


class CountIdentityV3(FrozenModel):
    schema_version: Literal["phase13_responses_count_projection_v1"] = "phase13_responses_count_projection_v1"
    provider: Literal["openai_responses"] = "openai_responses"
    base_url: str = Field(min_length=1)
    model: Literal["gpt-5.6-luna"] = "gpt-5.6-luna"
    sdk_version: str
    account_sha256: Sha256
    runtime_sha256: Sha256
    source_sha256: Sha256
    schema_sha256: Sha256


class CountOperationV3(FrozenModel):
    binding: PackageBindingV3
    key: RequestKeyV3
    compiled: CompiledRequestV3
    provider: CountIdentityV3
    projection_sha256: Sha256
    maximum_input_tokens: int
    maximum_count_calls: Literal[1] = 1
    delivery: Literal["POSSIBLY_SENT"] = "POSSIBLY_SENT"
    cost_status: Literal["UNKNOWN"] = "UNKNOWN"


class CountReceiptV3(FrozenModel):
    schema_version: Literal["phase13_provider_count_receipt_v1"] = "phase13_provider_count_receipt_v1"
    operation: CountOperationV3
    object: Literal["response.input_tokens"]
    input_tokens: Annotated[int, Field(strict=True, ge=0)]
    monetary_cost_usd: DecimalString | None

    def require_known_cost(self) -> int:
        if self.monetary_cost_usd is None:
            raise TerminalEvidenceError("MAIN_COUNT_COST_UNKNOWN")
        return reconcile_actual(ProviderCostEvidence(monetary_cost=self.monetary_cost_usd,
                                                     currency="USD")).realized_krw


def count_projection(compiled: CompiledProviderRequestV3) -> bytes:
    if compiled.request_bytes != compile_request_bytes(compiled.key, compiled.material):
        raise TerminalEvidenceError("MAIN_COUNT_REQUEST_BINDING_MISMATCH")
    body = json.loads(compiled.request_bytes)
    if set(body) != {"model", "input", "temperature", "top_p", "max_output_tokens",
                     "service_tier", "store", "tools", "reasoning"}:
        raise TerminalEvidenceError("MAIN_COUNT_UNSUPPORTED_REQUEST")
    if (body["model"] != "gpt-5.6-luna" or body["tools"] != [] or body["store"] is not False
            or body["service_tier"] != "default"
            or body["reasoning"] != {"mode": "standard", "effort": "none", "context": "current_turn"}):
        raise TerminalEvidenceError("MAIN_COUNT_UNSUPPORTED_REQUEST")
    return json.dumps({name: body[name] for name in ("model", "input", "tools", "reasoning")},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def count_operation(compiled: CompiledProviderRequestV3, provider: CountIdentityV3) -> CountOperationV3:
    return CountOperationV3(binding=compiled.binding, key=compiled.key, compiled=compiled.evidence,
        provider=provider, projection_sha256=hashlib.sha256(count_projection(compiled)).hexdigest(),
        maximum_input_tokens=STAGES[compiled.key.stage][0])


def validate_count_receipt(compiled: CompiledProviderRequestV3, receipt: CountReceiptV3,
                           provider: CountIdentityV3) -> None:
    if receipt.operation != count_operation(compiled, provider):
        raise TerminalEvidenceError("MAIN_COUNT_REQUEST_BINDING_MISMATCH")
    receipt.require_known_cost()
    if receipt.input_tokens > STAGES[compiled.key.stage][0]:
        raise TerminalEvidenceError("MAIN_INPUT_ENVELOPE_EXCEEDED")


def read_count_record(ledger: TerminalLedgerV3, unit_id: str, role: str) -> bytes | None:
    return ledger.count_record(unit_id, role)


def count_costs_krw(ledger: TerminalLedgerV3, unit_ids: tuple[str, ...] | None = None) -> int:
    total = 0
    identifiers = (tuple(unit_id for unit_id, _raw in ledger.count_records("count-started"))
                   if unit_ids is None else unit_ids)
    for unit_id in identifiers:
        receipt_raw = read_count_record(ledger, unit_id, "count-receipt")
        started = read_count_record(ledger, unit_id, "count-started")
        if receipt_raw is None or started is None:
            raise TerminalEvidenceError("MAIN_COUNT_COST_UNKNOWN")
        receipt = CountReceiptV3.model_validate_json(receipt_raw)
        operation = CountOperationV3.model_validate_json(started)
        if (receipt.operation != operation or operation.key.dispatch_id != unit_id
                or operation.compiled != ledger.state(unit_id).compiled
                or operation.binding.identity != ledger.binding.identity
                or operation.binding.package_sha256 != ledger.binding.package_sha256
                or operation.binding.authorization_sha256 != ledger.binding.authorization_sha256):
            raise TerminalEvidenceError("MAIN_COUNT_REQUEST_BINDING_MISMATCH")
        total += receipt.require_known_cost()
    return total


def count_recovery_gate(ledger: TerminalLedgerV3) -> None:
    for unit_id, started in ledger.count_records("count-started"):
        state = ledger.state(unit_id)
        operation = CountOperationV3.model_validate_json(started)
        if (operation.key.dispatch_id != unit_id or operation.compiled != state.compiled
                or operation.binding.identity != ledger.binding.identity
                or operation.binding.package_sha256 != ledger.binding.package_sha256
                or operation.binding.authorization_sha256 != ledger.binding.authorization_sha256):
            raise TerminalEvidenceError("MAIN_COUNT_REQUEST_BINDING_MISMATCH")
        receipt_raw = read_count_record(ledger, unit_id, "count-receipt")
        if receipt_raw is None:
            raise TerminalEvidenceError("MAIN_COUNT_AMBIGUOUS_RECONCILIATION_REQUIRED")
        receipt = CountReceiptV3.model_validate_json(receipt_raw)
        if receipt.operation != operation:
            raise TerminalEvidenceError("MAIN_COUNT_REQUEST_BINDING_MISMATCH")
        receipt.require_known_cost()
        if state.kind == "REQUEST_COMPILED":
            raise TerminalEvidenceError("MAIN_COUNT_RECEIPT_RECONCILIATION_REQUIRED")


COUNT_FAILURE_CODES: Final = frozenset({
    "MAIN_COUNT_FAILED_RECONCILIATION_REQUIRED", "MAIN_COUNT_RESPONSE_INVALID",
    "MAIN_COUNT_REQUEST_BINDING_MISMATCH", "MAIN_COUNT_UNSUPPORTED_REQUEST",
    "MAIN_COUNT_RUNTIME_CONTRACT_MISMATCH", "MAIN_COUNT_RECEIPT_REQUIRED",
    "MAIN_COUNT_COST_UNKNOWN", "MAIN_INPUT_ENVELOPE_EXCEEDED",
})


def count_failure_bytes(code: str) -> bytes:
    failure_code = code if code in COUNT_FAILURE_CODES else "MAIN_COUNT_FAILED_RECONCILIATION_REQUIRED"
    fields: dict[str, JsonValue] = {"schema_version": "phase13_provider_count_failure_v1",
        "failure_code": failure_code, "generation_attempts": 0, "count_calls_maximum": 1,
        "delivery": "POSSIBLY_SENT", "monetary_cost_usd": None}
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def receipt_bytes(receipt: CountReceiptV3 | CountOperationV3) -> bytes:
    return canonical_bytes(receipt)
