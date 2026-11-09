from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
from pydantic import TypeAdapter, ValidationError

from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema import CallEvent

from .phase13_cost_policy import Phase13ProviderCallError
from .phase13_diagnostic_exchange import (
    DiagnosticCallContract,
    DiagnosticCount,
    DiagnosticExchangeEvidence,
    DiagnosticOperationEvidence,
    DiagnosticStop,
    DiagnosticTransport,
    generation_cost,
    parsed_generation,
    persist_blob,
    rates,
)
from .phase13_live_diagnostic_evidence import call_receipt_sha256, live_call_cost
from .phase13_live_diagnostic_models import DiagnosticRequestReceipt, LiveDiagnosticError
from .phase13_shared_budget_models import BudgetError, BudgetOperation, FilePin
from .phase13_v3_request import STAGES, MessageV3, RequestMaterialV3, Stage

if TYPE_CHECKING:
    from .phase13_diagnostic_admission import DiagnosticRuntimeResources


class DiagnosticProviderFailure(RuntimeError):
    def __init__(self, response: LLMResponse, code: str) -> None:
        self.code = code
        self.provider_attempts_count = 1
        self.provider_latency_ms = response.latency_ms
        self.provider_token_usage = response.token_usage
        self.provider_status = response.raw["status"]
        self.provider_response_status = response.raw["status"]
        self.provider_incomplete_reason = response.raw.get("incomplete_reason")
        self.provider_response_id = response.raw["response_id"]
        self.provider_returned_model = response.raw["model"]
        self.provider_service_tier = response.raw["service_tier"]
        self.provider_usage = response.raw["usage"]
        self.provider_cost_usd = response.raw["cost_usd"]
        self.derived_cost_usd = response.raw["derived_cost_usd"]
        self.authoritative_provider_cost_usd = None
        self.provider_cost_source = response.raw["cost_source"]
        self.provider_request_contract = response.raw["request_contract"]
        self.provider_authority_contract = response.raw["authority_contract"]
        super().__init__(code)


class DiagnosticClient:
    def __init__(self, resources: DiagnosticRuntimeResources, transport: DiagnosticTransport) -> None:
        self.resources, self.transport = resources, transport
        observation = resources.context.binding.observations[resources.unit.sequence]
        self.plans = tuple(row for row in observation.requests if row.operation == "generation")
        first = self.plans[0].native_call_id
        if first is None:
            raise LiveDiagnosticError("DIAGNOSTIC_NATIVE_CALL_REQUIRED")
        self.recorder = MethodCallRecorder(self, event_callback=self.record,
            trial_context={"trial_id": first.rsplit(":call:", 1)[0]})
        self.active: DiagnosticOperationEvidence | None = None
        self.generation_attempts = 0
        self.method_call_files: list[FilePin] = []

    def send(self, operation: BudgetOperation, raw: bytes) -> DiagnosticOperationEvidence:
        resource = self.resources
        resource.context.verify()
        intent = resource.budget.reserve(operation)
        request = persist_blob(resource.exchange_root / f"{intent.attempt_id}.request", raw)
        resource.budget.claim_diagnostic(resource.ledger, intent)
        def before_egress() -> None:
            if operation.operation == "generation":
                self.generation_attempts += 1
        try:
            reply = self.transport.send(operation, raw, before_egress)
        except (TimeoutError, httpx.TransportError):
            response, status, provider_id = None, None, None
        else:
            response = persist_blob(resource.exchange_root / f"{intent.attempt_id}.response", reply.body)
            status, provider_id = reply.status_code, reply.provider_request_id
        evidence = DiagnosticOperationEvidence(provenance=self.transport.provenance, operation=operation,
            intent=intent, request=request, response=response, status_code=status, provider_request_id=provider_id)
        return evidence

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        try:
            return self._chat(messages, model, config)
        except (LiveDiagnosticError, BudgetError, ValidationError, OSError) as error:
            raise DiagnosticStop(str(error)) from error

    def _chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        resource = self.resources
        position = len(self.recorder.get_records())
        if position >= len(self.plans) or model != resource.context.contract.model:
            raise DiagnosticStop("DIAGNOSTIC_SCHEDULE_MISMATCH")
        plan = self.plans[position]
        if config.get("method_stage") != plan.stage:
            raise DiagnosticStop("DIAGNOSTIC_SCHEDULE_MISMATCH")
        material = RequestMaterialV3(messages=tuple(MessageV3.model_validate(row) for row in messages),
            native_state=resource.scope.state_utf8.encode(), temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0))
        try:
            compiled = resource.context.compile(plan.request_id, material)
            counted_operation, generation_operation = resource.context.price_operations(compiled, resource.unit.unit_id)
            for operation in (counted_operation, generation_operation):
                quote = next(row for row in resource.budget.authority.pricing.entries if row.price_id == operation.price_id)
                if quote.evidence != resource.context.contract.rate_card:
                    raise LiveDiagnosticError("DIAGNOSTIC_RATE_CARD_QUOTE_MISMATCH")
        except (LiveDiagnosticError, BudgetError) as error:
            raise DiagnosticStop(str(error)) from error
        card = rates(resource.context)
        if card.count_micros > counted_operation.bound_usd_micros:
            raise DiagnosticStop("DIAGNOSTIC_COUNT_BOUND_INSUFFICIENT")
        counted = self.send(counted_operation, compiled.count_bytes)
        count_pin = persist_blob(resource.exchange_root / f"{counted.intent.attempt_id}.json", counted.model_dump_json().encode())
        count_cost = None if counted.response is None else rates(resource.context).count_micros
        count_valid = counted.status_code == 200 and counted.response is not None
        from .phase13_diagnostic_runtime_contract import _read
        count = None
        if count_valid and counted.response is not None:
            try:
                count = DiagnosticCount.model_validate_json(_read(counted.response))
            except ValueError:
                count_valid = False
        resource.ledger.record_request(resource.unit.unit_id, DiagnosticRequestReceipt(intent=counted.intent,
            receipt_sha256=count_pin.sha256, realized_cost_usd_micros=count_cost,
            outcome="completed" if count_valid else "ambiguous"))
        if not count_valid or count is None:
            raise DiagnosticStop("DIAGNOSTIC_COUNT_UNRESOLVED")
        stage = TypeAdapter(Stage).validate_python(plan.stage)
        if count.input_tokens > STAGES[stage][0]:
            raise DiagnosticStop("INPUT_ENVELOPE_EXCEEDED_ZERO_GENERATION_ATTEMPTS")
        bound = Decimal(card.input_micros_per_token) * count.input_tokens + Decimal(card.output_micros_per_token) * STAGES[stage][1]
        if bound > generation_operation.bound_usd_micros:
            raise DiagnosticStop("DIAGNOSTIC_GENERATION_BOUND_INSUFFICIENT")
        generation = self.send(generation_operation, compiled.wire_bytes)
        self.active = generation
        exchange = DiagnosticExchangeEvidence(generation=generation, count=count_pin,
                                               native_state_sha256=compiled.native_state_sha256)
        exchange_pin = persist_blob(resource.exchange_root / f"{generation.intent.attempt_id}.json", exchange.model_dump_json().encode())
        response = parsed_generation(exchange, resource.context, stage)
        amount = generation_cost(response, resource.context)
        usage = None if response is None else response.usage
        metadata = {"attempts": 1, "status": None if response is None else response.status,
            "response_id": None if response is None else response.id,
            "model": None if response is None else response.model,
            "service_tier": None if response is None else response.service_tier,
            "usage": None if usage is None else usage.model_dump(),
            "cost_usd": None if amount is None else amount / 1000000,
            "derived_cost_usd": None if amount is None else amount / 1000000,
            "cost_source": None if amount is None else "DERIVED_FROM_PROVIDER_USAGE",
            "request_contract": DiagnosticCallContract(exchange=exchange_pin).model_dump(mode="json"),
            "authority_contract": resource.context.contract.authority.model_dump(mode="json", exclude={"evidence"}),
            "incomplete_reason": None if response is None or response.incomplete_details is None else response.incomplete_details.reason}
        result = LLMResponse("" if response is None else response.output_text, metadata,
            {} if usage is None else {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens}, 0)
        if response is None or response.status != "completed":
            raise Phase13ProviderCallError(DiagnosticProviderFailure(result, "DIAGNOSTIC_PROVIDER_NOT_COMPLETED"))
        return result

    def record(self, event: CallEvent) -> None:
        call = self.recorder.get_records()[-1]
        if self.active is None or event.call_id != self.plans[len(self.recorder.get_records()) - 1].native_call_id:
            raise LiveDiagnosticError("DIAGNOSTIC_RECORDER_IDENTITY_MISMATCH")
        pin = persist_blob(self.resources.exchange_root / f"{self.active.intent.attempt_id}.call.json",
            json.dumps(call.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode())
        self.method_call_files.append(pin)
        self.resources.ledger.record_request(self.resources.unit.unit_id, DiagnosticRequestReceipt(
            intent=self.active.intent, receipt_sha256=call_receipt_sha256(call),
            realized_cost_usd_micros=live_call_cost(call), outcome="failed" if call.error_type else "completed"))
        self.active = None
