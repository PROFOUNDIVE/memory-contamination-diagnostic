from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import TypeAdapter

from memcontam.logging.schema import MethodCall

from .phase13_diagnostic_exchange import (
    DiagnosticCallContract,
    DiagnosticExchangeEvidence,
    DiagnosticOperationEvidence,
    generation_cost,
    parsed_generation,
    rates,
)
from .phase13_diagnostic_runtime_contract import DiagnosticExecutionContext, _read
from .phase13_live_diagnostic_models import (
    DiagnosticRequestReceipt,
    LiveDiagnosticError,
    LiveDiagnosticScope,
)
from .phase13_main_runner_store import connect
from .phase13_shared_budget_models import FilePin
from .phase13_shared_provider_budget import SharedProviderBudget
from .phase13_v3_request import MessageV3, RequestMaterialV3, Stage


def validate_exchange(call: MethodCall, context: DiagnosticExecutionContext,
                      scope: LiveDiagnosticScope) -> DiagnosticExchangeEvidence:
    contract = DiagnosticCallContract.model_validate(call.provider_request_contract)
    exchange = DiagnosticExchangeEvidence.model_validate_json(_read(contract.exchange))
    from .phase13_live_diagnostic_evidence import call_receipt_sha256
    _read(FilePin(path=str(Path(contract.exchange.path).with_name(f"{exchange.generation.intent.attempt_id}.call.json")),
                  sha256=call_receipt_sha256(call)))
    counted = DiagnosticOperationEvidence.model_validate_json(_read(exchange.count))
    observation = next(row for row in context.binding.observations if row.run_id == scope.run_id)
    matching = [row for row in observation.requests if row.native_call_id == call.call_id and row.operation == "generation"]
    if len(matching) != 1 or matching[0].stage != call.stage:
        raise LiveDiagnosticError("DIAGNOSTIC_EXCHANGE_CALL_MISMATCH")
    material = RequestMaterialV3(messages=tuple(MessageV3.model_validate(row) for row in call.messages),
                                 native_state=scope.state_utf8.encode())
    compiled = context.compile(matching[0].request_id, material)
    generation = exchange.generation
    index = observation.requests.index(matching[0])
    if (index == 0 or observation.requests[index - 1].operation != "count"
        or observation.requests[index - 1].stage != call.stage
        or counted.intent.request_id != observation.requests[index - 1].request_id
        or generation.intent.request_id != compiled.request_id
        or generation.intent.attempt_id != compiled.attempt_id
        or exchange.native_state_sha256 != scope.state_sha256
        or _read(generation.request) != compiled.wire_bytes or _read(counted.request) != compiled.count_bytes
        or generation.intent.request_sha256 != compiled.wire_sha256
        or counted.intent.request_sha256 != compiled.count_sha256
        or call.provider_authority_contract != context.contract.authority.model_dump(mode="json", exclude={"evidence"})
        or call.transport_attempts != 1 or call.retry_count != 0 or call.temperature != 0.0 or call.top_p != 1.0):
        raise LiveDiagnosticError("DIAGNOSTIC_EXCHANGE_REQUEST_MISMATCH")
    budget = SharedProviderBudget.open(Path(context.contract.budget_binding.path))
    for evidence, expected_kind in ((counted, "count"), (generation, "generation")):
        operation = evidence.operation
        if (operation.operation != expected_kind or operation.intent(evidence.intent.reservation_id) != evidence.intent
            or operation.execution_binding_sha256 != context.binding.sha256()
            or operation.run_id != scope.run_id
            or operation.unit_id != hashlib.sha256((scope.run_id + scope.sha256()).encode()).hexdigest()
            or evidence.provenance != ("QA_IN_PROCESS" if context.contract.qualification == "QA_ONLY_NOT_PROVIDER_QUALIFIED" else "PROVIDER")
            or counted.provenance != generation.provenance):
            raise LiveDiagnosticError("DIAGNOSTIC_EXCHANGE_FINANCIAL_MISMATCH")
        budget.authority.check_operation(operation, current=False)
        price = next(row for row in budget.authority.pricing.entries if row.price_id == operation.price_id)
        if price.evidence != context.contract.rate_card:
            raise LiveDiagnosticError("DIAGNOSTIC_EXCHANGE_RATE_CARD_MISMATCH")
        with connect(budget.path) as connection:
            row = connection.execute("SELECT operation,state FROM reservations WHERE reservation_id=?",
                                     (evidence.intent.reservation_id,)).fetchone()
        if row is None or row[0] != operation.model_dump_json() or row[1] not in {"intent", "charge", "ambiguous"}:
            raise LiveDiagnosticError("DIAGNOSTIC_EXCHANGE_FINANCIAL_INTENT_MISSING")
    response = parsed_generation(exchange, context, TypeAdapter(Stage).validate_python(call.stage))
    usage = None if response is None else response.usage
    amount = generation_cost(response, context)
    expected_cost = None if amount is None else amount / 1000000
    status = None if response is None else response.status
    failed = response is None or status != "completed"
    if (call.model != context.contract.model or call.provider_returned_model != (None if response is None else response.model)
        or call.provider_response_id != (None if response is None else response.id)
        or call.provider_status != status or call.provider_response_status != status
        or call.provider_service_tier != (None if response is None else response.service_tier)
        or call.provider_usage != (None if usage is None else usage.model_dump())
        or call.token_usage != ({} if usage is None else {"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens})
        or call.provider_cost_usd != expected_cost or call.derived_cost_usd != expected_cost
        or call.authoritative_provider_cost_usd is not None
        or call.provider_cost_source != (None if amount is None else "DERIVED_FROM_PROVIDER_USAGE")
        or (call.error_type is not None) != failed
        or call.raw_response != (None if failed or response is None else response.output_text)):
        raise LiveDiagnosticError("DIAGNOSTIC_EXCHANGE_RESPONSE_MISMATCH")
    return exchange


def validate_exchange_receipts(exchange: DiagnosticExchangeEvidence,
                               receipts: dict[str, DiagnosticRequestReceipt], context: DiagnosticExecutionContext) -> None:
    counted = DiagnosticOperationEvidence.model_validate_json(_read(exchange.count))
    count = receipts.get(counted.intent.request_id)
    generation = receipts.get(exchange.generation.intent.request_id)
    if (count is None or generation is None or count.intent != counted.intent
        or generation.intent != exchange.generation.intent or count.receipt_sha256 != exchange.count.sha256
        or count.outcome != "completed" or count.realized_cost_usd_micros != rates(context).count_micros):
        raise LiveDiagnosticError("DIAGNOSTIC_EXCHANGE_RECEIPT_MISMATCH")
