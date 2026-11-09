from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from memcontam.logging.schema import MethodCall

from .phase13_authority_files import read_regular_nofollow
from .phase13_durable_call_join import validate_native_diagnostic_join
from .phase13_live_diagnostic_models import (
    LiveDiagnosticBinding, LiveDiagnosticError, LiveDiagnosticEvidence,
    LiveDiagnosticOutput, LiveDiagnosticScope,
)
from .phase13_main_live_evidence import MemoryUnitEvidence, _completed_call, _parse_evidence, _reconciled_cost
from .phase13_main_production import ProductionObject


def verify_live_sources(binding: LiveDiagnosticBinding, units: tuple[ProductionObject, ...]) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_source_paths

    required = {Path(__file__).with_name(name) for name in (
        "phase13_live_diagnostic.py", "phase13_live_diagnostic_models.py", "phase13_live_diagnostic_evidence.py",
        "phase13_main_runner_store.py", "phase13_main_live_dispatch.py", "phase13_main_live_evidence.py",
        "phase13_durable_call_join.py")}
    for unit in units:
        if unit.memory_baseline not in {"bot_style", "dc_rs"}:
            raise LiveDiagnosticError("DIAGNOSTIC_SOURCE_MISMATCH")
        required.update(snapshot_source_paths(unit.memory_baseline))
    if not required <= {Path(pin.path) for pin in binding.source_files}:
        raise LiveDiagnosticError("DIAGNOSTIC_SOURCE_MISMATCH")
    for pin in binding.source_files:
        path = Path(pin.path)
        if path != path.resolve() or hashlib.sha256(read_regular_nofollow(path)).hexdigest() != pin.sha256:
            raise LiveDiagnosticError("DIAGNOSTIC_SOURCE_MISMATCH")


def live_call_cost(call: MethodCall) -> int | None:
    if call.provider_cost_usd is None:
        if any(value is not None for value in (call.authoritative_provider_cost_usd,
                                               call.derived_cost_usd, call.provider_cost_source)):
            raise LiveDiagnosticError("DIAGNOSTIC_COST_CONTRADICTION")
        return None
    micros = Decimal(str(call.provider_cost_usd)) * 1000000
    if (not micros.is_finite() or micros <= 0 or micros != micros.to_integral_value()
        or not _reconciled_cost(call) or not call.provider_usage):
        raise LiveDiagnosticError("DIAGNOSTIC_COST_CONTRADICTION")
    return int(micros)


def validate_live_output(unit: ProductionObject, output: LiveDiagnosticOutput,
                         binding: LiveDiagnosticBinding) -> MemoryUnitEvidence:
    evidence = _parse_evidence(unit, output.evidence)
    if not isinstance(evidence, MemoryUnitEvidence):
        raise LiveDiagnosticError("DIAGNOSTIC_SCOPE_INVALID")
    scope = evidence.runtime_evidence.diagnostic_scope
    if not isinstance(scope, LiveDiagnosticScope):
        raise LiveDiagnosticError("DIAGNOSTIC_CROSS_SCOPE")
    observation = next((row for row in binding.observations if row.run_id == scope.run_id), None)
    pins = {pin.path: pin.sha256 for pin in binding.source_files}
    if (scope.binding_sha256 != binding.sha256() or observation is None
        or scope.child_execution_id != observation.child_execution_id
        or scope.input_sha256 != observation.input_sha256 or scope.state_sha256 != observation.state_sha256
        or scope.scientific_disposition != binding.scientific_disposition
        or any(pins.get(pin.path) != pin.sha256 for pin in scope.source_files)
        or unit.projected_cost_krw != 0):
        raise LiveDiagnosticError("DIAGNOSTIC_BINDING_MISMATCH")
    partial = output.disposition == "TERMINAL_FAILURE"
    from .phase13_diagnostic_exchange import selected_context
    from .phase13_diagnostic_exchange_validate import validate_exchange
    context = selected_context(binding)
    validate_native_diagnostic_join(unit, evidence, output.provider_calls, scope=scope, partial=partial,
                                    registered_live=context is not None)
    expected_calls = tuple(request.native_call_id for request in observation.requests
                           if request.operation == "generation")
    actual_calls = tuple(call.call_id for call in output.provider_calls)
    if actual_calls != (expected_calls[:len(actual_calls)] if partial else expected_calls):
        raise LiveDiagnosticError("DIAGNOSTIC_REQUEST_JOIN_INVALID")
    completed = output.provider_calls[:-1] if partial else output.provider_calls
    if context is not None:
        for call in output.provider_calls:
            validate_exchange(call, context, scope)
    elif any(not _completed_call(call, evidence.runtime_evidence.request, None) for call in completed):
        raise LiveDiagnosticError("DIAGNOSTIC_CALL_INVALID")
    costs = tuple(live_call_cost(call) for call in output.provider_calls)
    realized = None if None in costs else sum(cost for cost in costs if cost is not None)
    if (output.realized_cost_usd_micros != realized
        or (not partial and any(cost is None for cost in costs))):
        raise LiveDiagnosticError("DIAGNOSTIC_COST_CONTRADICTION")
    return evidence


def load_live_evidence(path: Path, evidence_sha256: str) -> LiveDiagnosticEvidence:
    raw = read_regular_nofollow(path)
    if hashlib.sha256(raw).hexdigest() != evidence_sha256:
        raise LiveDiagnosticError("DIAGNOSTIC_OUTER_HASH_MISMATCH")
    return LiveDiagnosticEvidence.model_validate_json(raw)


def call_receipt_sha256(call: MethodCall) -> str:
    return hashlib.sha256(json.dumps(call.model_dump(mode="json"), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
