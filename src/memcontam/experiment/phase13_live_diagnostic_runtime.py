from __future__ import annotations

from inspect import getfile
from pathlib import Path

from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.readiness.phase13_diagnostic_admission import (
    AdmittedDiagnosticExecution,
    DiagnosticRuntimeResources,
    admit_live,
    admit_qa,
)
from memcontam.readiness.phase13_diagnostic_exchange import (
    DiagnosticRuntimeHold,
    DiagnosticStop,
    persist_blob,
)
from memcontam.readiness.phase13_diagnostic_transport import open_live_transport
from memcontam.readiness.phase13_live_diagnostic_evidence import live_call_cost
from memcontam.readiness.phase13_live_diagnostic_models import (
    LiveDiagnosticError,
    LiveDiagnosticOutput,
)
from memcontam.readiness.phase13_main_live_runtime_support import dispatch_output
from memcontam.readiness.phase13_production_runtime_join import production_archive_from_ordinary

from .phase12.runtime_registry import RuntimeStateError
from .phase13_ordinary_runtime import ProspectiveOrdinaryRun, execute_prospective_ordinary


def runtime_source_paths() -> set[Path]:
    from memcontam.readiness import (
        phase13_diagnostic_admission,
        phase13_diagnostic_client,
        phase13_diagnostic_exchange,
        phase13_diagnostic_exchange_validate,
        phase13_diagnostic_native_join,
        phase13_diagnostic_runtime_contract,
        phase13_diagnostic_transport,
        phase13_live_diagnostic,
        phase13_live_diagnostic_evidence,
        phase13_live_diagnostic_models,
        phase13_main_live_dispatch,
        phase13_main_live_evidence,
        phase13_main_runner_store,
        phase13_shared_budget_authority,
        phase13_shared_budget_history,
        phase13_shared_budget_models,
        phase13_shared_provider_budget,
        phase13_v3_request,
    )
    consumed = {Path(__file__).resolve(), *(Path(getfile(module)).resolve() for module in (
        phase13_diagnostic_admission, phase13_diagnostic_client, phase13_diagnostic_exchange,
        phase13_diagnostic_exchange_validate, phase13_diagnostic_runtime_contract, phase13_diagnostic_transport, phase13_diagnostic_native_join,
        phase13_live_diagnostic, phase13_live_diagnostic_evidence, phase13_live_diagnostic_models,
        phase13_main_runner_store, phase13_main_live_dispatch, phase13_main_live_evidence, phase13_v3_request,
        phase13_shared_budget_authority, phase13_shared_budget_history, phase13_shared_budget_models, phase13_shared_provider_budget,
    ))}
    package = Path(__file__).resolve().parents[1]
    consumed.update(package / relative for relative in (
        "clients/base.py", "clients/recording.py", "logging/schema.py", "logging/schema_v3.py",
        "memory/checkpoint_v3.py", "memory/cards_v3.py", "memory/embeddings.py", "memory/stores.py",
        "memory/bot_buffer.py", "tasks/base.py", "tasks/dispatch.py",
        "verifiers/game24.py", "verifiers/math_equation_balancer.py", "verifiers/word_sorting.py",
        "readiness/phase13_production_runtime_join.py", "readiness/phase13_production_runtime_evidence.py",
        "readiness/phase13_production_runtime_models.py", "readiness/phase13_production_observability.py",
        "evaluation/phase13_observability_registration.py", "readiness/phase13_route_capacity.py",
    ))
    return consumed


def execute_diagnostic(run: ProspectiveOrdinaryRun) -> LiveDiagnosticOutput | DiagnosticRuntimeHold:
    context = run.diagnostic_context
    if type(context) is not AdmittedDiagnosticExecution:
        raise LiveDiagnosticError("DIAGNOSTIC_ADMISSION_REQUIRED")
    context.validate_run(run)
    resource = context.resources
    if not runtime_source_paths() <= {Path(pin.path) for pin in resource.context.binding.source_files}:
        raise LiveDiagnosticError("DIAGNOSTIC_RUNTIME_SOURCE_MISMATCH")
    resource.ledger.claim(resource.unit.unit_id)
    try:
        ordinary = execute_prospective_ordinary(run)
    except (DiagnosticStop, RuntimeStateError) as error:
        hold = DiagnosticRuntimeHold(code=str(error), generation_attempts=context.engine.generation_attempts,
                                     unit_id=resource.unit.unit_id, method_calls=tuple(context.engine.method_call_files))
        persist_blob(resource.exchange_root / f"{resource.unit.unit_id}.hold.json", hold.model_dump_json().encode())
        return hold
    trial = ordinary.trials[0]
    if tuple(trial.outcome.method_calls) != tuple(context.execution_client.get_records()):
        raise LiveDiagnosticError("DIAGNOSTIC_NATIVE_RECORDER_MISMATCH")
    archive = production_archive_from_ordinary(run, ordinary, resource.scope.archive_identity)
    output = dispatch_output(resource.unit, ordinary.trials, resource.scope.archive_identity, archive, realized_cost_krw=0)
    raw = output.evidence
    if not isinstance(raw, dict) or not isinstance(trial.state_after, NativeState) or run.branch is None:
        raise LiveDiagnosticError("DIAGNOSTIC_NATIVE_RESULT_MISMATCH")
    raw.update(diagnostic_scope=resource.scope.model_dump(mode="json"),
               diagnostic_post_state_utf8=serialize_checkpoint(trial.state_after).canonical_bytes.decode())
    partial = trial.outcome.status == "failed"
    if partial:
        raw["verifier_result"] = None
    checkpoint = run.branch.checkpoint
    costs = tuple(live_call_cost(call) for call in output.provider_calls)
    live = LiveDiagnosticOutput(evidence={"evidence_kind": "MEMORY_BEARING", "prefix_unit_id": resource.unit.prefix_unit_id,
        "consumed_checkpoint_id": checkpoint.identity.checkpoint_id,
        "consumed_checkpoint_identity_sha256": checkpoint.identity.sha256,
        "consumed_checkpoint_canonical_sha256": checkpoint.canonical_sha256, "runtime_evidence": raw},
        provider_calls=output.provider_calls, realized_cost_usd_micros=None if None in costs else sum(value for value in costs if value is not None),
        disposition="TERMINAL_FAILURE" if partial else "COMPLETED")
    digest = resource.ledger.persist_output(resource.unit.unit_id, live)
    resource.ledger.reconcile(resource.unit.unit_id, digest)
    return live


__all__ = ["DiagnosticRuntimeResources", "admit_live", "admit_qa", "execute_diagnostic", "open_live_transport", "runtime_source_paths"]
