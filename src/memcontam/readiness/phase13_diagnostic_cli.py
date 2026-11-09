from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict, assert_never

from memcontam.experiment.phase13_live_diagnostic_runtime import (
    execute_diagnostic,
    runtime_source_paths,
)
from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_source_paths
from memcontam.readiness import (
    phase13_diagnostic_cli_models,
    phase13_diagnostic_cli_native,
    phase13_diagnostic_intake,
    phase13_diagnostic_proposal,
    phase13_diagnostic_proposal_models,
    phase13_diagnostic_trusted_admission,
)

from .phase13_authority_files import read_regular_nofollow
from .phase13_diagnostic_admission import DiagnosticRuntimeResources, admit_live
from .phase13_diagnostic_cli_models import RegisteredDiagnostic
from .phase13_diagnostic_exchange import rates
from .phase13_diagnostic_runtime_contract import DiagnosticExecutionContext
from .phase13_diagnostic_transport import DiagnosticHttpTransport, DiagnosticLaunch
from .phase13_live_diagnostic import LiveDiagnosticLedger
from .phase13_live_diagnostic_models import (
    DiagnosticRequestIntent,
    DiagnosticStatus,
    LiveDiagnosticError,
)
from .phase13_main_live_dispatch import persist_evidence_bytes
from .phase13_main_live_evidence import _BASELINE_STAGES
from .phase13_main_runner_store import connect
from .phase13_shared_budget_authority import BudgetAuthority
from .phase13_shared_provider_budget import SharedProviderBudget


class PreflightResult(TypedDict):
    observations: int
    operations: int
    subject_sha256: str
    purpose: str
    provider_calls: int


def cli_source_paths() -> set[Path]:
    return {Path(__file__).resolve(), Path(phase13_diagnostic_cli_models.__file__).resolve(),
             Path(phase13_diagnostic_cli_native.__file__).resolve(),
             Path(phase13_diagnostic_intake.__file__).resolve(),
             Path(phase13_diagnostic_proposal.__file__).resolve(),
             Path(phase13_diagnostic_proposal_models.__file__).resolve(),
            Path(phase13_diagnostic_trusted_admission.__file__).resolve(),
            Path(__file__).resolve().parents[3] / "scripts/run_phase13_live_diagnostic.py"}


def load_registration(path: Path) -> RegisteredDiagnostic:
    registered = RegisteredDiagnostic.model_validate_json(read_regular_nofollow(path))
    registered.subject.verify()
    return registered


def check_admission(registered: RegisteredDiagnostic) -> DiagnosticExecutionContext:
    subject = registered.subject
    context = subject.verify()
    authority = BudgetAuthority.load(Path(context.contract.budget_binding.path))
    if authority.binding.account_id is None or not authority.reconciliation.complete:
        raise LiveDiagnosticError("DIAGNOSTIC_ACCOUNT_OR_BALANCE_UNESTABLISHED")
    trust = phase13_diagnostic_trusted_admission.authenticate(
        registered.admission, subject.sha256(), authority.binding.account_id)
    if authority.binding.proof_public_key in {trust.science, trust.launch, trust.credential}:
        raise LiveDiagnosticError("DIAGNOSTIC_FINANCIAL_ROLE_TRANSFER")
    required = cli_source_paths() | runtime_source_paths() | snapshot_source_paths("bot_style") | snapshot_source_paths("dc_rs")
    required.add(phase13_diagnostic_trusted_admission.TRUST_POLICY_PATH)
    if (not required <= {Path(pin.path) for pin in subject.binding.source_files}
        or context.contract.qualification != "LIVE_REQUIRES_ADMISSION"
        or context.contract.embedding_metadata_sha256 is None
        or (trust.purpose == "OPERATIONAL" and subject.binding.scientific_disposition != "DIAGNOSTIC_ONLY")):
        raise LiveDiagnosticError("DIAGNOSTIC_LAUNCH_SOURCE_OR_SCIENCE_MISMATCH")
    rates(context)
    for observation, unit in zip(subject.binding.observations, subject.units(), strict=True):
        stages = _BASELINE_STAGES[phase13_diagnostic_cli_models.NativeInput.model_validate_json(subject.scopes[unit.sequence].input_utf8).baseline]
        if tuple((row.operation, row.stage) for row in observation.requests) != tuple(
            (operation, stage) for stage in stages for operation in ("count", "generation")
        ):
            raise LiveDiagnosticError("DIAGNOSTIC_NATIVE_SCHEDULE_MISMATCH")
    for operation in subject.operations:
        authority.check_operation(operation)
    if sum(operation.bound_usd_micros for operation in subject.operations) > authority.binding.cap_usd_micros:
        raise LiveDiagnosticError("DIAGNOSTIC_SCHEDULE_EXCEEDS_CAP")
    return context


def preflight(path: Path) -> PreflightResult:
    registered = load_registration(path)
    check_admission(registered)
    return {"observations": len(registered.subject.scopes), "operations": len(registered.subject.operations),
            "subject_sha256": registered.subject.sha256(), "purpose": registered.admission.science.purpose,
            "provider_calls": 0}


def register(candidate: Path | RegisteredDiagnostic, destination: Path) -> str:
    match candidate:
        case Path():
            raw = read_regular_nofollow(candidate)
        case RegisteredDiagnostic():
            raw = candidate.model_dump_json().encode()
        case unreachable:
            assert_never(unreachable)
    registered = RegisteredDiagnostic.model_validate_json(raw)
    check_admission(registered)
    if destination != Path(registered.subject.registration_path) or destination != destination.resolve():
        raise LiveDiagnosticError("DIAGNOSTIC_REGISTRATION_PATH_MISMATCH")
    return persist_evidence_bytes(destination, raw, "DIAGNOSTIC_REGISTRATION_ALREADY_EXISTS")


def run_registered(registration: Path, child_id: str, api_key: str) -> DiagnosticStatus:
    registered = load_registration(registration)
    context = check_admission(registered)
    subject = registered.subject
    if registration != Path(subject.registration_path):
        raise LiveDiagnosticError("DIAGNOSTIC_REGISTRATION_PATH_MISMATCH")
    child = next((child for child in subject.children if child.child_execution_id == child_id), None)
    if child is None:
        raise LiveDiagnosticError("DIAGNOSTIC_EXACT_CHILD_REQUIRED")
    phase13_diagnostic_trusted_admission.verify_credential(registered.admission, api_key)
    budget = SharedProviderBudget.open(Path(context.contract.budget_binding.path))
    if not budget.status().paid_eligible:
        raise LiveDiagnosticError("DIAGNOSTIC_BUDGET_HELD")
    units = subject.units()
    ledger = (LiveDiagnosticLedger.open(subject.binding, units) if Path(subject.binding.ledger_path).exists()
              else LiveDiagnosticLedger.create(subject.binding, units))
    recover_registered(registration, Path(subject.parent_root))
    with connect(ledger.path) as connection:
        connection.execute("BEGIN")
        states = ledger._validate(connection)
    selected = [(scope, unit) for scope, unit in zip(subject.scopes, units, strict=True)
                if scope.child_execution_id == child_id and states[unit.sequence] != "COMPLETED"]
    if not selected or any(states[unit.sequence] != "PENDING" for _, unit in selected):
        raise LiveDiagnosticError("DIAGNOSTIC_COMPLETED_OR_AMBIGUOUS_REDISPATCH")
    embedding = phase13_diagnostic_cli_native.load_embedding(subject.embedding_cache)
    registration_digest = hashlib.sha256(read_regular_nofollow(registration)).hexdigest()
    subject_digest = subject.sha256()
    account_id = registered.admission.credential.account_id
    planned = {operation.request_id: operation for operation in subject.operations}

    def verify_admission(current: DiagnosticExecutionContext, child_execution_id: str, credential: str) -> None:
        if (current != context or child_execution_id != child_id
            or hashlib.sha256(read_regular_nofollow(registration)).hexdigest() != registration_digest):
            raise LiveDiagnosticError("DIAGNOSTIC_LAUNCH_CHANGED")
        current.verify()
        phase13_diagnostic_trusted_admission.authenticate(registered.admission, subject_digest, account_id)
        phase13_diagnostic_trusted_admission.verify_credential(registered.admission, credential)
        with connect(ledger.path) as connection:
            for unit_id, raw in connection.execute("SELECT unit_id,reason FROM events WHERE kind='REQUEST_INTENT'"):
                intent = DiagnosticRequestIntent.model_validate_json(raw)
                operation = planned.get(intent.request_id)
                if (operation is None or operation.unit_id != unit_id
                    or operation.intent(intent.reservation_id) != intent):
                    raise LiveDiagnosticError("DIAGNOSTIC_UNPLANNED_WIRE_INTENT")

    launch = DiagnosticLaunch(child_id, api_key, verify_admission)
    for scope, unit in selected:
        resources = DiagnosticRuntimeResources(context, scope, unit, ledger, budget, Path(child.run_root) / "exchanges")
        run = admit_live(lambda admitted: phase13_diagnostic_cli_native.build_native(admitted, embedding), resources, launch)
        assert run.diagnostic_context is not None
        transport = run.diagnostic_context.engine.transport
        assert isinstance(transport, DiagnosticHttpTransport)
        try:
            output = execute_diagnostic(run)
        finally:
            transport.close()
        if output.disposition != "COMPLETED":
            break
    return ledger.status()


def verify_registered(registration: Path, run_root: Path) -> DiagnosticStatus:
    registered = load_registration(registration)
    subject = registered.subject
    allowed = {subject.parent_root, *(child.run_root for child in subject.children)}
    if str(run_root) not in allowed or run_root != run_root.resolve():
        raise LiveDiagnosticError("DIAGNOSTIC_RUN_ROOT_MISMATCH")
    return LiveDiagnosticLedger.open(subject.binding, subject.units()).status()


def recover_registered(registration: Path, run_root: Path) -> DiagnosticStatus:
    verify_registered(registration, run_root)
    subject = load_registration(registration).subject
    ledger = LiveDiagnosticLedger.open(subject.binding, subject.units())
    with connect(ledger.path) as connection:
        connection.execute("BEGIN")
        states = ledger._validate(connection)
    for unit in ledger.units:
        evidence = ledger._evidence_path(unit.unit_id)
        if states[unit.sequence] == "DISPATCH_INTENT_PERSISTED" and evidence.is_file():
            ledger.reconcile(unit.unit_id, hashlib.sha256(read_regular_nofollow(evidence)).hexdigest())
    return ledger.status()
