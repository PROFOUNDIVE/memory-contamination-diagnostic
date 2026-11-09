from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from memcontam.clients.recording import MethodCallRecorder
from memcontam.memory.checkpoint_v3 import NativeState, deserialize_checkpoint, serialize_checkpoint

from .phase13_diagnostic_client import DiagnosticClient
from .phase13_diagnostic_exchange import (
    DiagnosticTransport,
    canonical_hash,
    rates,
    selected_context,
)
from .phase13_diagnostic_runtime_contract import DiagnosticExecutionContext, _read
from .phase13_live_diagnostic import LiveDiagnosticLedger
from .phase13_live_diagnostic_evidence import verify_live_sources
from .phase13_live_diagnostic_models import LiveDiagnosticError, LiveDiagnosticScope
from .phase13_main_production import ProductionObject
from .phase13_shared_budget_models import FilePin
from .phase13_shared_provider_budget import SharedProviderBudget

if TYPE_CHECKING:
    from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun

    from .phase13_diagnostic_transport import DiagnosticLaunch


@dataclass(frozen=True, slots=True)
class DiagnosticRuntimeResources:
    context: DiagnosticExecutionContext
    scope: LiveDiagnosticScope
    unit: ProductionObject
    ledger: LiveDiagnosticLedger
    budget: SharedProviderBudget
    exchange_root: Path

    def verify(self) -> None:
        if type(self.context) is not DiagnosticExecutionContext:
            raise LiveDiagnosticError("DIAGNOSTIC_ADMISSION_REQUIRED")
        self.context.verify()
        if (selected_context(self.context.binding) != self.context
            or self.ledger.binding != self.context.binding or self.unit not in self.ledger.units
            or self.scope.binding_sha256 != self.context.binding.sha256()
            or self.scope.sha256() != self.unit.diagnostic_scope_sha256
            or self.unit.unit_id != hashlib.sha256((self.scope.run_id + self.scope.sha256()).encode()).hexdigest()
            or self.budget.binding_path != Path(self.context.contract.budget_binding.path)):
            raise LiveDiagnosticError("DIAGNOSTIC_ADMISSION_BINDING_MISMATCH")
        observation = self.context.binding.observations[self.unit.sequence]
        if (self.scope.run_id, self.scope.child_execution_id, self.scope.input_sha256, self.scope.state_sha256) != (
            observation.run_id, observation.child_execution_id, observation.input_sha256, observation.state_sha256
        ):
            raise LiveDiagnosticError("DIAGNOSTIC_ADMISSION_BINDING_MISMATCH")
        verify_live_sources(self.context.binding, self.ledger.units)
        if self.scope.candidate is not None:
            _read(FilePin(path=self.scope.candidate.path, sha256=self.scope.candidate.sha256))
        rates(self.context)
        if not self.budget.status().paid_eligible:
            raise LiveDiagnosticError("DIAGNOSTIC_BUDGET_HELD")


    @property
    def prefix_history(self):
        from memcontam.experiment.phase13_dc_rs_validation import BoundOrdinaryPrefix
        records = [row for row in self.context.contract.prefix_histories if row.run_id == self.scope.run_id]
        if not records:
            return None
        if len(records) != 1 or _read(records[0].state) != self.scope.state_utf8.encode():
            raise LiveDiagnosticError("DIAGNOSTIC_PREFIX_IDENTITY_MISMATCH")
        state = NativeState.from_mapping(json.loads(self.scope.state_utf8))
        return BoundOrdinaryPrefix(records[0].source_run_id, records[0].checkpoint_index,
            tuple((row["entry_id"], canonical_hash(row)) for row in state.native_state["archive"]))

    def validate_inputs(self, run: ProspectiveOrdinaryRun) -> None:
        from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
        from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_input_utf8

        from .phase13_main_live_evidence import _BASELINE_STAGES

        self.verify()
        scope = self.scope
        if (run.baseline not in {"bot_style", "dc_rs"} or run.branch is None or len(run.tasks) != 1
            or run.initial_states or run.core_bundle is not None or run.run_id != scope.run_id
            or snapshot_input_utf8(run) != scope.input_utf8
            or hashlib.sha256(scope.input_utf8.encode()).hexdigest() != scope.input_sha256
            or run.branch.checkpoint.canonical_bytes != scope.state_utf8.encode()
            or run.branch.checkpoint.canonical_sha256 != scope.state_sha256
            or scope.archive_identity.scientific_result
            or run.trajectory_seed != scope.archive_identity.trajectory_seed
            or run.embedding_provider is None
            or canonical_hash(run.embedding_provider.metadata) != self.context.contract.embedding_metadata_sha256):
            raise LiveDiagnosticError("DIAGNOSTIC_NATIVE_INPUT_MISMATCH")
        deserialize_checkpoint(run.branch.checkpoint)
        history = self.prefix_history
        if history is not None and history.checkpoint_index != run.branch.checkpoint.checkpoint_index:
            raise LiveDiagnosticError("DIAGNOSTIC_PREFIX_IDENTITY_MISMATCH")
        state = PHASE13_CORE_BASELINE_REGISTRY[run.baseline].serialize_state(run.branch.state)
        if not isinstance(state, NativeState) or serialize_checkpoint(state).canonical_sha256 != scope.state_sha256:
            raise LiveDiagnosticError("DIAGNOSTIC_NATIVE_STATE_MISMATCH")
        sample_hash = hashlib.sha256(json.dumps((run.tasks[0].sample_id,), separators=(",", ":")).encode()).hexdigest()
        if (scope.archive_identity.ordered_sample_ids_sha256 != sample_hash
            or scope.archive_identity.execution_template_id != f"{run.task_name}:{run.baseline}:{run.arm}"):
            raise LiveDiagnosticError("DIAGNOSTIC_SAMPLE_ORDER_MISMATCH")
        if ((self.unit.task, self.unit.memory_baseline, self.unit.arm, self.unit.seed) != (
                run.task_name, run.baseline, run.arm, run.trajectory_seed)
            or self.unit.ordered_sample_ids_sha256 != sample_hash
            or self.unit.checkpoint_registry_sha256 != scope.state_sha256
            or self.unit.registration_packet_sha256 != scope.archive_identity.registration_packet_sha256
            or self.unit.prefix_unit_id != hashlib.sha256(run.branch.prefix_identity.encode()).hexdigest()):
            raise LiveDiagnosticError("DIAGNOSTIC_NATIVE_INPUT_MISMATCH")
        plans = self.context.binding.observations[self.unit.sequence].requests
        stages = _BASELINE_STAGES[run.baseline]
        if tuple((request.operation, request.stage) for request in plans) != tuple(
            (operation, stage) for stage in stages for operation in ("count", "generation")
        ):
            raise LiveDiagnosticError("DIAGNOSTIC_NATIVE_SCHEDULE_MISMATCH")


@dataclass(frozen=True, slots=True)
class AdmittedDiagnosticExecution:
    resources: DiagnosticRuntimeResources
    execution_client: MethodCallRecorder
    engine: DiagnosticClient

    @property
    def common_capacity_tokens(self) -> int:
        return self.resources.context.contract.common_capacity_tokens

    @property
    def prefix_history(self):
        return self.resources.prefix_history

    def validate_run(self, run: ProspectiveOrdinaryRun) -> None:
        if (type(self.resources) is not DiagnosticRuntimeResources or type(self.execution_client) is not MethodCallRecorder
            or run.allow_test_client or run.validated_resources is not None or run.production_identity is not None
            or run.client is not self.execution_client or self.execution_client is not self.engine.recorder
            or self.engine.resources is not self.resources or type(self.engine) is not DiagnosticClient):
            raise LiveDiagnosticError("DIAGNOSTIC_MAIN_OR_CLIENT_SCOPE_MISMATCH")
        self.resources.validate_inputs(run)
        qualification = self.resources.context.contract.qualification
        if qualification == "QA_ONLY_NOT_PROVIDER_QUALIFIED":
            if self.engine.transport.provenance != "QA_IN_PROCESS":
                raise LiveDiagnosticError("DIAGNOSTIC_TRANSPORT_SCOPE_MISMATCH")
        else:
            from .phase13_diagnostic_transport import DiagnosticHttpTransport
            transport = self.engine.transport
            if (type(transport) is not DiagnosticHttpTransport or transport.context != self.resources.context
                or transport.launch.child_execution_id != self.resources.scope.child_execution_id):
                raise LiveDiagnosticError("DIAGNOSTIC_TRANSPORT_SCOPE_MISMATCH")
            transport.verify_admission()


def admit_qa(run: ProspectiveOrdinaryRun, resources: DiagnosticRuntimeResources,
             transport: DiagnosticTransport) -> AdmittedDiagnosticExecution:
    if (type(resources) is not DiagnosticRuntimeResources
        or resources.context.contract.qualification != "QA_ONLY_NOT_PROVIDER_QUALIFIED"
        or transport.provenance != "QA_IN_PROCESS"):
        raise LiveDiagnosticError("DIAGNOSTIC_QA_ADMISSION_REQUIRED")
    resources.validate_inputs(run)
    engine = DiagnosticClient(resources, transport)
    admitted = AdmittedDiagnosticExecution(resources, engine.recorder, engine)
    return admitted


def admit_live(build_run: Callable[[AdmittedDiagnosticExecution], ProspectiveOrdinaryRun],
               resources: DiagnosticRuntimeResources, launch: DiagnosticLaunch | None) -> ProspectiveOrdinaryRun:
    from .phase13_diagnostic_transport import DiagnosticHttpTransport
    if type(resources) is not DiagnosticRuntimeResources:
        raise LiveDiagnosticError("DIAGNOSTIC_ADMISSION_REQUIRED")
    resources.verify()
    if launch is None or launch.child_execution_id != resources.scope.child_execution_id:
        raise LiveDiagnosticError("DIAGNOSTIC_LIVE_ADMISSION_UNESTABLISHED")
    transport = DiagnosticHttpTransport(resources.context, launch)
    engine = DiagnosticClient(resources, transport)
    admitted = AdmittedDiagnosticExecution(resources, engine.recorder, engine)
    run = build_run(admitted)
    if run.diagnostic_context is not admitted:
        raise LiveDiagnosticError("DIAGNOSTIC_ADMISSION_REQUIRED")
    admitted.validate_run(run)
    transport.open()
    return run
