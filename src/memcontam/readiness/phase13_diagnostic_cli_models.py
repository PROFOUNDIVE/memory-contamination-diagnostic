from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from memcontam.readiness.phase13_diagnostic_trusted_admission import AdmissionEvidence
from memcontam.tasks.base import TaskInstance

from .phase13_diagnostic_runtime_contract import DiagnosticExecutionContext
from .phase13_live_diagnostic_models import (
    DiagnosticModel,
    LiveDiagnosticBinding,
    LiveDiagnosticError,
    LiveDiagnosticScope,
)
from .phase13_main_production import ProductionObject
from .phase13_shared_budget_models import BudgetOperation, FilePin


class BranchInput(DiagnosticModel):
    arm: Literal["clean", "correct", "irrelevant", "contam"]
    prefix_identity: str
    source_identity: str
    checkpoint_index: int = Field(ge=0)
    root_count: Literal[0, 1]
    injected_root_id: str | None
    candidate_triplet_id: str | None
    native_render_id: str | None


class NativeInput(DiagnosticModel):
    tasks: tuple[TaskInstance, ...] = Field(min_length=1, max_length=1)
    task_name: Literal["game24", "math_equation_balancer", "word_sorting"]
    baseline: Literal["bot_style", "dc_rs"]
    arm: Literal["clean", "correct", "irrelevant", "contam"]
    model: str
    decoding: dict[str, JsonValue]
    baseline_configs: dict[str, dict[str, JsonValue]]
    trajectory_seed: int
    branch: BranchInput


class ChildRoot(DiagnosticModel):
    child_execution_id: str = Field(min_length=1)
    run_root: str = Field(min_length=1)


class RegistrationSubject(DiagnosticModel):
    schema_version: Literal["phase13_diagnostic_cli_subject_v1"]
    registration_path: str
    parent_root: str
    runtime_registration: FilePin
    binding: LiveDiagnosticBinding
    scopes: tuple[LiveDiagnosticScope, ...] = Field(min_length=16, max_length=16)
    children: tuple[ChildRoot, ...] = Field(min_length=2, max_length=2)
    operations: tuple[BudgetOperation, ...] = Field(min_length=80, max_length=80)
    embedding_cache: str | None

    def context(self) -> DiagnosticExecutionContext:
        if self.runtime_registration.sha256 != self.binding.registration_sha256:
            raise LiveDiagnosticError("DIAGNOSTIC_REGISTRATION_MISMATCH")
        return DiagnosticExecutionContext.load(self.binding, Path(self.runtime_registration.path))

    def units(self) -> tuple[ProductionObject, ...]:
        units = []
        for sequence, scope in enumerate(self.scopes):
            native = NativeInput.model_validate_json(scope.input_utf8)
            identity = scope.archive_identity
            units.append(ProductionObject(sequence=sequence,
                unit_id=hashlib.sha256((scope.run_id + scope.sha256()).encode()).hexdigest(),
                kind="MEMORY_BEARING", seed=native.trajectory_seed, task=native.task_name,
                memory_baseline=native.baseline, arm=native.arm,
                prefix_unit_id=hashlib.sha256(native.branch.prefix_identity.encode()).hexdigest(),
                projected_cost_krw=0, execution_template_id=identity.execution_template_id,
                ordered_sample_ids_sha256=identity.ordered_sample_ids_sha256,
                registration_packet_sha256=identity.registration_packet_sha256,
                checkpoint_registry_sha256=scope.state_sha256, diagnostic_trial_count=1,
                diagnostic_scope_sha256=scope.sha256()))
        return tuple(units)

    def verify(self) -> DiagnosticExecutionContext:
        context = self.context()
        parent = Path(self.parent_root)
        if (parent != parent.resolve() or Path(self.binding.ledger_path) != parent / "ledger.sqlite3"
            or Path(self.registration_path) != Path(self.registration_path).resolve()
            or len(self.binding.observations) != 16
            or len({child.child_execution_id for child in self.children}) != 2
            or len({child.run_root for child in self.children}) != 2):
            raise LiveDiagnosticError("DIAGNOSTIC_PARENT_REGISTRATION_MISMATCH")
        for child in self.children:
            root = Path(child.run_root)
            rows = [scope for scope in self.scopes if scope.child_execution_id == child.child_execution_id]
            if (root != root.resolve() or root.parent != parent or len(rows) != 8
                or len({NativeInput.model_validate_json(scope.input_utf8).baseline for scope in rows}) != 1):
                raise LiveDiagnosticError("DIAGNOSTIC_CHILD_REGISTRATION_MISMATCH")
        if {NativeInput.model_validate_json(scope.input_utf8).baseline for scope in self.scopes} != {"bot_style", "dc_rs"}:
            raise LiveDiagnosticError("DIAGNOSTIC_BASELINE_SCHEDULE_MISMATCH")
        pins = {pin.path: pin.sha256 for pin in self.binding.source_files}
        for scope, observation, unit in zip(self.scopes, self.binding.observations, self.units(), strict=True):
            if (scope.binding_sha256 != self.binding.sha256() or scope.scientific_disposition != self.binding.scientific_disposition
                or (scope.run_id, scope.child_execution_id, scope.input_sha256, scope.state_sha256) != (
                    observation.run_id, observation.child_execution_id, observation.input_sha256, observation.state_sha256)
                or hashlib.sha256(scope.input_utf8.encode()).hexdigest() != scope.input_sha256
                or hashlib.sha256(scope.state_utf8.encode()).hexdigest() != scope.state_sha256
                or any(pins.get(pin.path) != pin.sha256 for pin in scope.source_files)):
                raise LiveDiagnosticError("DIAGNOSTIC_SCOPE_MISMATCH")
            if scope.candidate is not None:
                FilePin(path=scope.candidate.path, sha256=scope.candidate.sha256).read()
            operations = [operation for operation in self.operations if operation.unit_id == unit.unit_id]
            if tuple((row.request_id, row.operation) for row in operations) != tuple(
                (row.request_id, row.operation) for row in observation.requests
            ) or any(row.run_id != scope.run_id or row.execution_binding_sha256 != self.binding.sha256()
                     or row.attempt_id != hashlib.sha256((self.binding.sha256() + row.request_id).encode()).hexdigest()
                     for row in operations):
                raise LiveDiagnosticError("DIAGNOSTIC_OPERATION_SCHEDULE_MISMATCH")
        return context


class RegisteredDiagnostic(DiagnosticModel):
    subject: RegistrationSubject
    admission: AdmissionEvidence
