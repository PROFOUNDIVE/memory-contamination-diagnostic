"""One-step native snapshot QA; byte bindings confer no scientific authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from inspect import getfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from memcontam.baselines import bot_phase12, bot_read, bot_runtime, bot_solve, bot_write
from memcontam.baselines import dynamic_cheatsheet_optional, dynamic_cheatsheet_phase12
from memcontam.experiment import phase13_dc_rs_runtime, phase13_dc_rs_validation, phase13_ordinary_runtime
from memcontam.experiment.phase12 import runtime_registry
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import (
    ProspectiveOrdinaryResult,
    ProspectiveOrdinaryRun,
    _context,
    execute_prospective_ordinary,
)
from memcontam.memory.checkpoint_v3 import (
    NativeState,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from memcontam.readiness.phase13_production_observability import ProductionObservabilityArchive
from memcontam.readiness.phase13_production_runtime_join import production_archive_from_ordinary
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity

if TYPE_CHECKING:
    from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
    from memcontam.readiness.phase13_main_production import ProductionObject


class SnapshotError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SnapshotFileBinding:
    path: Path
    sha256: str

    def verify(self) -> None:
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise SnapshotError("SNAPSHOT_FILE_BINDING_MISMATCH")


@dataclass(frozen=True, slots=True)
class SnapshotBindings:
    state_sha256: str
    input_sha256: str
    source_files: tuple[SnapshotFileBinding, ...]
    archive_identity: ProductionOrdinaryRunIdentity
    candidate: SnapshotFileBinding | None = None


@dataclass(frozen=True, slots=True)
class SnapshotUnitResult:
    run: ProspectiveOrdinaryRun
    ordinary: ProspectiveOrdinaryResult
    archive: ProductionObservabilityArchive
    bindings: SnapshotBindings
    clone_path: Path
    provenance: Literal["provider_free_qa"] = "provider_free_qa"


def snapshot_input_sha256(run: ProspectiveOrdinaryRun) -> str:
    """Bind the input, treatment and native configuration, excluding clone IDs."""
    return hashlib.sha256(snapshot_input_utf8(run).encode()).hexdigest()


def snapshot_input_utf8(run: ProspectiveOrdinaryRun) -> str:
    branch = run.branch
    payload = {
        "tasks": [task.model_dump(mode="json") for task in run.tasks],
        "task_name": run.task_name,
        "baseline": run.baseline,
        "arm": run.arm,
        "model": run.model,
        "decoding": dict(run.decoding),
        "baseline_configs": dict(run.baseline_configs),
        "trajectory_seed": run.trajectory_seed,
        "branch": None if branch is None else {
            "arm": branch.arm,
            "prefix_identity": branch.prefix_identity,
            "source_identity": branch.source_identity,
            "checkpoint_index": branch.checkpoint.checkpoint_index,
            "root_count": branch.root_count,
            "injected_root_id": branch.injected_root_id,
            "candidate_triplet_id": branch.candidate_triplet_id,
            "native_render_id": branch.native_render_id,
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def snapshot_source_paths(baseline: str) -> set[Path]:
    native_modules = {
        "bot_style": (bot_phase12, bot_runtime, bot_read, bot_solve, bot_write),
        "dc_rs": (dynamic_cheatsheet_phase12, dynamic_cheatsheet_optional,
                  phase13_dc_rs_runtime, phase13_dc_rs_validation),
    }[baseline]
    return {Path(getfile(module)).resolve() for module in (
        runtime_registry, phase13_ordinary_runtime, *native_modules,
    )} | {Path(__file__).resolve()}


def execute_snapshot_unit(
    run: ProspectiveOrdinaryRun, bindings: SnapshotBindings, clone_root: Path,
) -> SnapshotUnitResult:
    """Restore one bound native input and consume its ID once within clone_root.

    clone_root is a task-owned, persistent namespace shared by all its units.
    A consumed ID is never released, including on failure. Durable accounting
    and restart reconciliation remain owned by the existing runner ledger.
    Candidate pins prove bytes only; absent candidate qualification stays pending.
    """
    branch = run.branch
    if (run.baseline not in {"bot_style", "dc_rs"} or branch is None or len(run.tasks) != 1
        or run.validated_resources is not None or run.production_identity is not None
        or not run.allow_test_client or run.initial_states):
        raise SnapshotError("SNAPSHOT_SINGLE_NATIVE_QA_INPUT_REQUIRED")
    if run.arm not in {"clean", "correct", "irrelevant", "contam"} or branch.arm != run.arm:
        raise SnapshotError("SNAPSHOT_NATIVE_BRANCH_REQUIRED")
    if snapshot_input_sha256(run) != bindings.input_sha256:
        raise SnapshotError("SNAPSHOT_INPUT_BINDING_MISMATCH")
    required_sources = snapshot_source_paths(run.baseline)
    if not required_sources <= {source.path.resolve() for source in bindings.source_files}:
        raise SnapshotError("SNAPSHOT_SOURCE_BINDING_REQUIRED")
    for source in bindings.source_files:
        source.verify()
    if bindings.candidate is not None:
        bindings.candidate.verify()
    checkpoint = branch.checkpoint
    deserialize_checkpoint(checkpoint)
    if (checkpoint.canonical_sha256 != bindings.state_sha256
        or checkpoint.state.baseline != run.baseline):
        raise SnapshotError("SNAPSHOT_STATE_BINDING_MISMATCH")
    runtime = PHASE13_CORE_BASELINE_REGISTRY[run.baseline]
    current = runtime.serialize_state(branch.state)
    if not isinstance(current, NativeState) or serialize_checkpoint(current).canonical_sha256 != bindings.state_sha256:
        raise SnapshotError("SNAPSHOT_STATE_BINDING_MISMATCH")
    identity = bindings.archive_identity
    sample_hash = hashlib.sha256(json.dumps((run.tasks[0].sample_id,), separators=(",", ":")).encode()).hexdigest()
    registration = Path(__file__).resolve().parents[3] / "data/phase13/observability/registration_packet_v2.json"
    if (identity.scientific_result or identity.trajectory_seed != run.trajectory_seed
        or identity.ordered_sample_ids_sha256 != sample_hash
        or identity.execution_template_id != f"{run.task_name}:{run.baseline}:{run.arm}"
        or identity.registration_packet_sha256 != hashlib.sha256(registration.read_bytes()).hexdigest()):
        raise SnapshotError("SNAPSHOT_ARCHIVE_BINDING_MISMATCH")
    clone_path = clone_root / ("snapshot-clone-" + hashlib.sha256(run.run_id.encode()).hexdigest())
    try:
        clone_path.mkdir()
    except FileExistsError as error:
        raise SnapshotError("SNAPSHOT_CLONE_ALREADY_CONSUMED") from error
    restored = NativeState.from_mapping(json.loads(checkpoint.canonical_bytes))
    fresh_checkpoint = serialize_checkpoint(restored, checkpoint_index=checkpoint.checkpoint_index)
    fresh_state = runtime.restore_state(restored, _context(run, run.tasks[0], 1))
    clone_run = replace(run, branch=replace(branch, checkpoint=fresh_checkpoint, state=fresh_state))
    result = execute_prospective_ordinary(clone_run)
    archive = production_archive_from_ordinary(clone_run, result, identity)
    return SnapshotUnitResult(clone_run, result, archive, bindings, clone_path)


def snapshot_dispatch(snapshot: SnapshotUnitResult) -> tuple[ProductionObject, MainUnitDispatchOutput]:
    """Bind synthetic native output for the existing durable writer/reconciler."""
    from memcontam.readiness import phase13_durable_call_join
    from memcontam.readiness.phase13_durable_call_join import DiagnosticFile, DiagnosticScope
    from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
    from memcontam.readiness.phase13_main_live_runtime_support import dispatch_output
    from memcontam.readiness.phase13_main_production import ProductionObject

    run, bindings = snapshot.run, snapshot.bindings
    if run.branch is None:
        raise SnapshotError("SNAPSHOT_NATIVE_BRANCH_REQUIRED")
    checkpoint = run.branch.checkpoint
    identity = bindings.archive_identity.model_copy(update={"checkpoint_registry_sha256": bindings.state_sha256})
    source = Path(phase13_durable_call_join.__file__).resolve()
    sources = (*bindings.source_files, SnapshotFileBinding(source, hashlib.sha256(source.read_bytes()).hexdigest()))
    scope = DiagnosticScope(run_id=run.run_id, input_utf8=snapshot_input_utf8(run),
        input_sha256=bindings.input_sha256, state_utf8=checkpoint.canonical_bytes.decode(),
        state_sha256=bindings.state_sha256,
        source_files=tuple(DiagnosticFile(path=str(pin.path.resolve()), sha256=pin.sha256) for pin in sources),
        candidate=None if bindings.candidate is None else DiagnosticFile(
            path=str(bindings.candidate.path.resolve()), sha256=bindings.candidate.sha256), archive_identity=identity)
    unit = ProductionObject(sequence=0, unit_id=hashlib.sha256((run.run_id + scope.sha256()).encode()).hexdigest(),
        kind="MEMORY_BEARING", seed=identity.trajectory_seed, task=run.task_name,
        memory_baseline=run.baseline, arm=run.arm,
        prefix_unit_id=hashlib.sha256(run.branch.prefix_identity.encode()).hexdigest(), projected_cost_krw=0,
        execution_template_id=identity.execution_template_id, ordered_sample_ids_sha256=identity.ordered_sample_ids_sha256,
        registration_packet_sha256=identity.registration_packet_sha256, checkpoint_registry_sha256=identity.checkpoint_registry_sha256,
        diagnostic_trial_count=1, diagnostic_scope_sha256=scope.sha256())
    output = dispatch_output(unit, snapshot.ordinary.trials, identity, snapshot.archive, realized_cost_krw=0)
    runtime = output.evidence
    if not isinstance(runtime, dict):
        raise SnapshotError("SNAPSHOT_ARCHIVE_BINDING_MISMATCH")
    state_after = snapshot.ordinary.trials[0].state_after
    if not isinstance(state_after, NativeState):
        raise SnapshotError("SNAPSHOT_STATE_BINDING_MISMATCH")
    runtime = {**runtime, "diagnostic_scope": scope.model_dump(mode="json"),
               "diagnostic_post_state_utf8": serialize_checkpoint(state_after).canonical_bytes.decode()}
    if snapshot.ordinary.trials[0].outcome.status == "failed":
        runtime["verifier_result"] = None
    return unit, MainUnitDispatchOutput(evidence={
        "evidence_kind": "MEMORY_BEARING", "prefix_unit_id": unit.prefix_unit_id,
        "consumed_checkpoint_id": checkpoint.identity.checkpoint_id,
        "consumed_checkpoint_identity_sha256": checkpoint.identity.sha256,
        "consumed_checkpoint_canonical_sha256": checkpoint.canonical_sha256,
        "runtime_evidence": runtime,
    }, provider_calls=output.provider_calls, realized_cost_krw=0)
