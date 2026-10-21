"""One-step native snapshot QA; byte bindings confer no scientific authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from inspect import getfile
from pathlib import Path
from typing import Literal

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
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    native_modules = {
        "bot_style": (bot_phase12, bot_runtime, bot_read, bot_solve, bot_write),
        "dc_rs": (dynamic_cheatsheet_phase12, dynamic_cheatsheet_optional,
                  phase13_dc_rs_runtime, phase13_dc_rs_validation),
    }[run.baseline]
    required_sources = {Path(getfile(module)).resolve() for module in (
        runtime_registry, phase13_ordinary_runtime, *native_modules,
    )} | {Path(__file__).resolve()}
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
