from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Final, Mapping

from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.live_branch import Arm
from memcontam.experiment.phase12.live_suffix import run_live_matched_suffix
from memcontam.experiment.phase12.runtime_registry import RuntimeEntry, RuntimeTrialResult
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.main_registry import Task
from memcontam.readiness.phase13_analysis_contract import load_analysis_registry
from memcontam.readiness.phase13_calibration_v2 import CalibrationV2Config
from memcontam.readiness.phase13_calibration_v2_registry import validate_calibration_v2_registry
from memcontam.readiness.phase13_execution_contract import load_execution_registry
from memcontam.readiness.phase13_structural_authority import registered_checkpoints
from memcontam.readiness.phase13_provider_runtime import Phase13V2ProviderRuntime

from .phase13_calibration_v2_runtime_models import (
    CompletedTrajectory,
    InvalidatedTrajectory,
    TrajectoryEvent,
    TrajectoryRequest,
    TrajectoryResult,
    VerifiedRuntimeAuthorization,
    VerifiedRuntimeContext,
)

BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
ARMS: Final = ("clean", "correct", "irrelevant", "contam")
FORBIDDEN_CONFIG_TOKENS: Final = frozenset({"future", "horizon", "window", "task"})


class CalibrationV2RuntimeError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _Violation(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def verify_runtime_context(
    root: Path,
    config: CalibrationV2Config,
    authorization: VerifiedRuntimeAuthorization,
) -> VerifiedRuntimeContext:
    execution = load_execution_registry(root / config.authority.execution.path, root)
    analysis = load_analysis_registry(root / config.authority.analysis.path, root)
    partition = validate_calibration_v2_registry(root / "data/phase13/calibration_v2", root)
    expected = (
        config.config_id,
        execution.registry_hash,
        analysis.registry_hash,
        execution.execution_owner_id,
    )
    supplied = (
        authorization.config_id,
        authorization.execution_registry_hash,
        authorization.analysis_registry_hash,
        authorization.execution_owner_id,
    )
    if supplied != expected:
        raise CalibrationV2RuntimeError("AUTHORIZATION_AUTHORITY_MISMATCH")
    task_rows = partition.get("tasks")
    if not isinstance(task_rows, dict):
        raise CalibrationV2RuntimeError("SOURCE_STREAM_IDENTITY_INVALID")
    suffixes: dict[tuple[Task, int], tuple[str, ...]] = {}
    for task, value in task_rows.items():
        if task not in {"game24", "math_equation_balancer", "word_sorting"}:
            raise CalibrationV2RuntimeError("SOURCE_STREAM_IDENTITY_INVALID")
        typed_task: Task = task
        if not isinstance(value, dict) or not isinstance(value.get("trajectories"), list):
            raise CalibrationV2RuntimeError("SOURCE_STREAM_IDENTITY_INVALID")
        for trajectory in value["trajectories"]:
            if not isinstance(trajectory, dict):
                raise CalibrationV2RuntimeError("SOURCE_STREAM_IDENTITY_INVALID")
            seed, ordered = trajectory.get("seed_id"), trajectory.get("ordered_sample_ids")
            if not isinstance(seed, int) or not isinstance(ordered, list) or not all(
                isinstance(item, str) for item in ordered
            ):
                raise CalibrationV2RuntimeError("SOURCE_STREAM_IDENTITY_INVALID")
            suffixes[(typed_task, seed)] = tuple(ordered[1:])
    return VerifiedRuntimeContext(
        config,
        authorization,
        execution,
        analysis,
        config.authority.calibration_partition.file_sha256,
        suffixes,
    )


def execute_calibration_trajectory(request: TrajectoryRequest) -> TrajectoryResult:
    _validate_request(request)
    events: list[TrajectoryEvent] = []
    registry = _observed_registry(request, events)
    contexts = tuple(_context_with_client(context, request, "fh_bounded", "clean") for context in request.contexts)
    try:
        result = run_live_matched_suffix(
            branches_by_baseline=request.branches_by_baseline,
            contexts=contexts,
            registry=registry,
        )
        if result.nomem.underlying_execution_count != 1:
            raise _Violation("NOMEM_SINGLETON_REQUIRED")
        _reconcile(request)
    except _Violation as error:
        return InvalidatedTrajectory("invalidated", request.stream_id, tuple(events), error.code)
    except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        return InvalidatedTrajectory(
            "invalidated", request.stream_id, tuple(events),
            getattr(error, "code", "TRAJECTORY_EXECUTION_FAILED"),
        )
    return CompletedTrajectory("completed", request.stream_id, tuple(events), 1)


def _validate_request(request: TrajectoryRequest) -> None:
    execution = request.verified.execution
    stream = next((row for row in execution.task_streams if row.task == request.task), None)
    suffix = None if stream is None else next(
        (row for row in stream.suffixes if row.seed_id == request.seed_id), None
    )
    if suffix is None or suffix.source_ordered_stream_sha256 != request.source_ordered_stream_sha256:
        raise CalibrationV2RuntimeError("SOURCE_STREAM_IDENTITY_INVALID")
    if request.stream_id != f"{request.task}-seed-{request.seed_id}":
        raise CalibrationV2RuntimeError("SOURCE_STREAM_IDENTITY_INVALID")
    if tuple(request.branches_by_baseline) != BASELINES:
        raise CalibrationV2RuntimeError("BASELINE_PANEL_INVALID")
    registered = {row.baseline: row for row in registered_checkpoints(request.stream_id)}
    for baseline, branches in request.branches_by_baseline.items():
        if tuple(branches.arms) != ARMS or "filter" in branches.arms:
            raise CalibrationV2RuntimeError("FILTER_BRANCH_FORBIDDEN")
        authority = registered[baseline]
        if any(
            branch.prefix_identity != authority.checkpoint_id
            for branch in branches.arms.values()
        ):
            raise CalibrationV2RuntimeError("CHECKPOINT_AUTHORITY_MISMATCH")
    if len(request.contexts) != execution.timing.H_run:
        raise CalibrationV2RuntimeError("HORIZON_INVALID")
    if tuple(context.identities.order_key for context in request.contexts) != tuple(range(2, 12)):
        raise CalibrationV2RuntimeError("EVENT_RANGE_INVALID")
    expected_suffix = request.verified.ordered_suffixes.get((request.task, request.seed_id))
    if tuple(context.task.sample_id for context in request.contexts) != expected_suffix:
        raise CalibrationV2RuntimeError("SUFFIX_TASK_DRIFT")
    required = {(baseline, arm) for baseline in BASELINES for arm in ARMS} | {("nomem", "clean")}
    if set(request.providers) != required:
        raise CalibrationV2RuntimeError("OWNED_PROVIDER_PANEL_INVALID")
    if any(not isinstance(provider, Phase13V2ProviderRuntime) for provider in request.providers.values()):
        raise CalibrationV2RuntimeError("OWNED_PROVIDER_REQUIRED")


def _observed_registry(
    request: TrajectoryRequest, events: list[TrajectoryEvent]
) -> Mapping[str, RuntimeEntry]:
    registry: dict[str, RuntimeEntry] = {}
    for baseline in (*BASELINES, "nomem"):
        entry = request.registry.get(baseline)
        if entry is None:
            raise CalibrationV2RuntimeError("RUNTIME_REGISTRY_INCOMPLETE")

        def execute(context, state, *, baseline=baseline, entry=entry):  # noqa: ANN001, ANN202
            arm = "clean" if baseline == "nomem" else context.branch
            observed = replace(context, client=request.providers[(baseline, arm)])
            _validate_provider_configs((dict(context.decoding),))
            before = _state_hash(entry, state)
            result = entry.execute_trial(observed, state)
            if not isinstance(result, RuntimeTrialResult):
                raise _Violation("INVALID_LIVE_TRIAL_RESULT")
            after = _state_hash(entry, result.state)
            if after in _prior_hashes(events, baseline, arm) and after != before:
                raise _Violation("STATE_REWIND")
            if baseline == "rag_frozen" and (result.native_entries or result.write_envelopes):
                raise _Violation("RAG_WRITE_FORBIDDEN")
            if baseline == "reflexion_style" and bool(result.native_entries) != bool(
                result.write_envelopes
            ):
                raise _Violation("REFLEXION_WRITE_MISMATCH")
            if baseline != "nomem":
                events.append(_event(request, context, baseline, arm, before, after))
            return result

        registry[baseline] = replace(entry, execute_trial=execute)
    return registry


def _context_with_client(
    context: Game24RuntimeContext, request: TrajectoryRequest, baseline: str, arm: str
) -> Game24RuntimeContext:
    return replace(context, client=request.providers[(baseline, arm)])


def _state_hash(entry: RuntimeEntry, state: object) -> str:
    snapshot = entry.serialize_state(state)
    if isinstance(snapshot, NativeState):
        return serialize_checkpoint(snapshot).canonical_sha256
    return hashlib.sha256(repr(snapshot).encode()).hexdigest()


def _prior_hashes(events: list[TrajectoryEvent], baseline: str, arm: Arm) -> set[str]:
    return {
        event.state_before_sha256
        for event in events
        if (event.baseline, event.arm) == (baseline, arm)
    }


def _event(
    request: TrajectoryRequest,
    context: Game24RuntimeContext,
    baseline: str,
    arm: Arm,
    before: str,
    after: str,
) -> TrajectoryEvent:
    branch = request.branches_by_baseline[baseline].arms[arm]
    absolute = int(context.identities.order_key)
    return TrajectoryEvent(
        absolute - 2, absolute, baseline, arm, branch.prefix_identity,
        branch.checkpoint.identity.checkpoint_id, context.task.sample_id, request.task,
        context.model, request.session_id, branch.injected_root_id,
        request.verified.execution.execution_owner_id, before, after,
    )


def _validate_provider_configs(configs: tuple[dict[str, object], ...]) -> None:
    for config in configs:
        for key in config:
            tokens = frozenset(key.lower().replace("-", "_").split("_"))
            if tokens & FORBIDDEN_CONFIG_TOKENS:
                raise _Violation("PROVIDER_CONFIG_LEAKAGE")


def _reconcile(request: TrajectoryRequest) -> None:
    for provider in request.providers.values():
        report = provider.reconcile()
        configs = getattr(provider, "provider_configs", ())
        _validate_provider_configs(configs)
        if report.totals.semantic_calls == 0:
            raise _Violation("UNSETTLED_PROVIDER_CALL")


__all__ = (
    "CalibrationV2RuntimeError",
    "CompletedTrajectory",
    "InvalidatedTrajectory",
    "TrajectoryEvent",
    "TrajectoryRequest",
    "VerifiedRuntimeAuthorization",
    "VerifiedRuntimeContext",
    "execute_calibration_trajectory",
    "verify_runtime_context",
)
