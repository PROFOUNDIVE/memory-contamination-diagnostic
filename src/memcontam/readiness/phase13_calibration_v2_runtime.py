from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Final, Literal, Mapping

from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.live_branch import Arm
from memcontam.experiment.phase12.live_suffix import run_live_matched_suffix
from memcontam.experiment.phase12.runtime_registry import RuntimeEntry, RuntimeTrialResult
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.main_registry import Task
from memcontam.logging.schema import VerifierResult
from memcontam.readiness.phase13_analysis_contract import load_analysis_registry
from memcontam.readiness.phase13_calibration_v2 import CalibrationV2Config
from memcontam.readiness.phase13_calibration_v2_registry import validate_calibration_v2_registry
from memcontam.readiness.phase13_execution_contract import load_execution_registry
from memcontam.readiness.phase13_calibration_v2_accounting import close_accounting
from memcontam.readiness.phase13_calibration_v2_runtime_validation import (
    RuntimeValidationError,
    validate_trajectory_request,
)
from memcontam.readiness.phase13_runtime_event import build_trajectory_event

from .phase13_calibration_v2_runtime_models import (
    AuthorizedTrajectoryExecution,
    CompletedTrajectory,
    InvalidatedTrajectory,
    TrajectoryEvent,
    TrajectoryRequest,
    TrajectoryResult,
    RuntimeSourceSeal,
    VerifiedRuntimeAuthorization,
    VerifiedRuntimeContext,
)

BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
ARMS: Final = ("clean", "correct", "irrelevant", "contam")


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
        root,
        config,
        authorization,
        execution,
        analysis,
        config.authority.calibration_partition.file_sha256,
        suffixes,
    )


def execute_calibration_trajectory(request: TrajectoryRequest) -> TrajectoryResult:
    try:
        validate_trajectory_request(request)
    except RuntimeValidationError as error:
        raise CalibrationV2RuntimeError(error.code) from error
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
        closure = close_accounting(request)
        if closure.status != "closed_complete":
            raise _Violation("PROVIDER_CALL_LEDGER_MISMATCH")
    except _Violation as error:
        return InvalidatedTrajectory(
            "invalidated", request.stream_id, tuple(events), error.code, close_accounting(request)
        )
    except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK
        return InvalidatedTrajectory(
            "invalidated",
            request.stream_id,
            tuple(events),
            getattr(error, "code", "TRAJECTORY_EXECUTION_FAILED"),
            close_accounting(request),
        )
    raw_hash = hashlib.sha256(
        b"".join(
            (json.dumps(asdict(event), sort_keys=True, separators=(",", ":")) + "\n").encode()
            for event in events
        )
    ).hexdigest()
    return CompletedTrajectory(
        "completed",
        request.stream_id,
        tuple(events),
        request.stream_id,
        raw_hash,
        RuntimeSourceSeal(
            request.stream_id,
            raw_hash,
            request.verified.execution.registry_hash,
            request.verified.analysis.registry_hash,
        ),
        1,
        closure,
    )


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
            branch = None if baseline == "nomem" else request.branches_by_baseline[baseline].arms[arm]
            runtime_config = {
                **dict(context.decoding),
                **({"session_id": request.session_id} if baseline != "nomem" else {"session_id": request.session_id}),
                **(
                    {"intervention_id": branch.injected_root_id}
                    if branch is not None and branch.injected_root_id is not None
                    else {}
                ),
            }
            observed = replace(
                context,
                client=request.providers[(baseline, arm)],
                decoding=runtime_config,
            )
            before = _state_hash(entry, state)
            result = entry.execute_trial(observed, state)
            if not isinstance(result, RuntimeTrialResult):
                raise _Violation("INVALID_LIVE_TRIAL_RESULT")
            after = _state_hash(entry, result.state)
            if baseline != "nomem" and result.state is not state:
                raise _Violation("STATE_IDENTITY_DRIFT")
            prior_after = _prior_after(events, baseline, arm)
            if prior_after is not None and before != prior_after:
                raise _Violation("STATE_CHAIN_BROKEN")
            if after in _prior_before(events, baseline, arm) and after != before:
                raise _Violation("STATE_REWIND")
            if after == before and baseline not in {"rag_frozen", "nomem"}:
                raise _Violation("STATE_STAGNANT")
            if baseline == "rag_frozen" and (result.native_entries or result.write_envelopes):
                raise _Violation("RAG_WRITE_FORBIDDEN")
            if baseline == "reflexion_style" and bool(result.native_entries) != bool(
                result.write_envelopes
            ):
                raise _Violation("REFLEXION_WRITE_MISMATCH")
            if baseline != "nomem":
                match result.outcome.verifier_result:
                    case bool() as value:
                        verified: Literal[0, 1] = 1 if value else 0
                    case VerifierResult(is_correct=value):
                        verified = 1 if value else 0
                    case _:
                        raise _Violation("VERIFIER_RESULT_INVALID")
                events.append(
                    build_trajectory_event(
                        request, context, baseline, arm, (before, after),
                        (result.outcome.status, verified),
                    )
                )
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


def _prior_after(events: list[TrajectoryEvent], baseline: str, arm: Arm) -> str | None:
    prior = [event for event in events if (event.baseline, event.arm) == (baseline, arm)]
    return prior[-1].state_after_sha256 if prior else None


def _prior_before(events: list[TrajectoryEvent], baseline: str, arm: Arm) -> set[str]:
    return {
        event.state_before_sha256
        for event in events
        if (event.baseline, event.arm) == (baseline, arm)
    }


__all__ = (
    "AuthorizedTrajectoryExecution",
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
