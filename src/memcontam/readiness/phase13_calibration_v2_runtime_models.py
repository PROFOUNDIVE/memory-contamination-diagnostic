from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.live_branch import Arm, LiveThreeArmBranches
from memcontam.experiment.phase12.runtime_registry import RuntimeEntry
from memcontam.main_registry import Task
from memcontam.readiness.phase13_analysis_models import AnalysisRegistry
from memcontam.readiness.phase13_calibration_v2 import CalibrationV2Config
from memcontam.readiness.phase13_execution_models import ExecutionRegistry
from memcontam.readiness.phase13_provider_runtime import Phase13V2ProviderRuntime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memcontam.readiness.phase13_calibration_v2_accounting import AccountingClosure


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeAuthorization:
    authorization_id: str
    config_id: str
    execution_registry_hash: str
    analysis_registry_hash: str
    execution_owner_id: str

    def __post_init__(self) -> None:
        if not all((
            self.authorization_id,
            self.config_id,
            self.execution_registry_hash,
            self.analysis_registry_hash,
            self.execution_owner_id,
        )):
            raise ValueError("AUTHORIZATION_ID_REQUIRED")


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeContext:
    root: Path
    config: CalibrationV2Config
    authorization: VerifiedRuntimeAuthorization
    execution: ExecutionRegistry
    analysis: AnalysisRegistry
    partition_sha256: str
    ordered_suffixes: Mapping[tuple[Task, int], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class TrajectoryRequest:
    verified: VerifiedRuntimeContext
    stream_id: str
    task: Task
    seed_id: int
    source_ordered_stream_sha256: str
    session_id: str
    branches_by_baseline: Mapping[str, LiveThreeArmBranches]
    contexts: tuple[Game24RuntimeContext, ...]
    providers: Mapping[tuple[str, str], Phase13V2ProviderRuntime]
    registry: Mapping[str, RuntimeEntry]


@dataclass(frozen=True, slots=True)
class TrajectoryEvent:
    event_time: int
    absolute_trial_index: int
    baseline: str
    arm: Arm
    source_checkpoint_id: str
    branch_checkpoint_id: str
    suffix_id: str
    task: Task
    model: str
    session_id: str
    intervention_id: str | None
    execution_owner_id: str
    state_before_sha256: str
    state_after_sha256: str


@dataclass(frozen=True, slots=True)
class CompletedTrajectory:
    status: Literal["completed"]
    stream_id: str
    events: tuple[TrajectoryEvent, ...]
    nomem_underlying_execution_count: Literal[1]
    accounting_closure: AccountingClosure
    sealed: Literal[True] = True


@dataclass(frozen=True, slots=True)
class InvalidatedTrajectory:
    status: Literal["invalidated"]
    stream_id: str
    events: tuple[TrajectoryEvent, ...]
    failure_code: str
    accounting_closure: AccountingClosure
    sealed: Literal[True] = True


TrajectoryResult = CompletedTrajectory | InvalidatedTrajectory


@dataclass(frozen=True, slots=True)
class AuthorizedTrajectoryExecution:
    authorization: VerifiedRuntimeAuthorization
    request: TrajectoryRequest

    def __post_init__(self) -> None:
        if self.authorization is not self.request.verified.authorization:
            raise ValueError("AUTHORIZATION_REQUEST_MISMATCH")


__all__ = (
    "CompletedTrajectory",
    "AuthorizedTrajectoryExecution",
    "InvalidatedTrajectory",
    "TrajectoryEvent",
    "TrajectoryRequest",
    "TrajectoryResult",
    "VerifiedRuntimeAuthorization",
    "VerifiedRuntimeContext",
)
