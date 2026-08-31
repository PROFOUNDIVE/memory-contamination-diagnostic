from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze


LEDGER_SCHEMA_ID = "phase13-main-a-ledger-v1"
RUNNER_ID = "phase13-main-a-runner-v1"
UNIT_IDENTITY_LAW_ID = "phase13-main-a-disjoint-unit-id-v1"

UnitKind = Literal["MEMORY_BEARING", "NO_MEMORY_SINGLETON"]
UnitState = Literal[
    "PENDING",
    "DISPATCH_INTENT_PERSISTED",
    "COMPLETED",
    "TERMINAL_TECHNICAL_MISSING",
]
SessionState = Literal[
    "NOT_STARTED",
    "READY",
    "PAUSED_BEFORE_DISPATCH",
    "BLOCKED_IN_FLIGHT",
    "STOPPED_TERMINAL_TECHNICAL_MISSING",
    "COMPLETED",
]


class MainRunError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ExecutionUnit:
    sequence: int
    unit_id: str
    kind: UnitKind
    seed: int
    task: str
    memory_baseline: str | None
    arm: str


@dataclass(frozen=True, slots=True)
class MainRunBinding:
    package_id: str
    package_sha256: str
    package_hash: str
    authorization_id: str
    authorization_sha256: str
    authorization_hash: str
    runner_sha256: str


@dataclass(frozen=True, slots=True)
class DispatchCompleted:
    evidence_sha256: str
    realized_cost_krw: int

    def __post_init__(self) -> None:
        _require_sha256(self.evidence_sha256)
        _require_cost(self.realized_cost_krw)


class DispatchTechnicalFailure(RuntimeError):
    def __init__(
        self,
        code: str,
        evidence_sha256: str,
        realized_cost_krw: int = 0,
    ) -> None:
        self.code = code
        self.evidence_sha256 = evidence_sha256
        self.realized_cost_krw = realized_cost_krw
        _require_sha256(evidence_sha256)
        _require_cost(realized_cost_krw)
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class InFlightContext:
    unit_id: str
    intent_event_hash: str
    package_sha256: str
    authorization_sha256: str


@dataclass(frozen=True, slots=True)
class InFlightEvidence:
    disposition: Literal["NO_PROVIDER_REQUEST", "COMPLETED", "TERMINAL_FAILURE", "AMBIGUOUS"]
    context: InFlightContext
    evidence_sha256: str
    realized_cost_krw: int = 0

    def __post_init__(self) -> None:
        _require_sha256(self.context.intent_event_hash)
        _require_sha256(self.context.package_sha256)
        _require_sha256(self.context.authorization_sha256)
        _require_sha256(self.evidence_sha256)
        _require_cost(self.realized_cost_krw)

    @classmethod
    def no_provider_request(
        cls,
        context: InFlightContext,
        evidence_sha256: str,
    ) -> InFlightEvidence:
        return cls("NO_PROVIDER_REQUEST", context, evidence_sha256)

    @classmethod
    def completed(
        cls,
        context: InFlightContext,
        evidence_sha256: str,
        realized_cost_krw: int,
    ) -> InFlightEvidence:
        return cls("COMPLETED", context, evidence_sha256, realized_cost_krw)

    @classmethod
    def terminal_failure(
        cls,
        context: InFlightContext,
        evidence_sha256: str,
        realized_cost_krw: int = 0,
    ) -> InFlightEvidence:
        return cls("TERMINAL_FAILURE", context, evidence_sha256, realized_cost_krw)

    @classmethod
    def ambiguous(cls, context: InFlightContext, evidence_sha256: str) -> InFlightEvidence:
        return cls("AMBIGUOUS", context, evidence_sha256)


@dataclass(frozen=True, slots=True)
class MainRunStatus:
    session_state: SessionState
    total_count: int
    pending_count: int
    in_flight_count: int
    completed_count: int
    terminal_technical_missing_count: int
    realized_cost_krw: int


@dataclass(frozen=True, slots=True)
class MainRunReport:
    session_state: SessionState
    attempted_count: int
    completed_count: int
    terminal_technical_missing_count: int


def enumerate_execution_units(package: MainExecutionFreeze) -> tuple[ExecutionUnit, ...]:
    pairs_by_task = {
        task: tuple(baseline for pair_task, baseline in package.active_cells.included_task_baseline_pairs if pair_task == task)
        for task in package.dispatch.task_order
    }
    units: list[ExecutionUnit] = []
    for seed_rank, seed in enumerate(package.dispatch.concrete_seed_ids):
        arms = package.arm_order.sequences[package.arm_order.seed_sequence_indices[seed_rank]].arms
        for task in package.dispatch.task_order:
            for baseline in pairs_by_task[task]:
                for arm in arms:
                    units.append(_unit(len(units), "MEMORY_BEARING", seed, task, baseline, arm))
            units.append(
                _unit(len(units), "NO_MEMORY_SINGLETON", seed, task, None, "NOT_APPLICABLE")
            )
    if len(units) != package.active_cells.attempted_trajectory_count:
        raise MainRunError("MAIN_RUN_UNIT_DOMAIN_INVALID")
    if len({unit.unit_id for unit in units}) != len(units):
        raise MainRunError("MAIN_RUN_UNIT_IDENTITY_NOT_INJECTIVE")
    return tuple(units)


def units_sha256(units: tuple[ExecutionUnit, ...]) -> str:
    rows = [
        [unit.sequence, unit.unit_id, unit.kind, unit.seed, unit.task, unit.memory_baseline, unit.arm]
        for unit in units
    ]
    return hashlib.sha256(_canonical(rows)).hexdigest()


def _unit(
    sequence: int,
    kind: UnitKind,
    seed: int,
    task: str,
    baseline: str | None,
    arm: str,
) -> ExecutionUnit:
    identity = [UNIT_IDENTITY_LAW_ID, kind, seed, task, baseline, arm]
    return ExecutionUnit(
        sequence,
        hashlib.sha256(_canonical(identity)).hexdigest(),
        kind,
        seed,
        task,
        baseline,
        arm,
    )


def _canonical(value: list) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _require_sha256(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MainRunError("MAIN_RUN_EVIDENCE_INVALID")


def _require_cost(value: int) -> None:
    if type(value) is not int or value < 0:
        raise MainRunError("MAIN_RUN_COST_INVALID")
