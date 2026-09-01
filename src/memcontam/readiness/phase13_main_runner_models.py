from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_production import (
    ProductionObject as ExecutionUnit,
    build_production_objects,
    units_sha256 as units_sha256,
)


LEDGER_SCHEMA_ID = "phase13-main-a-ledger-v1"
RUNNER_ID = "phase13-main-a-runner-v1"
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
class MainRunBinding:
    package_id: str
    package_sha256: str
    package_hash: str
    authorization_id: str
    authorization_sha256: str
    authorization_hash: str
    runner_sha256: str
    core_authorization_gate_krw: int = 450000


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
    failure_code: str | None = None

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
        failure_code: str,
        realized_cost_krw: int = 0,
    ) -> InFlightEvidence:
        return cls(
            "TERMINAL_FAILURE",
            context,
            evidence_sha256,
            realized_cost_krw,
            failure_code,
        )

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


def enumerate_execution_units(
    package: MainExecutionFreeze,
    repository_root: Path | None = None,
) -> tuple[ExecutionUnit, ...]:
    try:
        ordered_hashes = None
        if repository_root is not None:
            from memcontam.readiness.phase13_main_checkpoint import CommonCheckpointRegistry

            binding = next(
                row for row in package.artifacts if row.role == "common_checkpoint_registry"
            )
            raw = read_regular_nofollow(repository_root / binding.path)
            if hashlib.sha256(raw).hexdigest() != binding.sha256:
                raise MainRunError("MAIN_RUN_CHECKPOINT_REGISTRY_INVALID")
            registry = CommonCheckpointRegistry.model_validate_json(raw)
            ordered_hashes = {
                (task, seed.seed): seed.suffix_sample_ids_sha256
                for task, task_row in registry.tasks.items()
                for seed in task_row.seeds
            }
        return build_production_objects(package, ordered_hashes)
    except (OSError, StopIteration, ValidationError, ValueError) as error:
        raise MainRunError(str(error)) from error


def _require_sha256(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MainRunError("MAIN_RUN_EVIDENCE_INVALID")


def _require_cost(value: int) -> None:
    if type(value) is not int or value < 0:
        raise MainRunError("MAIN_RUN_COST_INVALID")
