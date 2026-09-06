from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from memcontam.readiness.phase13_main_runner_ledger import MainRunLedger
from memcontam.readiness.phase13_main_runner_models import (
    DispatchCompleted,
    DispatchTechnicalFailure,
    ExecutionUnit,
    InFlightEvidence,
    MainRunBinding,
    MainRunError,
    MainRunReport,
    enumerate_execution_units,
)

from .phase13_main_live_runtime_support import pending_request_keys_v3
from .phase13_main_request_dispatch import DispatchTechnicalFailureV3, ProductionRequestDispatcherV3
from .phase13_main_request_recovery import require_known_costs
from .phase13_main_v3_runner import V3MainRun, V3RunStatus
from .phase13_v3_request import RequestKeyV3

Dispatch = Callable[[ExecutionUnit], DispatchCompleted]
RequestResult = TypeVar("RequestResult")


def run_pending_requests_v3(
    dispatcher: ProductionRequestDispatcherV3,
    keys: tuple[RequestKeyV3, ...],
    execute: Callable[[RequestKeyV3], RequestResult],
) -> tuple[str, ...]:
    completed: list[str] = []
    for key in pending_request_keys_v3(dispatcher, keys):
        if key.parent_id in dispatcher.terminal_parents:
            continue
        require_known_costs(dispatcher.ledger)
        try:
            execute(key)
        except DispatchTechnicalFailureV3:
            continue
        if dispatcher.ledger.state(key.dispatch_id).kind != "COMPLETED":
            raise MainRunError("MAIN_RUN_COMPLETION_EVIDENCE_INVALID")
        completed.append(key.dispatch_id)
    return tuple(completed)


@dataclass(frozen=True, slots=True)
class MainRunRequest:
    repository_root: Path
    package_path: Path
    authorization_path: Path
    expected_authorization_sha256: str
    run_root: Path
    run_id: str
    authority_root: Path | None = None
    expected_authorization_sha256_file: Path | None = None


def prepare_main_run(request: MainRunRequest) -> V3MainRun:
    return _open_v3(request, create=True)


def open_main_run(request: MainRunRequest) -> V3MainRun:
    return _open_v3(request, create=False)


def _open_v3(request: MainRunRequest, *, create: bool) -> V3MainRun:
    from .phase13_v3_entrypoint import SelectedExecutionV3, SelectionRequest, select_execution
    selected = select_execution(SelectionRequest(request.repository_root, request.package_path,
        request.authorization_path, request.authority_root, request.expected_authorization_sha256_file,
        request.run_id, request.expected_authorization_sha256 or None), "run" if create else "resume")
    if not isinstance(selected, SelectedExecutionV3):
        raise MainRunError("MAIN_PACKAGE_VERSION_UNSUPPORTED")
    return V3MainRun.open(selected, request.run_root / request.run_id, create=create)


def run_main(
    request: MainRunRequest,
    *,
    cache_root: Path,
    tranche_ceiling_krw: int,
    max_units: int | None = None,
) -> V3RunStatus:
    run = prepare_main_run(request)
    try:
        return run.execute(cache_root, tranche_ceiling_krw=tranche_ceiling_krw, max_units=max_units)
    finally:
        run.close()


def resume_main(
    request: MainRunRequest,
    *,
    cache_root: Path,
    tranche_ceiling_krw: int,
    max_units: int | None = None,
) -> V3RunStatus:
    run = open_main_run(request)
    try:
        return run.execute(cache_root, tranche_ceiling_krw=tranche_ceiling_krw, max_units=max_units)
    finally:
        run.close()


def run_pending(
    ledger: MainRunLedger,
    dispatch: Dispatch,
    *,
    tranche_ceiling_krw: int,
    max_units: int | None = None,
) -> MainRunReport:
    status = ledger.status()
    if status.in_flight_count:
        raise MainRunError("MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED")
    attempted = 0
    while max_units is None or attempted < max_units:
        unit = ledger.next_pending()
        if unit is None:
            break
        if not ledger.claim_dispatch(
            unit.unit_id,
            unit.projected_cost_krw,
            tranche_ceiling_krw,
        ):
            return _report(ledger, attempted)
        try:
            completed = dispatch(unit)
        except DispatchTechnicalFailure as failure:
            ledger.persist_terminal_missing(unit.unit_id, failure)
            return _report(ledger, attempted + 1)
        except ValueError as error:
            if isinstance(getattr(error, "code", None), str):
                raise
            raise MainRunError("MAIN_RUN_POST_INTENT_RUNTIME_FAILURE") from error
        ledger.persist_completed(unit.unit_id, completed)
        attempted += 1
        status = ledger.status()
    return _report(ledger, attempted)


def _report(ledger: MainRunLedger, attempted: int) -> MainRunReport:
    status = ledger.status()
    return MainRunReport(
        status.session_state,
        attempted,
        status.completed_count,
        status.terminal_technical_missing_count,
    )


__all__ = [
    "DispatchCompleted",
    "DispatchTechnicalFailure",
    "InFlightEvidence",
    "MainRunBinding",
    "MainRunError",
    "MainRunLedger",
    "MainRunRequest",
    "enumerate_execution_units",
    "open_main_run",
    "prepare_main_run",
    "resume_main",
    "run_main",
    "run_pending",
    "run_pending_requests_v3",
]
