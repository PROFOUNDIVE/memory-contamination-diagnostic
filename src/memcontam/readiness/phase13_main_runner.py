from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.readiness.phase13_main_execution import (
    Phase13MainExecutionError,
    validate_main_authorization,
)
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_live_contract import (
    MainLiveContractError,
    load_main_live_contract,
    validate_main_live_contract,
)
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


Dispatch = Callable[[ExecutionUnit], DispatchCompleted]


@dataclass(frozen=True, slots=True)
class MainRunRequest:
    repository_root: Path
    package_path: Path
    authorization_path: Path
    expected_authorization_sha256: str
    run_root: Path
    run_id: str


def prepare_main_run(request: MainRunRequest) -> MainRunLedger:
    package, binding = _validated_inputs(request)
    return MainRunLedger.create(
        _ledger_path(request),
        binding,
        enumerate_execution_units(package, request.repository_root),
    )


def open_main_run(request: MainRunRequest) -> MainRunLedger:
    package, binding = _validated_inputs(request)
    return MainRunLedger.open(
        _ledger_path(request),
        binding,
        enumerate_execution_units(package, request.repository_root),
    )


def run_main(
    request: MainRunRequest,
    dispatch: Dispatch,
    *,
    tranche_ceiling_krw: int,
    max_units: int | None = None,
) -> MainRunReport:
    return run_pending(
        prepare_main_run(request),
        dispatch,
        tranche_ceiling_krw=tranche_ceiling_krw,
        max_units=max_units,
    )


def resume_main(
    request: MainRunRequest,
    dispatch: Dispatch,
    *,
    tranche_ceiling_krw: int,
    max_units: int | None = None,
) -> MainRunReport:
    return run_pending(
        open_main_run(request),
        dispatch,
        tranche_ceiling_krw=tranche_ceiling_krw,
        max_units=max_units,
    )


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


def _validated_inputs(
    request: MainRunRequest,
) -> tuple[MainExecutionFreeze, MainRunBinding]:
    _ledger_path(request)
    try:
        authorization = validate_main_authorization(
            request.repository_root,
            request.package_path,
            request.authorization_path,
            request.expected_authorization_sha256,
        )
        package_raw = read_regular_nofollow(request.package_path)
        if hashlib.sha256(package_raw).hexdigest() != authorization.execution_package_sha256:
            raise MainRunError("MAIN_RUN_PACKAGE_BYTES_CHANGED")
        package = MainExecutionFreeze.model_validate_json(package_raw)
        contract = load_main_live_contract(
            request.repository_root / "data/phase13/main/main_live_contract_v1.json"
        )
        validate_main_live_contract(contract, package)
    except Phase13MainExecutionError as error:
        raise MainRunError(error.code) from error
    except MainRunError:
        raise
    except (AuthorityFileError, MainLiveContractError, OSError, ValidationError) as error:
        raise MainRunError("MAIN_RUN_INPUT_INVALID") from error
    return package, MainRunBinding(
        package.package_id,
        authorization.execution_package_sha256,
        package.package_hash,
        authorization.authorization_id,
        authorization.authorization_sha256,
        authorization.authorization_hash,
        package.execution_control.runner_code_sha256,
        package.cost_guard.core_authorization_gate_krw,
    )


def _ledger_path(request: MainRunRequest) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", request.run_id) is None:
        raise MainRunError("MAIN_RUN_ID_INVALID")
    return request.run_root / request.run_id / "main-run-v1.sqlite3"


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
]
