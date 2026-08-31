from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_runner import (
    DispatchCompleted,
    DispatchTechnicalFailure,
    InFlightEvidence,
    MainRunBinding,
    MainRunError,
    MainRunLedger,
    enumerate_execution_units,
    run_pending,
)


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"


def _package() -> MainExecutionFreeze:
    return MainExecutionFreeze.model_validate_json(P5.read_bytes())


def _binding() -> MainRunBinding:
    return MainRunBinding(
        package_id="phase13-main-a-execution-freeze-v1",
        package_sha256="1" * 64,
        package_hash="2" * 64,
        authorization_id="phase13-main-a-authorized-execution-v1",
        authorization_sha256="3" * 64,
        authorization_hash="4" * 64,
        runner_sha256="5" * 64,
    )


def _ledger(tmp_path: Path) -> MainRunLedger:
    return MainRunLedger.create(
        tmp_path / "main-run.sqlite3",
        _binding(),
        enumerate_execution_units(_package()),
    )


def _success(unit_id: str) -> DispatchCompleted:
    return DispatchCompleted(
        evidence_sha256=hashlib.sha256(unit_id.encode()).hexdigest(),
        realized_cost_krw=1,
    )


def test_frozen_domain_enumerates_970_injective_units_once() -> None:
    units = enumerate_execution_units(_package())

    assert len(units) == 970
    assert len({unit.unit_id for unit in units}) == 970
    assert tuple(unit.sequence for unit in units) == tuple(range(970))


def test_nomem_is_one_disjoint_singleton_per_task_and_seed() -> None:
    units = enumerate_execution_units(_package())
    nomem = tuple(unit for unit in units if unit.kind == "NO_MEMORY_SINGLETON")

    assert len(nomem) == 50
    assert {(unit.seed, unit.task) for unit in nomem} == {
        (seed, task) for seed in range(10) for task in _package().active_cells.tasks
    }
    assert {unit.arm for unit in nomem} == {"NOT_APPLICABLE"}
    assert {unit.memory_baseline for unit in nomem} == {None}


@pytest.mark.parametrize("terminal", ["COMPLETED", "TERMINAL_TECHNICAL_MISSING"])
def test_terminal_unit_is_never_redispatched(tmp_path: Path, terminal: str) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    if terminal == "COMPLETED":
        ledger.persist_completed(unit.unit_id, _success(unit.unit_id))
    else:
        ledger.persist_terminal_missing(
            unit.unit_id,
            DispatchTechnicalFailure("PROVIDER_QUOTA", "6" * 64),
        )
    calls: list[str] = []

    run_pending(
        ledger,
        lambda row: calls.append(row.unit_id) or _success(row.unit_id),
        projected_cost_krw=lambda _row: 1,
        tranche_ceiling_krw=10,
        max_units=1,
    )

    assert calls == [enumerate_execution_units(_package())[1].unit_id]


def test_clean_tranche_pause_precedes_dispatch_intent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    calls: list[str] = []

    report = run_pending(
        ledger,
        lambda unit: calls.append(unit.unit_id) or _success(unit.unit_id),
        projected_cost_krw=lambda _unit: 11,
        tranche_ceiling_krw=10,
    )

    assert report.session_state == "PAUSED_BEFORE_DISPATCH"
    assert report.attempted_count == 0
    assert calls == []
    assert ledger.status().in_flight_count == 0


def test_quota_failure_terminalizes_current_unit_and_stops(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    calls: list[str] = []

    def dispatch(unit) -> DispatchCompleted:
        calls.append(unit.unit_id)
        raise DispatchTechnicalFailure("PROVIDER_QUOTA", "7" * 64)

    report = run_pending(
        ledger,
        dispatch,
        projected_cost_krw=lambda _unit: 1,
        tranche_ceiling_krw=10,
    )

    assert report.session_state == "STOPPED_TERMINAL_TECHNICAL_MISSING"
    assert report.terminal_technical_missing_count == 1
    assert len(calls) == 1
    assert ledger.status().pending_count == 969


def test_repeated_resume_does_not_duplicate_calls(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    calls: list[str] = []

    def dispatch(unit) -> DispatchCompleted:
        calls.append(unit.unit_id)
        return _success(unit.unit_id)

    for _ in range(3):
        run_pending(
            ledger,
            dispatch,
            projected_cost_krw=lambda _unit: 1,
            tranche_ceiling_krw=10,
            max_units=1,
        )

    assert len(calls) == len(set(calls)) == 3
    assert ledger.status().completed_count == 3


def test_ledger_tampering_fails_closed(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE execution_units SET state = 'COMPLETED' WHERE unit_id = ?",
            (unit.unit_id,),
        )

    with pytest.raises(MainRunError, match="MAIN_RUN_LEDGER_INTEGRITY_INVALID"):
        MainRunLedger.open(ledger.path, _binding(), enumerate_execution_units(_package()))


def test_crash_before_intent_leaves_unit_safely_pending(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    with pytest.raises(RuntimeError, match="crash-before-intent"):
        run_pending(
            ledger,
            lambda unit: _success(unit.unit_id),
            projected_cost_krw=lambda _unit: (_ for _ in ()).throw(
                RuntimeError("crash-before-intent")
            ),
            tranche_ceiling_krw=10,
        )

    assert ledger.status().pending_count == 970
    assert ledger.status().in_flight_count == 0


def test_inflight_with_proven_no_request_can_return_to_pending(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)

    context = ledger.in_flight_context(unit.unit_id)
    ledger.reconcile(unit.unit_id, InFlightEvidence.no_provider_request(context, "8" * 64))

    assert ledger.next_pending() == unit
    assert ledger.status().in_flight_count == 0


def test_inflight_with_terminal_evidence_is_never_retried(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)

    context = ledger.in_flight_context(unit.unit_id)
    ledger.reconcile(unit.unit_id, InFlightEvidence.terminal_failure(context, "9" * 64))

    assert ledger.status().terminal_technical_missing_count == 1
    assert ledger.next_pending() != unit


def test_accepted_but_unpersisted_request_blocks_blind_resume(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    context = ledger.in_flight_context(unit.unit_id)

    with pytest.raises(MainRunError, match="MAIN_RUN_IN_FLIGHT_AMBIGUOUS"):
        ledger.reconcile(unit.unit_id, InFlightEvidence.ambiguous(context, "a" * 64))

    assert ledger.status().in_flight_count == 1


def test_crash_after_terminal_persist_skips_unit_on_resume(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    ledger.persist_completed(unit.unit_id, _success(unit.unit_id))

    reopened = MainRunLedger.open(
        ledger.path,
        _binding(),
        enumerate_execution_units(_package()),
    )

    assert reopened.next_pending() != unit
    assert reopened.status().completed_count == 1
