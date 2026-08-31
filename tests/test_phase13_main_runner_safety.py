from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import memcontam.readiness.phase13_main_runner_store as store_module
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


def _units():
    package = MainExecutionFreeze.model_validate_json(P5.read_bytes())
    return enumerate_execution_units(package)


def _binding() -> MainRunBinding:
    return MainRunBinding(
        "phase13-main-a-execution-freeze-v1",
        "1" * 64,
        "2" * 64,
        "phase13-main-a-authorized-execution-v1",
        "3" * 64,
        "4" * 64,
        "5" * 64,
    )


def _ledger(tmp_path: Path) -> MainRunLedger:
    return MainRunLedger.create(tmp_path / "main-run.sqlite3", _binding(), _units())


def test_dispatch_outcomes_reject_invalid_digest_and_negative_cost() -> None:
    with pytest.raises(MainRunError, match="MAIN_RUN_EVIDENCE_INVALID"):
        DispatchCompleted("not-a-sha256", 0)
    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        DispatchCompleted("6" * 64, -1)
    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        DispatchTechnicalFailure("PROVIDER_QUOTA", "7" * 64, -1)


def test_negative_projected_cost_fails_before_intent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)

    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        run_pending(
            ledger,
            lambda _unit: DispatchCompleted("8" * 64, 0),
            projected_cost_krw=lambda _unit: -1,
            tranche_ceiling_krw=0,
        )

    assert ledger.status().in_flight_count == 0


def test_second_runner_cannot_claim_while_unit_is_inflight(tmp_path: Path) -> None:
    first = _ledger(tmp_path)
    unit = first.next_pending()
    assert unit is not None
    first.persist_dispatch_intent(unit.unit_id)
    second = MainRunLedger.open(first.path, _binding(), _units())
    next_unit = second.next_pending()
    assert next_unit is not None

    with pytest.raises(MainRunError, match="MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"):
        second.persist_dispatch_intent(next_unit.unit_id)


def test_reconciliation_evidence_must_bind_current_intent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    context = ledger.in_flight_context(unit.unit_id)
    next_unit = _units()[1]
    wrong_context = replace(context, unit_id=next_unit.unit_id)
    evidence = InFlightEvidence.no_provider_request(wrong_context, "9" * 64)

    with pytest.raises(MainRunError, match="MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID"):
        ledger.reconcile(unit.unit_id, evidence)


def test_event_genesis_is_bound_and_nonempty(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)

    with sqlite3.connect(ledger.path) as connection:
        previous_hash = connection.execute(
            "SELECT previous_hash FROM events WHERE event_sequence = 0"
        ).fetchone()[0]

    assert len(previous_hash) == 64
    assert previous_hash != "0" * 64


def test_crash_during_creation_never_publishes_partial_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "main-run.sqlite3"

    def crash(*_args) -> None:
        raise MainRunError("TEST_CREATION_CRASH")

    monkeypatch.setattr(store_module, "_initialize_ledger", crash)

    with pytest.raises(MainRunError, match="TEST_CREATION_CRASH"):
        MainRunLedger.create(path, _binding(), _units())

    assert not path.exists()
