from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import memcontam.readiness.phase13_cost_policy as cost_policy
import memcontam.readiness.phase13_main_live_evidence as evidence_module
import memcontam.readiness.phase13_main_runner_models as runner_models
import memcontam.readiness.phase13_main_runner_store as store_module
from memcontam.readiness.phase13_main_live_dispatch import (
    persist_reconciliation_evidence,
    persist_unit_dispatch,
)
from memcontam.readiness.phase13_main_runner import (
    DispatchCompleted,
    DispatchTechnicalFailure,
    InFlightEvidence,
    MainRunBinding,
    MainRunError,
    MainRunLedger,
)
from memcontam.readiness.phase13_main_production import ProductionObject

from .phase13_v3_fixtures import HistoricalEvidencePolicy, SyntheticFixture
from .phase13_v3_fixtures import prefix_output as _prefix_output


ROOT = Path(__file__).resolve().parents[1]


def _units() -> tuple[ProductionObject, ...]:
    return SyntheticFixture().units()


def _binding() -> MainRunBinding:
    return SyntheticFixture().binding()


def _ledger(tmp_path: Path) -> MainRunLedger:
    return MainRunLedger.create(tmp_path / "main-run.sqlite3", _binding(), _units())


@pytest.fixture(autouse=True)
def reject_repository_artifact_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    read_bytes = Path.read_bytes
    read_text = Path.read_text

    def guarded_bytes(path: Path) -> bytes:
        assert not path.absolute().is_relative_to(ROOT) or path.is_relative_to(tmp_path), path
        return read_bytes(path)

    def guarded_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        assert not path.absolute().is_relative_to(ROOT) or path.is_relative_to(tmp_path), path
        return read_text(path, encoding=encoding, errors=errors)

    def denied_descriptor_read(path: Path) -> bytes:
        raise AssertionError(f"repository artifact read: {path}")

    def synthetic_policy(_root: Path) -> HistoricalEvidencePolicy:
        return HistoricalEvidencePolicy()

    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(runner_models, "read_regular_nofollow", denied_descriptor_read)
    monkeypatch.setattr(cost_policy, "read_regular_nofollow", denied_descriptor_read)
    monkeypatch.setattr(evidence_module, "load_cost_policy_bundle", synthetic_policy)


def _tamper(path: Path, action: str) -> None:
    if action == "delete":
        path.unlink()
    else:
        path.write_bytes(b"{}")


def test_dispatch_outcomes_reject_invalid_digest_and_negative_cost() -> None:
    with pytest.raises(MainRunError, match="MAIN_RUN_EVIDENCE_INVALID"):
        DispatchCompleted("not-a-sha256", 0)
    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        DispatchCompleted("6" * 64, -1)
    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        DispatchTechnicalFailure("PROVIDER_QUOTA", "7" * 64, -1)


def test_negative_projected_cost_fails_before_intent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None

    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        ledger.claim_dispatch(unit.unit_id, -1, 0)

    assert ledger.status().in_flight_count == 0


def test_tranche_cannot_exceed_frozen_core_gate(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None

    with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
        ledger.claim_dispatch(unit.unit_id, unit.projected_cost_krw, 450001)

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


def test_completed_reconciliation_rejects_caller_digest_without_durable_unit_evidence(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    evidence = InFlightEvidence.completed(
        ledger.in_flight_context(unit.unit_id),
        "a" * 64,
        16,
    )

    with pytest.raises(MainRunError, match="MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID"):
        ledger.reconcile(unit.unit_id, evidence)

    assert ledger.status().in_flight_count == 1


def test_completed_reconciliation_accepts_exact_durable_unit_evidence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    assert unit.kind == "CLEAN_PREFIX"
    assert unit.memory_baseline == "fh_bounded"
    ledger.persist_dispatch_intent(unit.unit_id)
    completed = persist_unit_dispatch(tmp_path, unit, _prefix_output(unit))

    ledger.reconcile(
        unit.unit_id,
        InFlightEvidence.completed(
            ledger.in_flight_context(unit.unit_id),
            completed.evidence_sha256,
            completed.realized_cost_krw,
        ),
    )

    assert ledger.status().completed_count == 1


@pytest.mark.parametrize("action", ["delete", "replace"])
def test_completed_evidence_remains_part_of_ledger_integrity(
    tmp_path: Path,
    action: str,
) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    completed = persist_unit_dispatch(tmp_path, unit, _prefix_output(unit))
    ledger.persist_completed(unit.unit_id, completed)
    _tamper(tmp_path / "units" / f"{unit.sequence:06d}-{unit.unit_id}.json", action)

    with pytest.raises(MainRunError, match="MAIN_RUN_COMPLETION_EVIDENCE_INVALID"):
        ledger.status()


@pytest.mark.parametrize("action", ["delete", "replace"])
def test_reconciliation_evidence_remains_part_of_ledger_integrity(
    tmp_path: Path,
    action: str,
) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    ledger.persist_dispatch_intent(unit.unit_id)
    context = ledger.in_flight_context(unit.unit_id)
    evidence = persist_reconciliation_evidence(tmp_path, "NO_PROVIDER_REQUEST", context)
    ledger.reconcile(unit.unit_id, evidence)
    _tamper(tmp_path / "reconciliation" / f"{context.intent_event_hash}.json", action)

    with pytest.raises(MainRunError, match="MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID"):
        ledger.status()


def test_realized_overrun_is_preserved_and_pauses_before_next_dispatch(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    unit = ledger.next_pending()
    assert unit is not None
    assert ledger.claim_dispatch(unit.unit_id, unit.projected_cost_krw, 20)
    completed = persist_unit_dispatch(tmp_path, unit, _prefix_output(unit, cost_usd=0.015625))
    ledger.persist_completed(unit.unit_id, completed)

    assert ledger.status().realized_cost_krw == 25
    next_unit = ledger.next_pending()
    assert next_unit is not None
    assert not ledger.claim_dispatch(next_unit.unit_id, next_unit.projected_cost_krw, 20)
    assert ledger.status().session_state == "PAUSED_BEFORE_DISPATCH"


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


def test_synthetic_fixture_rejects_mixed_authority_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = SyntheticFixture()
    mixed = replace(fixture.authorization, authority_sha256="0" * 64)
    create = MainRunLedger.create
    creations: list[Path] = []

    def record_create(
        path: Path, binding: MainRunBinding, units: tuple[ProductionObject, ...],
    ) -> MainRunLedger:
        creations.append(path)
        return create(path, binding, units)

    monkeypatch.setattr(MainRunLedger, "create", record_create)
    with pytest.raises(MainRunError, match="^MAIN_AUTHORITY_BINDING_MISMATCH$"):
        rejected = replace(fixture, authorization=mixed)
        MainRunLedger.create(tmp_path / "rejected.sqlite3", rejected.binding(), rejected.units())
    assert creations == []
