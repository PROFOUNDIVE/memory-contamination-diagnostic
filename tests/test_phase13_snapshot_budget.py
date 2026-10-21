from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict, replace
from multiprocessing.synchronize import Barrier
from pathlib import Path

if __name__ == "__main__":
    __package__ = "tests"

import pytest

import memcontam.readiness.phase13_main_live_evidence as evidence_module
from memcontam.readiness.phase13_main_live_dispatch import (
    MainLiveDispatchError, persist_unit_dispatch,
)
from memcontam.readiness.phase13_main_runner_ledger import MainRunLedger
from memcontam.readiness.phase13_main_runner_models import (
    InFlightEvidence, MainRunBinding, MainRunError, units_sha256,
)
from .phase13_v3_fixtures import HistoricalEvidencePolicy, SyntheticFixture, prefix_output


def _binding(cap: int = 20) -> MainRunBinding:
    return replace(SyntheticFixture().binding(), core_authorization_gate_krw=cap)


def _policy(_root: Path) -> HistoricalEvidencePolicy:
    return HistoricalEvidencePolicy()


@pytest.fixture(autouse=True)
def synthetic_evidence_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evidence_module, "load_cost_policy_bundle", _policy)


def _race_claim(root: Path, barrier: Barrier) -> None:
    unit = SyntheticFixture().units()[0]
    ledger = MainRunLedger.open(root / "ledger.sqlite3", _binding(), SyntheticFixture().units())
    barrier.wait(timeout=20)
    try:
        result = str(ledger.claim_dispatch(unit.unit_id, unit.projected_cost_krw, 10))
    except MainRunError as error:
        result = error.code
    (root / f"race-{os.getpid()}.json").write_text(json.dumps({"pid": os.getpid(), "result": result}))


def _child_reopen(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-m", "tests.phase13_recovery_offline", str(Path(__file__).resolve())],
        env={**os.environ, "TASK12_REOPEN_ROOT": str(root)},
        capture_output=True, text=True, check=True, timeout=30,
    )


def test_one_durable_intent_when_two_processes_race(tmp_path: Path) -> None:
    """Given one fixture allowance, when two owners race, then only one reserves."""
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), SyntheticFixture().units())
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(3)
    workers = [context.Process(target=_race_claim, args=(tmp_path, barrier)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        barrier.wait(timeout=20)
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            worker.close()
    outcomes = [json.loads(path.read_text()) for path in tmp_path.glob("race-*.json")]
    assert sorted(row["result"] for row in outcomes) == [
        "MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED", "True",
    ]
    assert len({row["pid"] for row in outcomes}) == 2
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT kind FROM events").fetchall() == [("INTENT",)]
    assert ledger.status().in_flight_count == 1
    reopened = json.loads(_child_reopen(tmp_path).stdout)
    assert reopened["status"]["in_flight_count"] == 1
    assert reopened["status"]["realized_cost_krw"] == 0


def test_completion_survives_new_process_without_replay_or_double_charge(tmp_path: Path) -> None:
    """Given durable fake evidence, when reopened, then receipt and identity survive."""
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), units)
    assert ledger.claim_dispatch(units[0].unit_id, units[0].projected_cost_krw, 20)
    completed = persist_unit_dispatch(tmp_path, units[0], prefix_output(units[0]))
    ledger.persist_completed(units[0].unit_id, completed)
    result = json.loads(_child_reopen(tmp_path).stdout)
    assert result["pid"] != os.getpid()
    assert result["receipt"] == asdict(completed)
    assert result["binding"] == asdict(_binding())
    assert result["unit"] == asdict(units[0])
    assert result["next_pending"] == units[1].unit_id
    assert result["status"]["completed_count"] == 1
    assert result["status"]["realized_cost_krw"] == 16
    request = prefix_output(units[0]).provider_calls[0].provider_request_contract
    assert request is not None
    assert result["request_sha256"] == request["input_sha256"]
    before = ledger.status()
    with pytest.raises(MainRunError, match="MAIN_RUN_STATE_TRANSITION_INVALID"):
        ledger.persist_completed(units[0].unit_id, completed)
    with pytest.raises(MainRunError, match="MAIN_RUN_DISPATCH_ORDER_INVALID"):
        ledger.claim_dispatch(units[0].unit_id, 10, 20)
    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_EVIDENCE_ALREADY_EXISTS"):
        persist_unit_dispatch(tmp_path, units[0], prefix_output(units[0]))
    assert ledger.status() == before
    assert not ledger.claim_dispatch(units[1].unit_id, 10, 20)
    assert ledger.status().realized_cost_krw == 16


def test_ambiguous_intent_requires_reconciliation_after_new_process(tmp_path: Path) -> None:
    """Given an owner exit after intent, when reopened, then retry stays blocked."""
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), units)
    assert ledger.claim_dispatch(units[0].unit_id, 10, 20)
    reopened = json.loads(_child_reopen(tmp_path).stdout)
    assert reopened["status"]["session_state"] == "BLOCKED_IN_FLIGHT"
    assert reopened["intent"] == asdict(ledger.in_flight_context(units[0].unit_id))
    before = ledger.status()
    with pytest.raises(MainRunError, match="MAIN_RUN_IN_FLIGHT_AMBIGUOUS"):
        ledger.reconcile(units[0].unit_id, InFlightEvidence.ambiguous(
            ledger.in_flight_context(units[0].unit_id), "a" * 64,
        ))
    assert ledger.status() == before


@pytest.mark.parametrize("cap", [0, 450000])
def test_real_preparation_zero_cap_never_inherits_main_allowance(tmp_path: Path, cap: int) -> None:
    """Given explicit prep zero, when claiming cost or Main cap, then no intent."""
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(0), units)
    if cap:
        with pytest.raises(MainRunError, match="MAIN_RUN_COST_INVALID"):
            ledger.claim_dispatch(units[0].unit_id, 10, cap)
    else:
        assert not ledger.claim_dispatch(units[0].unit_id, 10, cap)
    with pytest.raises(MainRunError, match="MAIN_RUN_BINDING_MISMATCH"):
        MainRunLedger.open(ledger.path, SyntheticFixture().binding(), units)
    assert ledger.status().in_flight_count == ledger.status().realized_cost_krw == 0


@pytest.mark.parametrize("field", ["package_sha256", "authorization_sha256", "runner_sha256"])
def test_wrong_binding_refuses_before_accounting(tmp_path: Path, field: str) -> None:
    """Given an existing ledger, when a source/package binding changes, then refuse."""
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), units)
    with pytest.raises(MainRunError, match="MAIN_RUN_BINDING_MISMATCH"):
        MainRunLedger.open(ledger.path, replace(_binding(), **{field: "f" * 64}), units)
    assert ledger.status().realized_cost_krw == ledger.status().in_flight_count == 0


@pytest.mark.parametrize("field", [
    "unit_id", "registration_packet_sha256", "ordered_sample_ids_sha256",
    "execution_template_id", "checkpoint_registry_sha256",
])
def test_wrong_unit_identity_refuses_before_accounting(tmp_path: Path, field: str) -> None:
    """Given frozen units, when unit or registration input changes, then refuse."""
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), units)
    wrong = (replace(units[0], **{field: "f" * 64}), units[1])
    with pytest.raises(MainRunError, match="MAIN_RUN_BINDING_MISMATCH"):
        MainRunLedger.open(ledger.path, _binding(), wrong)
    assert ledger.status().realized_cost_krw == ledger.status().in_flight_count == 0


@pytest.mark.parametrize("in_flight", [False, True])
@pytest.mark.parametrize("value", [None, "f" * 64])
def test_missing_or_wrong_supplemental_binding_refuses_without_migration(
    tmp_path: Path, in_flight: bool, value: str | None,
) -> None:
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), units)
    if in_flight:
        assert ledger.claim_dispatch(units[0].unit_id, 10, 20)
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("DELETE FROM metadata WHERE key = 'full_units_identity_sha256'")
        if value is not None:
            connection.execute("INSERT INTO metadata VALUES ('full_units_identity_sha256', ?)", (value,))
    before = ledger.path.read_bytes()
    code = "MAIN_RUN_UNIT_IDENTITY_BINDING_MISSING" if value is None else "MAIN_RUN_BINDING_MISMATCH"
    with pytest.raises(MainRunError, match=code):
        ledger.status()
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _child_reopen(tmp_path)
    assert code in caught.value.stderr
    assert ledger.path.read_bytes() == before


def test_new_metadata_keeps_legacy_units_digest_unchanged(tmp_path: Path) -> None:
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), units)
    with sqlite3.connect(ledger.path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    assert metadata["units_sha256"] == units_sha256(units) == "af76a84ec2293ea5c89525a33dad1be34e13b1ba1bf1b5a312754e909149f75f"
    canonical = json.dumps([asdict(unit) for unit in units], sort_keys=True, separators=(",", ":")).encode()
    assert metadata["full_units_identity_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_corrupt_chain_refuses_in_new_process(tmp_path: Path) -> None:
    """Given a durable intent, when its chain changes, then reopen refuses."""
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), units)
    assert ledger.claim_dispatch(units[0].unit_id, 10, 20)
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("UPDATE events SET previous_hash = ?", ("0" * 64,))
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _child_reopen(tmp_path)
    assert "MAIN_RUN_LEDGER_INTEGRITY_INVALID" in caught.value.stderr


def test_wrong_request_and_unit_receipt_cannot_complete(tmp_path: Path) -> None:
    """Given an intent, when request bytes or receipt unit differ, then no charge."""
    units = SyntheticFixture().units()
    ledger = MainRunLedger.create(tmp_path / "ledger.sqlite3", _binding(), units)
    assert ledger.claim_dispatch(units[0].unit_id, 10, 20)
    output = prefix_output(units[0])
    call = output.provider_calls[0].model_copy(update={"messages": [{"role": "user", "content": "swapped"}]})
    with pytest.raises(MainLiveDispatchError):
        persist_unit_dispatch(tmp_path, units[0], output.model_copy(update={"provider_calls": (call,)}))
    completed = persist_unit_dispatch(tmp_path, units[0], output)
    with pytest.raises(MainRunError, match="MAIN_RUN_COMPLETION_EVIDENCE_INVALID"):
        ledger.persist_completed(units[1].unit_id, completed)
    assert ledger.status().realized_cost_krw == 0
    assert ledger.status().in_flight_count == 1


if __name__ == "__main__":
    root = Path(os.environ["TASK12_REOPEN_ROOT"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evidence_module, "load_cost_policy_bundle", _policy)
        units = SyntheticFixture().units()
        ledger = MainRunLedger.open(root / "ledger.sqlite3", _binding(), units)
        status = ledger.status()
        report = {"pid": os.getpid(), "status": asdict(status), "binding": asdict(_binding())}
        if status.in_flight_count:
            report["intent"] = asdict(ledger.in_flight_context(units[0].unit_id))
            for unit in units:
                with pytest.raises(MainRunError, match="MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"):
                    ledger.claim_dispatch(unit.unit_id, 10, 20)
        if status.completed_count:
            receipt = root / "units" / f"000000-{units[0].unit_id}.json"
            raw = json.loads(receipt.read_text())
            pending = ledger.next_pending()
            assert pending is not None
            report.update(receipt={"evidence_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                   "realized_cost_krw": raw["realized_cost_krw"]},
                          unit=asdict(units[0]), next_pending=pending.unit_id,
                          request_sha256=raw["provider_calls"][0]["provider_request_contract"]["input_sha256"])
            with pytest.raises(MainRunError, match="MAIN_RUN_DISPATCH_ORDER_INVALID"):
                ledger.claim_dispatch(units[0].unit_id, 10, 20)
        print(json.dumps(report, sort_keys=True))
