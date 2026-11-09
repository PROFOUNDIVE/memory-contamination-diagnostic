from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_dispatch
from memcontam.readiness.phase13_main_live_evidence import MainRuntimeEvidence
from memcontam.readiness.phase13_live_diagnostic import LiveDiagnosticLedger

from .phase13_snapshot_durable_fixtures import native_snapshot
from .test_phase13_snapshot_bot import capacity_fixture as capacity_fixture
from .phase13_live_diagnostic_fixtures import bundle, record_calls


def _intent(binding, request_id):
    from memcontam.readiness.phase13_live_diagnostic_models import DiagnosticRequestIntent
    return DiagnosticRequestIntent(request_id=request_id,
        attempt_id=hashlib.sha256((binding.sha256() + request_id).encode()).hexdigest(),
        request_sha256="a" * 64, reservation_id="qa-reservation")


def test_live_scope_is_distinct_from_zero_only_qa(tmp_path: Path) -> None:
    _, output = snapshot_dispatch(native_snapshot(tmp_path, "bot_style"))
    raw = output.model_dump(mode="json")["evidence"]["runtime_evidence"]
    scope = raw["diagnostic_scope"]
    del scope["real_spend_cap_krw"]
    scope.update(provenance="LIVE_DIAGNOSTIC", main_scope="NOT_MAIN",
                 binding_sha256="a" * 64, child_execution_id="bot-child",
                 scientific_disposition="NOT_ADOPTED")

    parsed = MainRuntimeEvidence.model_validate_json(json.dumps(raw))

    assert parsed.diagnostic_scope is not None
    assert parsed.diagnostic_scope.provenance == "LIVE_DIAGNOSTIC"
    assert "real_spend_cap_krw" not in parsed.diagnostic_scope.model_dump()


@pytest.mark.parametrize("baseline", ["bot_style", "dc_rs"])
@pytest.mark.parametrize("terminal_cost", [None, "paid", "unknown"])
def test_native_completed_and_partial_reopen(tmp_path: Path, baseline: str, terminal_cost: str | None) -> None:

    binding, units, outputs = bundle(tmp_path, (baseline,), terminal_cost)
    ledger = LiveDiagnosticLedger.create(binding, units)
    ledger.claim(units[0].unit_id)
    record_calls(ledger, units[0], outputs[0], binding)
    digest = ledger.persist_output(units[0].unit_id, outputs[0])
    (tmp_path / "reopen.json").write_text(json.dumps({"binding": binding.model_dump(mode="json"),
        "units": [asdict(unit) for unit in units], "digests": [digest]}))

    child = subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline",
        str(Path(__file__).with_name("phase13_live_diagnostic_fixtures.py"))],
        env={**os.environ, "TASK8_ROOT": str(tmp_path), "TASK8_RECONCILE": "1"},
        capture_output=True, text=True, timeout=30)

    assert child.returncode == 0, child.stderr
    status = json.loads(child.stdout)
    assert status["pid"] != os.getpid()
    assert status["completed"] == int(terminal_cost is None)
    assert status["terminal"] == int(terminal_cost is not None)
    assert status["unknown_cost_requests"] == int(terminal_cost == "unknown")
    assert status["known_cost_usd_micros"] == sum(10000 for call in outputs[0].provider_calls if call.provider_cost_usd is not None)
    closed = subprocess.run(child.args, env={**os.environ, "TASK8_ROOT": str(tmp_path), "TASK8_RECONCILE": "0"},
        capture_output=True, text=True, timeout=30)
    assert closed.returncode == 0, closed.stderr
    assert {key: value for key, value in json.loads(closed.stdout).items() if key != "pid"} == {
        key: value for key, value in status.items() if key != "pid"}
    with pytest.raises(ValueError):
        ledger.claim(units[0].unit_id)


def test_two_child_processes_share_authority_while_preserving_local_order(tmp_path: Path) -> None:

    binding, units, outputs = bundle(tmp_path, ("bot_style", "dc_rs"))
    ledger = LiveDiagnosticLedger.create(binding, units)
    (tmp_path / "reopen.json").write_text(json.dumps({"binding": binding.model_dump(mode="json"),
        "units": [asdict(unit) for unit in units], "outputs": [output.model_dump(mode="json") for output in outputs]}))
    children = []
    try:
        for index in range(2):
            children.append(subprocess.Popen([sys.executable, "-B", "-m", "tests.phase13_recovery_offline",
                str(Path(__file__).with_name("phase13_live_diagnostic_fixtures.py"))],
                env={**os.environ, "TASK8_ROOT": str(tmp_path), "TASK8_CHILD": str(index)},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for child in children:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "CLAIMED"
        assert ledger.status()["in_flight"] == 2
        for unit, observation in zip(units, binding.observations, strict=True):
            with pytest.raises(ValueError, match="ORDER_OR_AMBIGUOUS"):
                ledger.claim_request(unit.unit_id, _intent(binding, observation.requests[1].request_id))
        for child in children:
            assert child.stdin is not None
            child.stdin.write("continue\n")
            child.stdin.flush()
        for child in children:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "REQUEST_CLAIMED"
        assert ledger.status()["unresolved_requests"] == 2
        for child in children:
            assert child.stdin is not None
            child.stdin.write("continue\n")
            child.stdin.flush()
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stderr
            assert json.loads(stdout)["completed"] >= 1
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    assert ledger.status()["completed"] == 2
    assert ledger.status()["known_cost_usd_micros"] == 50000
    assert list(tmp_path.rglob("ledger.sqlite3")) == [tmp_path / "ledger.sqlite3"]


@pytest.mark.parametrize("field", ["authorization_sha256", "registration_sha256", "source_revision", "run_id", "ledger_path"])
def test_forged_binding_refused_on_reopen(tmp_path: Path, field: str) -> None:
    binding, units, _ = bundle(tmp_path)
    LiveDiagnosticLedger.create(binding, units)
    changed = binding.model_copy(update={field: "f" * (40 if field == "source_revision" else 64)})
    with pytest.raises(ValueError):
        LiveDiagnosticLedger.open(changed, units)


def test_ambiguous_request_intent_cannot_be_redispatched(tmp_path: Path) -> None:
    binding, units, _ = bundle(tmp_path)
    ledger = LiveDiagnosticLedger.create(binding, units)
    ledger.claim(units[0].unit_id)
    intent = _intent(binding, binding.observations[0].requests[0].request_id)
    ledger.claim_request(units[0].unit_id, intent)
    reopened = LiveDiagnosticLedger.open(binding, units)
    for request in binding.observations[0].requests:
        with pytest.raises(ValueError, match="ORDER_OR_AMBIGUOUS"):
            reopened.claim_request(units[0].unit_id, _intent(binding, request.request_id))
    assert reopened.status()["unresolved_requests"] == 1


@pytest.mark.parametrize("mutation", ["bytes", "answer", "cost", "scope"])
def test_changed_outer_or_semantic_hash_refused(tmp_path: Path, mutation: str) -> None:
    binding, units, outputs = bundle(tmp_path)
    ledger = LiveDiagnosticLedger.create(binding, units)
    ledger.claim(units[0].unit_id)
    record_calls(ledger, units[0], outputs[0], binding)
    digest = ledger.persist_output(units[0].unit_id, outputs[0])
    path = next((tmp_path / "units").glob("*.json"))
    raw = json.loads(path.read_bytes())
    if mutation == "answer":
        raw["output"]["evidence"]["runtime_evidence"]["production_observability_archive"]["records"][0]["evidence"]["target_set"]["answer_call_id"] = "forged"
    elif mutation == "cost":
        raw["output"]["realized_cost_usd_micros"] = 0
    elif mutation == "scope":
        raw["binding_sha256"] = "f" * 64
    path.write_bytes(json.dumps(raw).encode() + b" ")
    if mutation != "bytes":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        ledger.reconcile(units[0].unit_id, digest)
    assert ledger.status()["in_flight"] == 1


def test_live_and_main_paths_refuse_cross_scope(tmp_path: Path) -> None:
    from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput, persist_unit_dispatch
    from memcontam.readiness.phase13_main_runner_ledger import MainRunLedger
    from .phase13_v3_fixtures import SyntheticFixture
    binding, units, outputs = bundle(tmp_path)
    LiveDiagnosticLedger.create(binding, units)
    with pytest.raises(ValueError):
        MainRunLedger.open(tmp_path / "ledger.sqlite3", SyntheticFixture().binding(), units)
    with pytest.raises(ValueError):
        persist_unit_dispatch(tmp_path, units[0], MainUnitDispatchOutput(evidence=outputs[0].evidence,
            provider_calls=outputs[0].provider_calls, realized_cost_krw=0))
    assert not (tmp_path / "units").exists()


@pytest.mark.parametrize("currency", ["KRW", "usd", "EUR"])
def test_wrong_currency_is_rejected(tmp_path: Path, currency: str) -> None:
    from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticBinding, LiveDiagnosticOutput
    binding, _, outputs = bundle(tmp_path)
    for model in (binding, outputs[0]):
        raw = model.model_dump(mode="json")
        raw["currency"] = currency
        with pytest.raises(ValueError):
            (LiveDiagnosticBinding if model is binding else LiveDiagnosticOutput).model_validate_json(json.dumps(raw))


def test_duplicate_request_registration_is_rejected(tmp_path: Path) -> None:
    from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticBinding
    binding, _, _ = bundle(tmp_path)
    raw = binding.model_dump(mode="json")
    raw["observations"][0]["requests"][1]["request_id"] = raw["observations"][0]["requests"][0]["request_id"]
    with pytest.raises(ValueError, match="DUPLICATE_IDENTITY"):
        LiveDiagnosticBinding.model_validate_json(json.dumps(raw))


def test_count_attempt_and_receipt_share_the_executor_ledger(tmp_path: Path) -> None:
    from memcontam.readiness.phase13_live_diagnostic_models import DiagnosticRequestReceipt
    binding, units, outputs = bundle(tmp_path, count=True)
    ledger = LiveDiagnosticLedger.create(binding, units)
    ledger.claim(units[0].unit_id)
    intent = _intent(binding, binding.observations[0].requests[0].request_id)
    ledger.claim_request(units[0].unit_id, intent)
    receipt = DiagnosticRequestReceipt(intent=intent, receipt_sha256="b" * 64,
        realized_cost_usd_micros=5000, outcome="completed")
    ledger.record_request(units[0].unit_id, receipt)
    with pytest.raises(ValueError):
        ledger.record_request(units[0].unit_id, receipt)
    record_calls(ledger, units[0], outputs[0], binding)
    ledger.reconcile(units[0].unit_id, ledger.persist_output(units[0].unit_id, outputs[0]))
    assert LiveDiagnosticLedger.open(binding, units).status()["known_cost_usd_micros"] == 35000


def test_source_drift_refuses_before_intent(tmp_path: Path) -> None:
    from memcontam.readiness.phase13_durable_call_join import DiagnosticFile
    binding, units, _ = bundle(tmp_path)
    source = tmp_path / "source.py"
    source.write_bytes(b"approved bytes")
    binding = binding.model_copy(update={"source_files": (*binding.source_files,
        DiagnosticFile(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))})
    ledger = LiveDiagnosticLedger.create(binding, units)
    source.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="SOURCE"):
        ledger.claim(units[0].unit_id)


def test_main_serializer_and_metadata_are_unchanged() -> None:
    from memcontam.readiness.phase13_main_runner_store import expected_metadata, _full_units_identity_sha256
    from memcontam.readiness.phase13_main_production import units_sha256
    from .phase13_v3_fixtures import SyntheticFixture
    fixture = SyntheticFixture()
    binding, units = fixture.binding(), fixture.units()
    expected = {"ledger_schema_id": "phase13-main-a-ledger-v1", "units_sha256": units_sha256(units),
        "full_units_identity_sha256": _full_units_identity_sha256(units),
        **{key: str(value) for key, value in asdict(binding).items()}}
    assert expected_metadata(binding, units) == expected
    assert units_sha256(units) == "af76a84ec2293ea5c89525a33dad1be34e13b1ba1bf1b5a312754e909149f75f"


def test_unpinned_live_implementation_refuses_before_create(tmp_path: Path) -> None:
    binding, units, _ = bundle(tmp_path)
    changed = binding.model_copy(update={"source_files": tuple(pin for pin in binding.source_files
        if not pin.path.endswith("phase13_live_diagnostic.py"))})
    with pytest.raises(ValueError, match="SOURCE"):
        LiveDiagnosticLedger.create(changed, units)
    assert not (tmp_path / "ledger.sqlite3").exists()


@pytest.mark.parametrize("field,value", [("provider_cost_usd", 0.0), ("derived_cost_usd", 1.0),
    ("provider_cost_source", "INVENTED"), ("authoritative_provider_cost_usd", 1.0)])
def test_live_paid_cost_contradictions_refuse(tmp_path: Path, field: str, value) -> None:
    from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticOutput
    binding, units, outputs = bundle(tmp_path, terminal_cost="paid")
    ledger = LiveDiagnosticLedger.create(binding, units)
    ledger.claim(units[0].unit_id)
    record_calls(ledger, units[0], outputs[0], binding)
    raw = outputs[0].model_dump(mode="json")
    raw["provider_calls"][-1][field] = value
    with pytest.raises(ValueError):
        ledger.persist_output(units[0].unit_id, LiveDiagnosticOutput.model_validate_json(json.dumps(raw)))


@pytest.mark.parametrize("amount", [None, 5000])
def test_ambiguous_count_blocks_other_child_even_with_known_charge(tmp_path: Path, amount: int | None) -> None:
    from memcontam.readiness.phase13_live_diagnostic_models import DiagnosticRequestReceipt
    binding, units, _ = bundle(tmp_path, ("bot_style", "dc_rs"), count=True)
    ledger = LiveDiagnosticLedger.create(binding, units)
    ledger.claim(units[0].unit_id)
    ledger.claim(units[1].unit_id)
    intent = _intent(binding, binding.observations[0].requests[0].request_id)
    ledger.claim_request(units[0].unit_id, intent)
    ledger.record_request(units[0].unit_id, DiagnosticRequestReceipt(intent=intent,
        receipt_sha256="b" * 64, realized_cost_usd_micros=amount, outcome="ambiguous"))
    with pytest.raises(ValueError):
        ledger.claim_request(units[1].unit_id, _intent(binding, binding.observations[1].requests[0].request_id))


def test_paid_request_cannot_claim_zero_without_charge_evidence() -> None:
    from memcontam.readiness.phase13_live_diagnostic_models import DiagnosticRequestIntent, DiagnosticRequestReceipt
    intent = DiagnosticRequestIntent(request_id="count", attempt_id="a" * 64,
        request_sha256="b" * 64, reservation_id="qa-reservation")
    with pytest.raises(ValueError):
        DiagnosticRequestReceipt(intent=intent, receipt_sha256="c" * 64,
            realized_cost_usd_micros=0, outcome="completed")
