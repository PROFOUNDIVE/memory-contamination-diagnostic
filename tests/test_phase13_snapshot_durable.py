import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from memcontam.readiness.phase13_main_live_dispatch import MainLiveDispatchError, persist_unit_dispatch, persist_reconciliation_evidence
from memcontam.readiness.phase13_main_live_evidence import _completed_call, _parse_evidence, _reconciled_cost
from memcontam.readiness.phase13_main_runner_ledger import MainRunLedger
from memcontam.readiness.phase13_main_runner_models import MainRunError
from memcontam.readiness.phase13_main_production import units_sha256

from .phase13_snapshot_durable_fixtures import ContractTransport, native_snapshot, ordinary_output
from .test_phase13_snapshot_bot import capacity_fixture as capacity_fixture
from .phase13_v3_fixtures import SyntheticFixture


@pytest.mark.parametrize("baseline", ["bot_style", "dc_rs"])
def test_one_step_requires_explicit_diagnostic_cardinality(tmp_path: Path, baseline: str) -> None:
    snapshot = native_snapshot(tmp_path, baseline)
    unit, output = ordinary_output(snapshot)

    evidence = _parse_evidence(unit, output.evidence)
    assert all(_completed_call(call, evidence.runtime_evidence.request, None) and _reconciled_cost(call)
               for call in output.provider_calls)
    with pytest.raises(MainLiveDispatchError, match="MAIN_UNIT_PROVIDER_CALLS_INVALID"):
        persist_unit_dispatch(tmp_path, unit, output)


def _bound(snapshot):
    from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_dispatch
    return snapshot_dispatch(snapshot)


def _ledger(root, unit):
    binding = replace(SyntheticFixture().binding(), core_authorization_gate_krw=0)
    (root / "reopen.json").write_text(json.dumps({"unit": asdict(unit), "binding": asdict(binding)}))
    ledger = MainRunLedger.create(root / "ledger.sqlite3", binding, (unit,))
    assert ledger.claim_dispatch(unit.unit_id, 0, 0)
    return ledger


def _reopen(root: Path, *, reconcile: bool = False, load_only: bool = False):
    script = Path(__file__).with_name("phase13_snapshot_durable_fixtures.py")
    return subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline", str(script)],
        env={**os.environ, "TASK14_REOPEN_ROOT": str(root), "TASK14_RECONCILE": str(int(reconcile)),
             "TASK14_LOAD_ONLY": str(int(load_only))},
        capture_output=True, text=True, timeout=30, check=False)


@pytest.mark.parametrize("baseline", ["bot_style", "dc_rs"])
def test_native_snapshot_survives_fresh_process(tmp_path: Path, baseline: str) -> None:
    snapshot = native_snapshot(tmp_path, baseline)
    unit, output = _bound(snapshot)
    ledger = _ledger(tmp_path, unit)
    receipt = persist_unit_dispatch(tmp_path, unit, output)
    ledger.persist_completed(unit.unit_id, receipt)

    child = _reopen(tmp_path)

    assert child.returncode == 0, child.stderr
    report = json.loads(child.stdout)
    assert report["pid"] != os.getpid() and report["status"]["completed_count"] == 1
    assert report["archive"] == snapshot.archive.model_dump(mode="json")
    assert report["scope_sha256"] == unit.diagnostic_scope_sha256
    with pytest.raises(MainLiveDispatchError, match="ALREADY_EXISTS"):
        persist_unit_dispatch(tmp_path, unit, output)
    with pytest.raises(MainRunError):
        ledger.claim_dispatch(unit.unit_id, 0, 0)


@pytest.mark.parametrize("field,value", [
    ("diagnostic_trial_count", None), ("diagnostic_trial_count", 2), ("diagnostic_trial_count", True),
    ("diagnostic_scope_sha256", None), ("diagnostic_scope_sha256", "bad"),
    ("diagnostic_scope_sha256", "0" * 64),
])
def test_invalid_pair_refuses_before_write(tmp_path: Path, field: str, value) -> None:
    unit, output = _bound(native_snapshot(tmp_path, "bot_style"))
    with pytest.raises((ValueError, MainLiveDispatchError)):
        persist_unit_dispatch(tmp_path, replace(unit, **{field: value}), output)
    assert not (tmp_path / "units").exists()


@pytest.mark.parametrize("mutation", ["answer", "order", "request", "task", "source", "verifier", "scientific", "archive", "state", "spans"])
def test_join_forgery_refuses_even_with_valid_outer_payload(tmp_path: Path, mutation: str) -> None:
    unit, output = _bound(native_snapshot(tmp_path, "dc_rs"))
    raw = output.model_dump(mode="json")
    runtime = raw["evidence"]["runtime_evidence"]
    row = runtime["production_observability_archive"]["records"][0]
    if mutation == "answer":
        row["evidence"]["target_set"]["answer_call_id"] = row["method_calls"][0]["call_id"]
    elif mutation == "order":
        row["evidence"]["order_key"] += 1
    elif mutation == "request":
        raw["provider_calls"][-1]["messages"][0]["content"] = "wrong input"
    elif mutation == "task":
        row["task_instance"]["input"]["words"] = ["wrong"]
    elif mutation == "source":
        runtime["diagnostic_scope"]["source_files"] = []
    elif mutation == "verifier":
        row["evidence"]["verified_outcome"] = 0
    elif mutation == "scientific":
        runtime["production_identity"]["scientific_result"] = True
    elif mutation == "state":
        runtime["diagnostic_scope"]["state_utf8"] = "{}"
    elif mutation == "spans":
        row["evidence"]["target_set"]["answer_call_spans"] = []
    else:
        runtime["production_observability_archive"] = None
    with pytest.raises(ValueError):
        persist_unit_dispatch(tmp_path, unit, output.model_validate_json(json.dumps(raw)))


@pytest.mark.parametrize("mutation", ["scope", "count", "candidate", "bytes"])
def test_fresh_process_tamper_refuses(tmp_path: Path, mutation: str) -> None:
    snapshot = native_snapshot(tmp_path, "bot_style")
    unit, output = _bound(snapshot)
    ledger = _ledger(tmp_path, unit)
    ledger.persist_completed(unit.unit_id, persist_unit_dispatch(tmp_path, unit, output))
    if mutation in {"scope", "count"}:
        payload = json.loads((tmp_path / "reopen.json").read_text())
        payload["unit"]["diagnostic_scope_sha256" if mutation == "scope" else "diagnostic_trial_count"] = "f" * 64 if mutation == "scope" else 2
        (tmp_path / "reopen.json").write_text(json.dumps(payload))
    elif mutation == "candidate":
        assert snapshot.bindings.candidate is not None
        snapshot.bindings.candidate.path.write_text("tampered")
    else:
        path = next((tmp_path / "units").glob("*.json"))
        path.write_bytes(path.read_bytes() + b" ")

    child = _reopen(tmp_path)

    assert child.returncode != 0 and ("INVALID" in child.stderr or "MISMATCH" in child.stderr)


@pytest.mark.parametrize("baseline,stage", [("bot_style", "bot_instantiate_solve"), ("bot_style", "bot_thought_distill"), ("dc_rs", "dc_rs_generate")])
def test_terminal_partial_keeps_calls_without_completed_unit(tmp_path: Path, baseline: str, stage: str) -> None:
    snapshot = native_snapshot(tmp_path, baseline, stage)
    unit, output = _bound(snapshot)
    ledger = _ledger(tmp_path, unit)
    with pytest.raises(MainLiveDispatchError):
        persist_unit_dispatch(tmp_path, unit, output)
    receipt = persist_reconciliation_evidence(tmp_path, "TERMINAL_FAILURE", ledger.in_flight_context(unit.unit_id),
        failure_code=snapshot.archive.records[0].terminal_failure_code, diagnostic_unit=unit, diagnostic_output=output)
    ledger.reconcile(unit.unit_id, receipt)

    child = _reopen(tmp_path)

    assert child.returncode == 0, child.stderr
    report = json.loads(child.stdout)
    assert report["status"]["completed_count"] == 0
    assert report["status"]["terminal_technical_missing_count"] == 1
    assert report["status"]["session_state"] == "STOPPED_TERMINAL_TECHNICAL_MISSING"
    assert report["archive"] == snapshot.archive.model_dump(mode="json")
    assert not (tmp_path / "units").exists()
    assert snapshot.archive.records[0].evidence.verified_outcome is None


@pytest.mark.parametrize("partial", [False, True])
def test_crash_after_evidence_reconciles_without_redispatch(tmp_path: Path, partial: bool) -> None:
    snapshot = native_snapshot(tmp_path, "bot_style", "bot_instantiate_solve" if partial else None)
    unit, output = _bound(snapshot)
    ledger = _ledger(tmp_path, unit)
    if partial:
        persist_reconciliation_evidence(tmp_path, "TERMINAL_FAILURE", ledger.in_flight_context(unit.unit_id),
            failure_code=snapshot.archive.records[0].terminal_failure_code, diagnostic_unit=unit, diagnostic_output=output)
    else:
        persist_unit_dispatch(tmp_path, unit, output)
    assert isinstance(snapshot.run.client, ContractTransport)
    count = snapshot.run.client.calls

    child = _reopen(tmp_path, reconcile=True)

    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["status"]["in_flight_count"] == 0
    assert snapshot.run.client.calls == count


def test_default_hash_is_byte_identical_and_scope_changes_hash(tmp_path: Path) -> None:
    assert units_sha256(SyntheticFixture().units()) == "af76a84ec2293ea5c89525a33dad1be34e13b1ba1bf1b5a312754e909149f75f"
    unit, _ = _bound(native_snapshot(tmp_path, "bot_style"))
    assert units_sha256((unit,)) != units_sha256((replace(unit, diagnostic_scope_sha256="f" * 64),))


@pytest.mark.parametrize("mutation", ["missing_sources", "input", "source_digest", "prefix", "target_unit"])
def test_repinning_scope_does_not_replace_mechanical_validation(tmp_path: Path, mutation: str) -> None:
    from memcontam.readiness.phase13_durable_call_join import DiagnosticScope

    unit, output = _bound(native_snapshot(tmp_path, "bot_style"))
    raw = output.model_dump(mode="json")
    scope = raw["evidence"]["runtime_evidence"]["diagnostic_scope"]
    if mutation == "missing_sources":
        scope["source_files"] = []
    elif mutation == "source_digest":
        scope["source_files"][0]["sha256"] = "0" * 64
    elif mutation == "input":
        inputs = json.loads(scope["input_utf8"])
        inputs["tasks"][0]["input"]["numbers"] = [2, 2, 2, 2]
        scope["input_utf8"] = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
        scope["input_sha256"] = hashlib.sha256(scope["input_utf8"].encode()).hexdigest()
    digest = DiagnosticScope.model_validate_json(json.dumps(scope)).sha256()
    unit = replace(unit, diagnostic_scope_sha256=digest,
        unit_id=hashlib.sha256((scope["run_id"] + digest).encode()).hexdigest())
    if mutation == "target_unit":
        unit = replace(unit, unit_id="e" * 64)
    if mutation == "prefix":
        unit = replace(unit, prefix_unit_id="e" * 64)
        raw["evidence"]["prefix_unit_id"] = unit.prefix_unit_id
    raw["evidence"]["runtime_evidence"]["unit_id"] = unit.unit_id

    with pytest.raises(MainLiveDispatchError):
        persist_unit_dispatch(tmp_path, unit, output.model_validate_json(json.dumps(raw)))


@pytest.mark.parametrize("field", ["answer", "order", "request", "verifier", "source"])
def test_fresh_loader_checks_semantics_after_outer_hash_is_recomputed(tmp_path: Path, field: str) -> None:
    unit, output = _bound(native_snapshot(tmp_path, "dc_rs"))
    _ledger(tmp_path, unit)
    persist_unit_dispatch(tmp_path, unit, output)
    path = next((tmp_path / "units").glob("*.json"))
    raw = json.loads(path.read_text())
    runtime = raw["evidence"]["runtime_evidence"]
    row = runtime["production_observability_archive"]["records"][0]
    if field == "answer":
        row["evidence"]["target_set"]["answer_call_id"] = row["method_calls"][0]["call_id"]
    elif field == "order":
        row["evidence"]["order_key"] += 1
    elif field == "request":
        for call in (row["method_calls"][-1], raw["provider_calls"][-1]):
            call["messages"][0]["content"] = "wrong input"
    elif field == "verifier":
        row["evidence"]["verified_outcome"] = 0
    else:
        runtime["diagnostic_scope"]["source_files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(raw))

    child = _reopen(tmp_path, load_only=True)

    assert child.returncode != 0 and "MAIN_DIAGNOSTIC_SCOPE_INVALID" in child.stderr


@pytest.mark.parametrize("field,value", [("provider_status", "completed"), ("provider_response_status", "in_progress"),
    ("model", "wrong-model"), ("failure_code", "INPUT_ENVELOPE_EXCEEDED"), ("taxonomy", "invented")])
def test_partial_refuses_contradictory_terminal_failure(tmp_path: Path, field: str, value: str) -> None:
    from memcontam.logging.schema import MethodCall
    from memcontam.readiness.phase13_production_observability import terminal_provider_evidence

    unit, output = _bound(native_snapshot(tmp_path, "bot_style", "bot_instantiate_solve"))
    ledger = _ledger(tmp_path, unit)
    raw = output.model_dump(mode="json")
    row = raw["evidence"]["runtime_evidence"]["production_observability_archive"]["records"][0]
    if field == "taxonomy":
        row["terminal_failure_code"] = value
    else:
        for call in (row["method_calls"][-1], row["terminal_method_call"], raw["provider_calls"][-1]):
            call[field] = value
    row["terminal_provider_evidence"] = terminal_provider_evidence(
        MethodCall.model_validate_json(json.dumps(row["terminal_method_call"])), row["terminal_failure_code"]).model_dump(mode="json")
    with pytest.raises(ValueError):
        persist_reconciliation_evidence(tmp_path, "TERMINAL_FAILURE", ledger.in_flight_context(unit.unit_id),
            failure_code=row["terminal_failure_code"], diagnostic_unit=unit,
            diagnostic_output=output.model_validate_json(json.dumps(raw)))
