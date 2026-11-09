from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import pytest

from memcontam.readiness.phase13_live_diagnostic import LiveDiagnosticLedger
from memcontam.readiness.phase13_live_diagnostic_models import (
    DiagnosticRequest, DiagnosticRequestIntent, DiagnosticRequestReceipt,
    LiveDiagnosticBinding, LiveDiagnosticEvidence, LiveDiagnosticError,
    LiveDiagnosticOutput, LiveDiagnosticScope,
)
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_runner_store import append_event, connect

from .phase13_live_diagnostic_fixtures import bundle, record_calls
from .test_phase13_snapshot_bot import capacity_fixture as capacity_fixture


def _trailing_bundle(root: Path, operation: Literal["count", "prefix", "setup"]) -> tuple[
    LiveDiagnosticBinding, tuple[ProductionObject, ...], tuple[LiveDiagnosticOutput, ...]
]:
    binding, units, outputs = bundle(root)
    observation = binding.observations[0]
    trailing = DiagnosticRequest(request_id="tail-operation", operation=operation, stage="postflight")
    binding = binding.model_copy(update={"observations": (observation.model_copy(
        update={"requests": (*observation.requests, trailing)}),)})
    raw = outputs[0].model_dump(mode="json")
    scope = raw["evidence"]["runtime_evidence"]["diagnostic_scope"]
    scope["binding_sha256"] = binding.sha256()
    scope_hash = LiveDiagnosticScope.model_validate_json(json.dumps(scope)).sha256()
    unit = replace(units[0], diagnostic_scope_sha256=scope_hash,
        unit_id=hashlib.sha256((scope["run_id"] + scope_hash).encode()).hexdigest())
    raw["evidence"]["runtime_evidence"]["unit_id"] = unit.unit_id
    return binding, (unit,), (LiveDiagnosticOutput.model_validate_json(json.dumps(raw)),)


def _reopen(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline",
        str(Path(__file__).with_name("phase13_live_diagnostic_fixtures.py"))],
        env={**os.environ, "TASK8_ROOT": str(root), "TASK8_RECONCILE": "0"},
        capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("operation", ["count", "prefix", "setup"])
@pytest.mark.parametrize("outcome,amount", [
    ("ambiguous", None), ("ambiguous", 5000), ("failed", 5000), ("completed", None),
])
def test_completion_refuses_incomplete_required_operation(
    tmp_path: Path, operation: Literal["count", "prefix", "setup"],
    outcome: Literal["ambiguous", "failed", "completed"], amount: int | None,
) -> None:
    binding, units, outputs = _trailing_bundle(tmp_path, operation)
    unit, output = units[0], outputs[0]
    ledger = LiveDiagnosticLedger.create(binding, units)
    ledger.claim(unit.unit_id)
    record_calls(ledger, unit, output, binding)
    intent = DiagnosticRequestIntent(request_id="tail-operation",
        attempt_id=hashlib.sha256((binding.sha256() + "tail-operation").encode()).hexdigest(),
        request_sha256="a" * 64, reservation_id="qa-only-tail")
    ledger.claim_request(unit.unit_id, intent)
    ledger.record_request(unit.unit_id, DiagnosticRequestReceipt(intent=intent,
        receipt_sha256="b" * 64, realized_cost_usd_micros=amount, outcome=outcome))

    with pytest.raises(LiveDiagnosticError, match="DIAGNOSTIC_REQUEST_JOIN_INVALID"):
        ledger.persist_output(unit.unit_id, output)

    assert not (tmp_path / "units").exists()
    (tmp_path / "reopen.json").write_text(json.dumps({"binding": binding.model_dump(mode="json"),
        "units": [asdict(unit)]}))
    child = _reopen(tmp_path)
    assert child.returncode == 0, child.stderr
    status = json.loads(child.stdout)
    assert status["pid"] != os.getpid()
    assert status["completed"] == 0 and status["in_flight"] == 1
    assert status["known_cost_usd_micros"] == 30000 + (amount or 0)
    assert status["unknown_cost_requests"] == int(amount is None)
    with pytest.raises(LiveDiagnosticError):
        ledger.claim(unit.unit_id)
    with pytest.raises(LiveDiagnosticError):
        ledger.claim_request(unit.unit_id, intent)

    with connect(ledger.path) as connection:
        event_hash = connection.execute("SELECT event_hash FROM events WHERE kind='INTENT'").fetchone()[0]
    record = LiveDiagnosticEvidence(binding_sha256=binding.sha256(), intent_event_hash=event_hash,
        unit_id=unit.unit_id, sequence=unit.sequence, output=output)
    raw = record.model_dump_json().encode()
    digest = hashlib.sha256(raw).hexdigest()
    evidence_path = ledger._evidence_path(unit.unit_id)
    evidence_path.parent.mkdir()
    evidence_path.write_bytes(raw)
    with pytest.raises(LiveDiagnosticError, match="DIAGNOSTIC_REQUEST_JOIN_INVALID"):
        ledger.reconcile(unit.unit_id, digest)
    with connect(ledger.path) as connection:
        append_event(connection, unit.unit_id, "COMPLETED", evidence_sha256=digest)
    child = _reopen(tmp_path)
    assert child.returncode != 0
    assert "DIAGNOSTIC_REQUEST_JOIN_INVALID" in child.stderr


@pytest.mark.parametrize("operation", ["count", "prefix", "setup"])
def test_completion_accepts_known_completed_trailing_operation(
    tmp_path: Path, operation: Literal["count", "prefix", "setup"],
) -> None:
    binding, units, outputs = _trailing_bundle(tmp_path, operation)
    unit, output = units[0], outputs[0]
    ledger = LiveDiagnosticLedger.create(binding, units)
    ledger.claim(unit.unit_id)
    record_calls(ledger, unit, output, binding)
    intent = DiagnosticRequestIntent(request_id="tail-operation",
        attempt_id=hashlib.sha256((binding.sha256() + "tail-operation").encode()).hexdigest(),
        request_sha256="a" * 64, reservation_id="qa-only-tail")
    ledger.claim_request(unit.unit_id, intent)
    ledger.record_request(unit.unit_id, DiagnosticRequestReceipt(intent=intent,
        receipt_sha256="b" * 64, realized_cost_usd_micros=5000, outcome="completed"))

    ledger.reconcile(unit.unit_id, ledger.persist_output(unit.unit_id, output))

    (tmp_path / "reopen.json").write_text(json.dumps({"binding": binding.model_dump(mode="json"),
        "units": [asdict(unit)]}))
    child = _reopen(tmp_path)
    assert child.returncode == 0, child.stderr
    status = json.loads(child.stdout)
    assert status["completed"] == 1 and status["known_cost_usd_micros"] == 35000
    assert status["unknown_cost_requests"] == 0 and status["unresolved_requests"] == 0
