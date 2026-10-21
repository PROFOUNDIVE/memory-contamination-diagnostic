from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, Literal, assert_never

import pytest

from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_main_live_dispatch import persist_reconciliation_evidence
from memcontam.readiness.phase13_main_runner_models import InFlightEvidence, MainRunError
from memcontam.readiness.phase13_main_runner_store import canonical, connect
from memcontam.readiness.phase13_production_observability import terminal_provider_evidence

from . import test_phase13_snapshot_bot
from .phase13_snapshot_durable_fixtures import native_snapshot
from .test_phase13_snapshot_durable import _bound, _ledger, _reopen

capacity_fixture = test_phase13_snapshot_bot.capacity_fixture
CONTRADICTIONS: Final = (
    ("provider_cost_usd", 123.0),
    ("provider_cost_usd", 0.0),
    ("authoritative_provider_cost_usd", 123.0),
    ("authoritative_provider_cost_usd", 0.0),
    ("derived_cost_usd", 123.0),
    ("derived_cost_usd", 0.0),
    ("provider_cost_source", "AUTHORITATIVE_PROVIDER"),
    ("provider_cost_source", "DERIVED_FROM_PROVIDER_USAGE"),
    ("provider_cost_source", ""),
    ("provider_usage", {"input_tokens": 1}),
    ("provider_usage", {}),
    ("token_usage", {"input_tokens": 0}),
    ("nested_cost", 123.0),
    ("nested_cost", 0.0),
    ("nested_usage", {"input_tokens": 1}),
    ("nested_usage", {}),
    ("nested_tokens", {"input_tokens": 0}),
)


@pytest.mark.parametrize("field,value", CONTRADICTIONS)
@pytest.mark.parametrize("route", ["writer", "reconcile", "reopen", "closed_reopen"])
def test_partial_rejects_cost_contradiction_when_outer_hash_recomputed(
    tmp_path: Path, field: str, value: float | str | dict[str, int],
    route: Literal["writer", "reconcile", "reopen", "closed_reopen"],
) -> None:
    unit, output = _bound(native_snapshot(tmp_path, "bot_style", "bot_instantiate_solve"))
    ledger = _ledger(tmp_path, unit)
    context = ledger.in_flight_context(unit.unit_id)
    receipt = persist_reconciliation_evidence(tmp_path, "TERMINAL_FAILURE", context,
        failure_code="provider_call_failed", diagnostic_unit=unit, diagnostic_output=output)
    if route == "closed_reopen":
        ledger.reconcile(unit.unit_id, receipt)
    path = tmp_path / "reconciliation" / f"{context.intent_event_hash}.json"
    record = json.loads(path.read_bytes())
    raw = output.model_dump(mode="json") if route == "writer" else record["diagnostic_partial"]
    row = raw["evidence"]["runtime_evidence"]["production_observability_archive"]["records"][0]
    nested_field = {"nested_cost": "cost_usd", "nested_usage": "usage", "nested_tokens": "token_usage"}.get(field)
    if nested_field is not None:
        row["terminal_provider_evidence"][nested_field] = value
    else:
        for call in (raw["provider_calls"][-1], row["method_calls"][-1], row["terminal_method_call"]):
            call[field] = value
        row["terminal_provider_evidence"] = terminal_provider_evidence(
            MethodCall.model_validate_json(json.dumps(row["terminal_method_call"])),
            "provider_call_failed").model_dump(mode="json")

    match route:
        case "writer":
            path.unlink()
            with pytest.raises(ValueError):
                persist_reconciliation_evidence(tmp_path, "TERMINAL_FAILURE", context,
                    failure_code="provider_call_failed", diagnostic_unit=unit,
                    diagnostic_output=output.model_validate_json(json.dumps(raw)))
            assert not path.exists()
        case "reconcile":
            path.write_text(json.dumps(record))
            resealed = InFlightEvidence.terminal_failure(context,
                hashlib.sha256(path.read_bytes()).hexdigest(), "provider_call_failed", 0)
            with pytest.raises(MainRunError, match="MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID"):
                ledger.reconcile(unit.unit_id, resealed)
        case "reopen":
            path.write_text(json.dumps(record))
            child = _reopen(tmp_path, reconcile=True)
            assert child.returncode != 0, child.stdout
            assert ("MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID" in child.stderr
                    or "TERMINAL_PROVIDER_EVIDENCE_MISMATCH" in child.stderr)
        case "closed_reopen":
            path.write_text(json.dumps(record))
            with connect(tmp_path / "ledger.sqlite3") as connection:
                event = list(connection.execute("SELECT * FROM events ORDER BY event_sequence DESC LIMIT 1").fetchone())
                event[3] = hashlib.sha256(path.read_bytes()).hexdigest()
                event[7] = hashlib.sha256(canonical(event[:7])).hexdigest()
                connection.execute("UPDATE events SET evidence_sha256 = ?, event_hash = ? WHERE event_sequence = ?",
                                   (event[3], event[7], event[0]))
            child = _reopen(tmp_path)
            assert child.returncode != 0, child.stdout
            assert "MAIN_RUN_RECONCILIATION_EVIDENCE_INVALID" in child.stderr
        case unreachable:
            assert_never(unreachable)
    if route != "closed_reopen":
        assert ledger.status().in_flight_count == 1
    assert receipt.realized_cost_krw == 0


@pytest.mark.parametrize("baseline,stage", [
    ("bot_style", "bot_instantiate_solve"), ("dc_rs", "dc_rs_generate"),
])
def test_no_response_partial_preserves_absent_cost_when_durably_reopened(
    tmp_path: Path, baseline: str, stage: str,
) -> None:
    unit, output = _bound(native_snapshot(tmp_path, baseline, stage))
    ledger = _ledger(tmp_path, unit)
    receipt = persist_reconciliation_evidence(tmp_path, "TERMINAL_FAILURE",
        ledger.in_flight_context(unit.unit_id), failure_code="provider_call_failed",
        diagnostic_unit=unit, diagnostic_output=output)
    ledger.reconcile(unit.unit_id, receipt)

    child = _reopen(tmp_path)

    assert child.returncode == 0, child.stderr
    report = json.loads(child.stdout)
    row = report["archive"]["records"][0]
    terminal = row["terminal_method_call"]
    for field in ("provider_cost_usd", "authoritative_provider_cost_usd", "derived_cost_usd",
                  "provider_cost_source", "provider_usage"):
        assert terminal[field] is None
    assert terminal["token_usage"] == {}
    for field in ("cost_usd", "usage", "token_usage"):
        assert row["terminal_provider_evidence"][field] is None
    assert report["status"]["realized_cost_krw"] == 0
    assert report["status"]["terminal_technical_missing_count"] == 1
