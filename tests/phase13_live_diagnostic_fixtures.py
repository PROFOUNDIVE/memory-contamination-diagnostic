from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

if __name__ == "__main__":
    __package__ = "tests"

from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_dispatch
from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_live_diagnostic_models import (
    DiagnosticObservation, DiagnosticRequest, DiagnosticRequestIntent,
    DiagnosticRequestReceipt, LiveDiagnosticBinding, LiveDiagnosticOutput, LiveDiagnosticScope,
)
from memcontam.readiness.phase13_production_observability import terminal_provider_evidence

from .phase13_snapshot_durable_fixtures import native_snapshot


def bundle(root: Path, baselines=("bot_style",), terminal_cost: str | None = None, *, count: bool = False):
    snapshots = []
    for baseline in baselines:
        child = root / baseline
        child.mkdir()
        failure = ("bot_instantiate_solve" if baseline == "bot_style" else "dc_rs_generate") if terminal_cost else None
        snapshots.append(snapshot_dispatch(native_snapshot(child, baseline, failure)))
    raws = [output.model_dump(mode="json") for _, output in snapshots]
    scopes = [raw["evidence"]["runtime_evidence"]["diagnostic_scope"] for raw in raws]
    observations = tuple(DiagnosticObservation(
        run_id=scope["run_id"], child_execution_id=baseline,
        input_sha256=scope["input_sha256"], state_sha256=scope["state_sha256"],
        requests=((DiagnosticRequest(request_id=scope["run_id"] + ":count", operation="count", stage="input_count"),)
            if count else ()) + tuple(DiagnosticRequest(request_id=call["call_id"], operation="generation",
            stage=call["stage"], native_call_id=call["call_id"]) for call in raw["provider_calls"]),
    ) for baseline, scope, raw in zip(baselines, scopes, raws, strict=True))
    source_map = {pin["path"]: pin for scope in scopes for pin in scope["source_files"]}
    readiness = Path(__file__).resolve().parents[1] / "src/memcontam/readiness"
    for name in ("phase13_live_diagnostic.py", "phase13_live_diagnostic_models.py", "phase13_live_diagnostic_evidence.py",
                 "phase13_main_runner_store.py", "phase13_main_live_dispatch.py", "phase13_main_live_evidence.py"):
        path = readiness / name
        source_map[str(path)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    binding = LiveDiagnosticBinding.model_validate_json(json.dumps({
        "run_id": "qa-task8-not-a-provider-grant", "ledger_path": str(root / "ledger.sqlite3"),
        "source_revision": "3c764d8ee30f9edc9240f20904eef501cc137541",
        "source_files": list(source_map.values()), "registration_id": "qa-registration",
        "registration_sha256": "1" * 64, "authorization_id": "qa-not-a-live-authorization",
        "authorization_sha256": "2" * 64, "scientific_disposition": "NOT_ADOPTED",
        "observations": [row.model_dump(mode="json") for row in observations],
    }))
    units, outputs = [], []
    for sequence, ((unit, _), raw, scope, baseline) in enumerate(zip(snapshots, raws, scopes, baselines, strict=True)):
        del scope["real_spend_cap_krw"]
        scope.update(provenance="LIVE_DIAGNOSTIC", main_scope="NOT_MAIN", binding_sha256=binding.sha256(),
                     child_execution_id=baseline, scientific_disposition="NOT_ADOPTED")
        live_scope = LiveDiagnosticScope.model_validate_json(json.dumps(scope))
        unit = replace(unit, sequence=sequence, diagnostic_scope_sha256=live_scope.sha256(),
                       unit_id=hashlib.sha256((live_scope.run_id + live_scope.sha256()).encode()).hexdigest())
        runtime = raw["evidence"]["runtime_evidence"]
        runtime["unit_id"] = unit.unit_id
        row = runtime["production_observability_archive"]["records"][0]
        for call in raw["provider_calls"]:
            if call["error_type"] is None or terminal_cost == "paid":
                call.update(provider_cost_usd=0.01, derived_cost_usd=0.01,
                            provider_cost_source="DERIVED_FROM_PROVIDER_USAGE",
                            provider_usage={"input_tokens": 1, "output_tokens": 1},
                            token_usage={"prompt_tokens": 1, "completion_tokens": 1})
        row["method_calls"] = raw["provider_calls"]
        if terminal_cost:
            row["terminal_method_call"] = raw["provider_calls"][-1]
            row["terminal_provider_evidence"] = terminal_provider_evidence(
                MethodCall.model_validate_json(json.dumps(row["terminal_method_call"])),
                row["terminal_failure_code"]).model_dump(mode="json")
        outputs.append(LiveDiagnosticOutput.model_validate_json(json.dumps({
            "evidence": raw["evidence"], "provider_calls": raw["provider_calls"],
            "realized_cost_usd_micros": None if terminal_cost == "unknown" else len(raw["provider_calls"]) * 10000,
            "disposition": "TERMINAL_FAILURE" if terminal_cost else "COMPLETED",
        })))
        units.append(unit)
    return binding, tuple(units), tuple(outputs)


def record_calls(ledger, unit, output, binding, *, pause_first: bool = False):
    for index, call in enumerate(output.provider_calls):
        assert call.call_id is not None
        request_hash = hashlib.sha256(json.dumps(call.messages, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        intent = DiagnosticRequestIntent(request_id=call.call_id,
            attempt_id=hashlib.sha256((binding.sha256() + call.call_id).encode()).hexdigest(),
            request_sha256=request_hash, reservation_id=f"qa-reservation:{call.call_id}")
        ledger.claim_request(unit.unit_id, intent)
        if pause_first and index == 0:
            print("REQUEST_CLAIMED", flush=True)
            input()
        receipt = DiagnosticRequestReceipt(intent=intent,
            receipt_sha256=hashlib.sha256(json.dumps(call.model_dump(mode="json"), sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            realized_cost_usd_micros=None if call.provider_cost_usd is None else 10000,
            outcome="failed" if call.error_type else "completed")
        ledger.record_request(unit.unit_id, receipt)


if __name__ == "__main__":
    import os
    from memcontam.readiness.phase13_live_diagnostic import LiveDiagnosticLedger
    from memcontam.readiness.phase13_main_production import ProductionObject

    root = Path(os.environ["TASK8_ROOT"])
    payload = json.loads((root / "reopen.json").read_bytes())
    binding = LiveDiagnosticBinding.model_validate_json(json.dumps(payload["binding"]))
    units = tuple(ProductionObject(**row) for row in payload["units"])
    ledger = LiveDiagnosticLedger.open(binding, units)
    if "TASK8_CHILD" in os.environ:
        index = int(os.environ["TASK8_CHILD"])
        output = LiveDiagnosticOutput.model_validate_json(json.dumps(payload["outputs"][index]))
        ledger.claim(units[index].unit_id)
        print("CLAIMED", flush=True)
        input()
        record_calls(ledger, units[index], output, binding, pause_first=True)
        digest = ledger.persist_output(units[index].unit_id, output)
        ledger.reconcile(units[index].unit_id, digest)
    elif os.environ.get("TASK8_RECONCILE") == "1":
        for unit, digest in zip(units, payload["digests"], strict=True):
            ledger.reconcile(unit.unit_id, digest)
    print(json.dumps({"pid": os.getpid(), **ledger.status()}))
