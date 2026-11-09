from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def budget_api():
    name = "memcontam.readiness.phase13_shared_provider_budget"
    assert importlib.util.find_spec(name) is not None, "durable shared USD budget is not implemented"
    return importlib.import_module(name)


def pin(path: Path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path: Path, value):
    path.write_text(json.dumps(value, indent=2))
    return pin(path)


def signed(value):
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {**value, "signature": key.sign(raw).hex()}


def fixture(root: Path, *, complete=True, prior=5_000_000, hold=2_000_000, allowance=38_000_000):
    root.mkdir(parents=True, exist_ok=True)
    evidence = save(root / "qa-evidence.json", {"provenance": "QA_ONLY_NOT_A_PROVIDER_GRANT"})
    reconciliation = {
        "schema_version": "phase13_budget_reconciliation_v1", "currency": "USD",
        "account_id": "qa-account", "accounting_root": str(root.resolve()),
        "parent_approval": evidence, "execution_approval": evidence,
        "complete": complete, "remaining_allowance_usd_micros": allowance if complete else None,
        "lanes": [{"lane_id": lane, "complete": complete, "funding": "aggregate",
                   "known_spend_usd_micros": prior if lane == "D" else 0,
                   "unknown_hold_usd_micros": hold if lane == "D" else 0,
                   "evidence": [evidence], "reason": "Synthetic accounting fixture"}
                  for lane in ("D", "M")],
        "operations": [{"lane_id": "D", "run_id": "old-run", "request_id": "old-request",
                        "attempt_id": "old-attempt", "known_usd_micros": prior,
                        "hold_usd_micros": hold, "state": "ambiguous" if hold else "settled", "evidence": evidence}],
        "blockers": [] if complete else ["UNRESOLVED_FUNDING"],
    }
    price = {"price_id": "qa-generation", "operation": "generation", "endpoint": "qa://generation",
             "model": "qa-model", "service_tier": "qa-tier", "currency": "USD",
             "units": "USD_micros_per_bounded_operation", "upper_bound_usd_micros": 25_000_000,
             "request_sha256": "b" * 64,
             "as_of": "2026-10-22", "valid_until": "2099-12-31", "evidence": evidence,
             "unknown_billing": False}
    pricing = {"schema_version": "phase13_budget_pricing_v1", "account_id": "qa-account",
               "entries": [price, {**price, "price_id": "qa-count", "operation": "count",
                                   "endpoint": "qa://count", "unknown_billing": True}],
               "blockers": []}
    binding = {"schema_version": "phase13_shared_provider_budget_v1", "currency": "USD",
               "cap_usd_micros": 45_000_000, "account_id": "qa-account", "run_id": "qa-run",
               "allowed_runs": ["qa-run"],
               "ledger_path": str(root.resolve() / "aggregate.sqlite3"), "lanes": ["D", "M"],
               "parent_approval": evidence, "execution_approval": evidence,
               "reconciliation": save(root / "reconciliation.json", reconciliation),
               "pricing": save(root / "pricing.json", pricing), "source_files": [evidence],
               "proof_authority": "qa-accounting-owner",
               "proof_public_key": Ed25519PrivateKey.from_private_bytes(bytes(range(32))).public_key().public_bytes(
                   Encoding.Raw, PublicFormat.Raw).hex()}
    save(root / "binding.json", binding)
    return root / "binding.json"


def operation(*, request="request-1", lane="D", kind="generation", bound=25_000_000):
    return {"lane_id": lane, "run_id": "qa-run", "execution_binding_sha256": "e" * 64,
            "unit_id": "a" * 64, "request_id": request,
            "attempt_id": hashlib.sha256(request.encode()).hexdigest(), "request_sha256": "b" * 64,
            "operation": kind, "endpoint": "qa://" + kind, "model": "qa-model",
            "service_tier": "qa-tier", "price_id": "qa-" + kind,
            "bound_usd_micros": bound, "unknown_count_decision": None}


def proof(root: Path, budget, intent, *, kind="charge", amount: int | None = 3_000_000):
    evidence = pin(root / "qa-evidence.json")
    return save(root / ("proof-" + intent.reservation_id + ".json"), signed({
        "schema_version": "phase13_budget_proof_v1", "budget_binding_sha256": budget.binding.sha256(),
        "account_id": "qa-account", "issuer": "qa-accounting-owner", "intent": intent.model_dump(mode="json"),
        "kind": kind, "realized_usd_micros": amount, "provider_request_id": None if kind == "no_request" else "provider-1",
        "owner_quiescent": True, "evidence": [evidence],
    }))


if __name__ == "__main__":
    import os
    api = budget_api()
    budget = api.SharedProviderBudget.open(Path(os.environ["BUDGET_BINDING"]))
    print("READY", flush=True)
    input()
    try:
        intent = budget.reserve(api.BudgetOperation.model_validate_json(os.environ["BUDGET_OPERATION"]))
        print(intent.model_dump_json(), flush=True)
        if os.environ.get("BUDGET_CRASH") == "after_intent":
            budget.mark_intent(intent)
            os._exit(73)
        if os.environ.get("BUDGET_CRASH") == "1":
            os._exit(73)
    except ValueError as error:
        print(str(error), flush=True)
        raise SystemExit(3)
