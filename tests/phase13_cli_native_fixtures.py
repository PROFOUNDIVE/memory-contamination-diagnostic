from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import TypeAdapter

import memcontam.readiness.phase13_diagnostic_cli as cli
import memcontam.readiness.phase13_diagnostic_trusted_admission as trust
from memcontam.experiment import phase13_ordinary_runtime
from memcontam.experiment.phase13_live_diagnostic_runtime import runtime_source_paths
from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_dispatch
from memcontam.readiness.phase13_diagnostic_cli_models import (
    RegisteredDiagnostic,
    RegistrationSubject,
)
from memcontam.readiness.phase13_live_diagnostic import LiveDiagnosticLedger
from memcontam.readiness.phase13_live_diagnostic_models import (
    LiveDiagnosticBinding,
    LiveDiagnosticScope,
)
from memcontam.readiness.phase13_shared_provider_budget import SharedProviderBudget
from memcontam.readiness.phase13_v3_request import (
    MessageV3,
    RequestKeyV3,
    RequestMaterialV3,
    Stage,
    compile_request_bytes,
)

from .phase13_budget_fixtures import pin, save
from .phase13_cli_cases import native_case
from .phase13_cli_fixtures import signed_evidence
from .phase13_recovery_offline import install_offline_denial
from .test_phase13_live_diagnostic_runtime import fixture as contract_fixture


def registered_fixture(root: Path, monkeypatch, failure=None):
    install_offline_denial(monkeypatch)
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    root.mkdir(parents=True, exist_ok=True)
    prep = root / "prep"
    prep.mkdir()
    _, _, runtime_path, contract = contract_fixture(prep)
    keys = {role: Ed25519PrivateKey.generate() for role in ("science", "launch", "credential")}
    _, policy = signed_evidence(trust, root, keys=keys)
    policy_path = root / "synthetic-trust.json"
    save(policy_path, policy.model_dump(mode="json"))
    monkeypatch.setattr(trust, "TRUST_POLICY_PATH", policy_path)
    monkeypatch.setattr(trust, "load_policy", lambda: policy)
    snapshots = []
    for index in range(16):
        folder = root / f"input-{index}"
        folder.mkdir()
        snapshots.append(native_case(folder, index))
    rate_pin = save(prep / "rates.json", {"input_micros_per_token": "1", "output_micros_per_token": "1", "count_micros": 5})
    observations, scopes, prices, planned = [], [], {}, []
    responses = {}
    sources = {str(path): pin(path) for path in runtime_source_paths() | cli.cli_source_paths() | {policy_path}}
    for index, snapshot in enumerate(snapshots):
        _, output = snapshot_dispatch(snapshot)
        scope = output.model_dump(mode="json")["evidence"]["runtime_evidence"]["diagnostic_scope"]
        sources.update({item["path"]: item for item in scope["source_files"]})
        scopes.append(scope)
        child = "qa-bot" if index < 8 else "qa-dc"
        observation = {"run_id": snapshot.run.run_id, "child_execution_id": child,
                       "input_sha256": scope["input_sha256"], "state_sha256": scope["state_sha256"], "requests": []}
        for call in output.provider_calls:
            wire = compile_request_bytes(RequestKeyV3(parent_id="a" * 64, stage=TypeAdapter(Stage).validate_python(call.stage), ordinal=0),
                RequestMaterialV3(messages=tuple(MessageV3.model_validate(row) for row in call.messages), native_state=scope["state_utf8"].encode()))
            body = json.loads(wire)
            count = json.dumps({key: body[key] for key in ("model", "input", "tools", "reasoning")}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            responses[hashlib.sha256(wire).hexdigest()] = call.raw_response
            for kind, raw, suffix in (("count", count, "/responses/input_tokens"), ("generation", wire, "/responses")):
                digest = hashlib.sha256(raw).hexdigest()
                request_id = f"{call.call_id}:{kind}"
                observation["requests"].append({"request_id": request_id, "operation": kind, "stage": call.stage,
                                                "native_call_id": call.call_id if kind == "generation" else None})
                prices[digest] = {"price_id": digest, "operation": kind, "endpoint": contract["endpoint"] + suffix,
                    "model": contract["model"], "service_tier": "default", "currency": "USD",
                    "units": "USD_micros_per_bounded_operation", "upper_bound_usd_micros": 10000,
                    "request_sha256": digest, "as_of": "2026-01-01", "valid_until": "2099-12-31", "evidence": rate_pin, "unknown_billing": False}
                planned.append((index, request_id, prices[digest]))
        observations.append(observation)
    budget_path = Path(contract["budget_binding"]["path"])
    budget = json.loads(budget_path.read_bytes())
    budget["allowed_runs"] = [row["run_id"] for row in observations]
    budget["pricing"] = save(budget_path.parent / "pricing.json", {"schema_version": "phase13_budget_pricing_v1",
        "account_id": "qa-account", "entries": list(prices.values()), "blockers": []})
    contract.update(budget_binding=save(budget_path, budget), observations=observations,
        qualification="LIVE_REQUIRES_ADMISSION", rate_card=rate_pin,
        embedding_metadata_sha256=hashlib.sha256(json.dumps(snapshots[0].run.embedding_provider.metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    contract_pin = save(prep / "runtime-contract.json", contract)
    runtime = json.loads(runtime_path.read_bytes())
    runtime["runtime_contract"] = contract_pin
    runtime_pin = save(runtime_path, runtime)
    sources.update({row["path"]: row for row in (contract_pin, runtime_pin)})
    parent = root / "run"
    binding = LiveDiagnosticBinding.model_validate_json(json.dumps({
        **{key: runtime[key] for key in ("run_id", "source_revision", "registration_id", "authorization_id", "authorization_sha256", "scientific_disposition")},
        "registration_sha256": runtime_pin["sha256"], "ledger_path": str(parent / "ledger.sqlite3"),
        "source_files": list(sources.values()), "observations": observations}))
    live_scopes = []
    for index, scope in enumerate(scopes):
        scope.pop("real_spend_cap_krw")
        scope.update(provenance="LIVE_DIAGNOSTIC", main_scope="NOT_MAIN", binding_sha256=binding.sha256(),
            child_execution_id=observations[index]["child_execution_id"], scientific_disposition="NOT_ADOPTED", source_files=list(sources.values()))
        live_scopes.append(LiveDiagnosticScope.model_validate_json(json.dumps(scope)))
    operations = []
    for index, request_id, price in planned:
        scope = live_scopes[index]
        operations.append({"lane_id": "D", "run_id": scope.run_id, "execution_binding_sha256": binding.sha256(),
            "unit_id": hashlib.sha256((scope.run_id + scope.sha256()).encode()).hexdigest(), "request_id": request_id,
            "attempt_id": hashlib.sha256((binding.sha256() + request_id).encode()).hexdigest(),
            **{key: price[key] for key in ("request_sha256", "operation", "endpoint", "model", "service_tier", "price_id")},
            "bound_usd_micros": price["upper_bound_usd_micros"], "unknown_count_decision": None})
    if failure == "unplanned_wire":
        alternative = next(price for price in prices.values() if price["operation"] == "count"
                           and price["request_sha256"] != operations[0]["request_sha256"])
        operations[0].update(request_sha256=alternative["request_sha256"], price_id=alternative["price_id"])
    subject = RegistrationSubject.model_validate_json(json.dumps({"schema_version": "phase13_diagnostic_cli_subject_v1",
        "registration_path": str(root / "registration.json"), "parent_root": str(parent), "runtime_registration": runtime_pin,
        "binding": binding.model_dump(mode="json"), "scopes": [scope.model_dump(mode="json") for scope in live_scopes],
        "children": [{"child_execution_id": child, "run_root": str(parent / child)} for child in ("qa-bot", "qa-dc")],
        "operations": operations, "embedding_cache": None}))
    evidence, _ = signed_evidence(trust, root, subject.sha256(), keys)
    candidate = root / "candidate.json"
    save(candidate, RegisteredDiagnostic(subject=subject, admission=evidence).model_dump(mode="json"))
    cli.register(candidate, root / "registration.json")
    SharedProviderBudget.create(budget_path)
    sent = []

    def handle(request):
        sent.append(request.url.path)
        assert failure != "unplanned_wire", "unexpected wire reached fake egress"
        if request.url.path.endswith("input_tokens"):
            if failure == "count_timeout":
                raise httpx.ReadTimeout("synthetic count ambiguity")
            return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 10}, headers={"x-request-id": f"qa-{len(sent)}"})
        response = {"id": f"qa-response-{len(sent)}", "object": "response", "created_at": 1, "model": "gpt-5.6-luna",
            "status": "incomplete" if failure == "incomplete" else "completed", "service_tier": "default",
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "output": [{"id": f"qa-message-{len(sent)}", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": responses[hashlib.sha256(request.content).hexdigest()], "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "incomplete_details": {"reason": "max_output_tokens"} if failure == "incomplete" else None}
        return httpx.Response(200, json=response, headers={"x-request-id": f"qa-{len(sent)}"})

    monkeypatch.setattr(httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(handle))
    if failure == "qa_client":
        original_build = cli.phase13_diagnostic_cli_native.build_native
        monkeypatch.setattr(cli.phase13_diagnostic_cli_native, "build_native",
            lambda admitted, embedding: replace(original_build(admitted, embedding), allow_test_client=True))
    if failure in {"cost_contradiction", "after_output"}:
        original_persist = LiveDiagnosticLedger.persist_output
        def changed_output(self, unit_id, output):
            if failure == "cost_contradiction":
                return original_persist(self, unit_id, output.model_copy(update={"realized_cost_usd_micros": 999999}))
            original_persist(self, unit_id, output)
            raise SystemExit("QA interruption after durable output")
        monkeypatch.setattr(LiveDiagnosticLedger, "persist_output", changed_output)
    if failure == "after_receipt":
        original_record = LiveDiagnosticLedger.record_request
        def interrupted(self, unit_id, receipt):
            original_record(self, unit_id, receipt)
            raise SystemExit("QA interruption after durable receipt")
        monkeypatch.setattr(LiveDiagnosticLedger, "record_request", interrupted)
    return root / "registration.json", parent, sent


def mutate_registration(path: Path, mutation: str) -> Path:
    raw = json.loads(path.read_bytes())
    if mutation in {"science", "launch"}:
        raw["admission"][mutation]["signature"] = "00" * 64
    elif mutation == "proposal128":
        raw["subject"]["scopes"] *= 8
    elif mutation == "price":
        raw["subject"]["operations"] = []
    elif mutation == "source":
        raw["subject"]["binding"]["source_files"][0]["sha256"] = "f" * 64
    elif mutation == "cost":
        raw["subject"]["operations"][0]["bound_usd_micros"] = 0
    path.write_text(json.dumps(raw))
    return path
