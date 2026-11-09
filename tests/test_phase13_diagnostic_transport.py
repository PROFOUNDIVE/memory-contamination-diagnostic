from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from .phase13_budget_fixtures import pin, save
from .test_phase13_live_diagnostic_runtime import fixture


def test_live_factory_requires_trusted_admission_before_client_construction(tmp_path: Path, monkeypatch) -> None:
    name = "memcontam.readiness.phase13_diagnostic_transport"
    assert importlib.util.find_spec(name) is not None, "guarded production factory missing"
    module = importlib.import_module(name)
    api, binding, registration_path, contract = fixture(tmp_path)
    contract["qualification"] = "LIVE_REQUIRES_ADMISSION"
    contract["rate_card"] = save(tmp_path / "qa-rates.json", {"input_micros_per_token": "1", "output_micros_per_token": "1", "count_micros": 5})
    contract["embedding_metadata_sha256"] = "f" * 64
    for index, stage in enumerate(("bot_instantiate_solve", "bot_thought_distill"), start=2):
        contract["observations"][0]["requests"].extend([
            {"request_id": f"qa-count-{index}", "operation": "count", "stage": stage},
            {"request_id": f"qa-generation-{index}", "operation": "generation", "stage": stage,
             "native_call_id": f"qa-native:call:{index}"}])
    contract_pin = save(tmp_path / "runtime-contract.json", contract)
    registration = json.loads(registration_path.read_bytes())
    registration["runtime_contract"] = contract_pin
    registration_pin = save(registration_path, registration)
    from memcontam.experiment.phase13_live_diagnostic_runtime import runtime_source_paths
    from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_source_paths
    from memcontam.readiness.phase13_durable_call_join import DiagnosticFile
    from memcontam.readiness.phase13_live_diagnostic_models import DiagnosticObservation
    sources = {row.path: row for row in binding.source_files}
    for path in runtime_source_paths() | snapshot_source_paths("bot_style") | {Path(__file__).resolve()}:
        sources[str(path)] = DiagnosticFile(**pin(path))
    sources[contract_pin["path"]] = DiagnosticFile(**contract_pin)
    binding = binding.model_copy(update={"registration_sha256": registration_pin["sha256"], "source_files": tuple(sources.values()),
        "observations": tuple(DiagnosticObservation.model_validate_json(json.dumps(row)) for row in contract["observations"])})
    context = api.DiagnosticExecutionContext.load(binding, registration_path)
    constructed = []
    admissions = []
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: constructed.append(kwargs))
    def denied(context, child_execution_id, _api_key):
        admissions.append(child_execution_id)
        raise ValueError("QA_OWNER_ADMISSION_DENIED")
    with pytest.raises(ValueError):
        module.open_live_transport(context, None)
    launch = module.DiagnosticLaunch("qa-child", "qa-not-a-secret", denied)
    with pytest.raises(ValueError):
        module.open_live_transport(context, launch)
    assert constructed == []
    assert admissions == ["qa-child"]


@pytest.mark.parametrize("mutation", ["bad_count", "bad_usage", "bad_response_id", "bad_model"])
def test_invalid_wire_reply_becomes_truthful_hold(tmp_path: Path, monkeypatch, mutation: str) -> None:
    from memcontam.experiment import phase13_ordinary_runtime

    from .phase13_diagnostic_runtime_fixtures import native_fixture
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    module, run, resources, transport = native_fixture(tmp_path)
    original = transport.send
    def changed(operation, raw, before_egress):
        from dataclasses import replace
        reply = original(operation, raw, before_egress)
        body = json.loads(reply.body)
        if mutation == "bad_count" and operation.operation == "count":
            body["input_tokens"] = "not-a-count"
        elif mutation != "bad_count" and operation.operation == "generation":
            if mutation == "bad_usage":
                body["usage"]["input_tokens"] = 11
            elif mutation == "bad_response_id":
                body["id"] = ""
            else:
                body["model"] = "foreign"
        return replace(reply, body=json.dumps(body).encode())
    monkeypatch.setattr(transport, "send", changed)
    result = module.execute_diagnostic(run)
    assert result.disposition == "HOLD"
    assert result.generation_attempts == (0 if mutation == "bad_count" else 1)
    assert resources.ledger.status()["completed"] == 0
    assert not list((tmp_path / "units").glob("*.json"))


def test_pre_egress_generation_refusal_is_not_a_provider_attempt(tmp_path: Path, monkeypatch) -> None:
    from memcontam.experiment import phase13_ordinary_runtime
    from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticError

    from .phase13_diagnostic_runtime_fixtures import native_fixture
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    module, run, _, transport = native_fixture(tmp_path)
    original = transport.send
    def denied(operation, raw, *args):
        if operation.operation == "generation":
            raise LiveDiagnosticError("QA_REJECTED_BEFORE_TRANSPORT_EGRESS")
        return original(operation, raw, *args)
    monkeypatch.setattr(transport, "send", denied)
    result = module.execute_diagnostic(run)
    assert result.disposition == "HOLD" and result.generation_attempts == 0
    assert transport.sent == ["count"]


def test_budget_implementation_source_drift_refuses_before_count(tmp_path: Path, monkeypatch) -> None:
    from memcontam.experiment import phase13_ordinary_runtime
    from memcontam.readiness import phase13_shared_provider_budget

    from .phase13_diagnostic_runtime_fixtures import native_fixture
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    module, run, _, transport = native_fixture(tmp_path)
    foreign = tmp_path / "foreign-budget.py"
    foreign.write_text("different QA implementation")
    monkeypatch.setattr(phase13_shared_provider_budget, "__file__", str(foreign))
    with pytest.raises(ValueError, match="SOURCE"):
        module.execute_diagnostic(run)
    assert transport.sent == []


def test_live_run_builder_needs_no_main_or_test_client_escape(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from memcontam.experiment import phase13_ordinary_runtime
    from memcontam.readiness import phase13_diagnostic_transport

    from .phase13_diagnostic_runtime_fixtures import native_fixture
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    module, template, resources, transport = native_fixture(tmp_path, live_candidate=True)
    events = []
    def verified(context, child, api_key):
        assert context == resources.context and child == resources.scope.child_execution_id
        assert api_key == "qa-not-a-secret"
        events.append("admit")
    def constructor(**kwargs):
        events.append("construct")
        assert kwargs["timeout"] == 180 and kwargs["follow_redirects"] is False and kwargs["trust_env"] is False
        assert kwargs["headers"]["Authorization"] == "Bearer qa-not-a-secret"
        return SimpleNamespace(close=lambda: None)
    def build(admitted):
        events.append("build")
        return replace(template, allow_test_client=False, client=admitted.execution_client, diagnostic_context=admitted)
    monkeypatch.setattr(phase13_diagnostic_transport.httpx, "Client", constructor)
    monkeypatch.setattr(phase13_diagnostic_transport.httpx, "HTTPTransport", lambda **kwargs: ("QA_ONLY", kwargs))
    launch = phase13_diagnostic_transport.DiagnosticLaunch(resources.scope.child_execution_id, "qa-not-a-secret", verified)
    run = module.admit_live(build, resources, launch)
    assert run.allow_test_client is False and run.validated_resources is None and run.production_identity is None
    assert events[0] == "build" and events[-1] == "construct" and set(events[1:-1]) == {"admit"}
    assert resources.budget.status().reservations == 0
    assert transport.sent == []
