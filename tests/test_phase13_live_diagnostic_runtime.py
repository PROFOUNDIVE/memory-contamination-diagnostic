"""Provider-free contract-unit QA, not provider or scientific qualification."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticBinding
from memcontam.readiness.phase13_v3_request import (
    MessageV3,
    RequestMaterialV3,
    compile_request_bytes,
)

from .phase13_budget_fixtures import fixture as budget_fixture
from .phase13_budget_fixtures import pin, save


def api():
    name = "memcontam.readiness.phase13_diagnostic_runtime_contract"
    assert importlib.util.find_spec(name) is not None, "registration-bound Diagnostic contract missing"
    return importlib.import_module(name)


def fixture(root: Path):
    module = api()
    budget_path = budget_fixture(root / "budget")
    resource = save(root / "qa-resource.json", {"purpose": "QA_ONLY_NOT_QUALIFIED"})
    observation = {"run_id": "qa-run", "child_execution_id": "qa-child",
                   "input_sha256": "a" * 64, "state_sha256": hashlib.sha256(b"qa native state").hexdigest(),
                   "requests": [
                       {"request_id": "qa-count", "operation": "count", "stage": "bot_problem_distill"},
                       {"request_id": "qa-generation", "operation": "generation",
                        "stage": "bot_problem_distill", "native_call_id": "qa-native:call:1"},
                   ]}
    contract = {
        "schema_version": "phase13_diagnostic_runtime_contract_v1",
        "provenance": "LIVE_DIAGNOSTIC", "main_scope": "NOT_MAIN",
        "qualification": "QA_ONLY_NOT_PROVIDER_QUALIFIED",
        "model": "gpt-5.6-luna", "service_tier": "default",
        "endpoint": "https://api.openai.com/v1", "timeout_seconds": 180,
        "attempt_limit": 1, "retries": 0,
        "budget_binding": pin(budget_path),
        "resource_files": [resource],
        "authority": {
            "execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V4",
            "execution_envelope_sha256": "1" * 64,
            "transport_contract_id": "CORE_TRANSPORT_ATTEMPT_CONTRACT_V3",
            "transport_contract_sha256": "2" * 64,
            "terminal_failure_contract_id": "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V2",
            "terminal_failure_contract_sha256": "3" * 64,
            "evidence": resource,
        },
        "observations": [observation],
    }
    authority_pin = save(root / "qa-authority.json", {
        key: value for key, value in contract["authority"].items() if key != "evidence"})
    contract["authority"]["evidence"] = authority_pin
    contract_pin = save(root / "runtime-contract.json", contract)
    registration = {"schema_version": "phase13_diagnostic_runtime_registration_v1",
                    "registration_id": "qa-registration", "run_id": "qa-parent",
                    "source_revision": "c" * 40, "authorization_id": "qa-not-authorized",
                    "authorization_sha256": "d" * 64,
                    "scientific_disposition": "NOT_ADOPTED", "runtime_contract": contract_pin}
    registration_pin = save(root / "registration.json", registration)
    assert module.__file__ is not None
    source = Path(module.__file__).resolve()
    compiler_module = importlib.import_module(compile_request_bytes.__module__)
    assert compiler_module.__file__ is not None
    compiler = Path(compiler_module.__file__).resolve()
    binding = LiveDiagnosticBinding.model_validate_json(json.dumps({
        "run_id": registration["run_id"], "ledger_path": str(root / "ledger.sqlite3"),
        "source_revision": registration["source_revision"],
        "source_files": [pin(source), pin(compiler), contract_pin, resource],
        "registration_id": registration["registration_id"],
        "registration_sha256": registration_pin["sha256"],
        "authorization_id": registration["authorization_id"],
        "authorization_sha256": registration["authorization_sha256"],
        "scientific_disposition": "NOT_ADOPTED", "observations": [observation],
    }))
    return module, binding, Path(registration_pin["path"]), contract


def material():
    return RequestMaterialV3(messages=(MessageV3(role="user", content="qa input"),),
                             native_state=b"qa native state")


def test_registered_contract_is_immutable_and_does_not_grant_transport(tmp_path: Path) -> None:
    module, binding, registration, _ = fixture(tmp_path)
    context = module.DiagnosticExecutionContext.load(binding, registration)
    detached = context.contract.model_dump(mode="json")
    detached["retries"] = 5

    assert context.contract.retries == 0
    with pytest.raises((AttributeError, ValueError)):
        context.contract.retries = 5
    with pytest.raises(ValueError, match="LIVE_ADMISSION_UNESTABLISHED"):
        context.require_live_admission()
    assert not (tmp_path / "ledger.sqlite3").exists()
    assert not (tmp_path / "budget/aggregate.sqlite3").exists()


@pytest.mark.parametrize("field,value", [
    ("retries", 1), ("attempt_limit", 2), ("timeout_seconds", 30),
    ("main_scope", "MAIN"), ("model", "replay"), ("service_tier", "priority"),
    ("qualification", "PROVIDER_QUALIFIED"),
])
def test_contract_refuses_unregistered_execution_policy(tmp_path: Path, field: str, value) -> None:
    module, _, _, raw = fixture(tmp_path)
    raw[field] = value
    with pytest.raises(ValueError):
        module.DiagnosticRuntimeContract.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("field", ["run_id", "registration_id", "source_revision", "authorization_id",
                                  "authorization_sha256", "registration_sha256", "observations"])
def test_context_refuses_foreign_binding(tmp_path: Path, field: str) -> None:
    module, binding, registration, _ = fixture(tmp_path)
    changed = binding.model_copy(update={field: () if field == "observations" else "f" * 64})
    with pytest.raises(ValueError):
        module.DiagnosticExecutionContext.load(changed, registration)


@pytest.mark.parametrize("target", ["runtime-contract.json", "qa-resource.json", "budget/binding.json"])
def test_source_resource_drift_refuses_before_count(tmp_path: Path, target: str) -> None:
    module, binding, registration, _ = fixture(tmp_path)
    context = module.DiagnosticExecutionContext.load(binding, registration)
    path = tmp_path / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        context.verify()


def test_legacy_v2_validator_still_rejects_v4() -> None:
    from memcontam.readiness.phase13_readiness0_evidence_models import ProviderAuthorityContract

    with pytest.raises(ValueError):
        ProviderAuthorityContract.model_validate({
            "maximum_input_tokens": 1177, "maximum_output_tokens": 384,
            "execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V4",
            "execution_envelope_sha256": "1" * 64,
            "failure_contract_id": "CORE_TRANSPORT_ATTEMPT_CONTRACT_V3",
            "failure_contract_sha256": "2" * 64,
            "terminal_failure_contract_id": "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V2",
            "terminal_failure_contract_sha256": "3" * 64, "rate_card_sha256": "4" * 64,
        })


def test_wire_request_and_count_projection_are_distinct(tmp_path: Path) -> None:
    module, binding, registration, _ = fixture(tmp_path)
    context = module.DiagnosticExecutionContext.load(binding, registration)
    compiled = context.compile("qa-generation", material())

    assert compiled.wire_sha256 != compiled.messages_sha256
    assert compiled.count_sha256 != compiled.wire_sha256
    body = json.loads(compiled.wire_bytes)
    assert body["max_output_tokens"] == 384
    assert json.loads(compiled.count_bytes) == {
        name: body[name] for name in ("model", "input", "tools", "reasoning")}
    assert compiled.attempt_id == hashlib.sha256((binding.sha256() + "qa-generation").encode()).hexdigest()
    assert compiled.native_call_id == "qa-native:call:1"
    assert compiled.native_state_sha256 == hashlib.sha256(material().native_state).hexdigest()


@pytest.mark.parametrize("request_id", ["qa-count", "foreign", ""])
def test_unregistered_or_non_generation_compile_refuses(tmp_path: Path, request_id: str) -> None:
    module, binding, registration, _ = fixture(tmp_path)
    context = module.DiagnosticExecutionContext.load(binding, registration)
    with pytest.raises(ValueError, match="GENERATION_IDENTITY"):
        context.compile(request_id, material())


def test_unpriced_response_dependent_request_refuses(tmp_path: Path) -> None:
    module, binding, registration, _ = fixture(tmp_path)
    context = module.DiagnosticExecutionContext.load(binding, registration)
    compiled = context.compile("qa-generation", material())

    with pytest.raises(ValueError, match="EXACT_PRICE"):
        context.price_operations(compiled, "e" * 64)
    assert not (tmp_path / "budget/aggregate.sqlite3").exists()


def test_exact_quotes_bind_count_and_generation_without_reserving(tmp_path: Path) -> None:
    module, binding, registration, contract = fixture(tmp_path)
    preliminary = module.DiagnosticExecutionContext.load(binding, registration).compile("qa-generation", material())
    budget_path = tmp_path / "budget/binding.json"
    budget = json.loads(budget_path.read_bytes())
    pricing_path = tmp_path / "budget/pricing.json"
    pricing = json.loads(pricing_path.read_bytes())
    for price, kind, digest, endpoint in zip(pricing["entries"], ("generation", "count"),
            (preliminary.wire_sha256, preliminary.count_sha256), ("/responses", "/responses/input_tokens"), strict=True):
        price.update(operation=kind, request_sha256=digest, endpoint=contract["endpoint"] + endpoint,
                     model=contract["model"], service_tier="default", unknown_billing=False,
                     upper_bound_usd_micros=10000)
    budget["pricing"] = save(pricing_path, pricing)
    contract["budget_binding"] = save(budget_path, budget)
    contract_pin = save(tmp_path / "runtime-contract.json", contract)
    registered = json.loads(registration.read_bytes())
    registered["runtime_contract"] = contract_pin
    registration_pin = save(registration, registered)
    binding = binding.model_copy(update={"registration_sha256": registration_pin["sha256"],
        "source_files": tuple(row.model_copy(update={"sha256": contract_pin["sha256"]})
                              if row.path == contract_pin["path"] else row for row in binding.source_files)})
    context = module.DiagnosticExecutionContext.load(binding, registration)
    compiled = context.compile("qa-generation", material())

    counted, generation = context.price_operations(compiled, "e" * 64)

    assert counted.operation == "count" and generation.operation == "generation"
    assert counted.request_sha256 == compiled.count_sha256
    assert generation.request_sha256 == compiled.wire_sha256
    assert counted.attempt_id != generation.attempt_id
    assert counted.execution_binding_sha256 == generation.execution_binding_sha256 == binding.sha256()
    assert counted.unit_id == generation.unit_id == "e" * 64
    assert not (tmp_path / "budget/aggregate.sqlite3").exists()


def test_pinned_authority_witness_must_match_the_registered_projection(tmp_path: Path) -> None:
    module, binding, registration, contract = fixture(tmp_path)
    witness = tmp_path / "qa-authority.json"
    raw = json.loads(witness.read_bytes())
    raw["execution_envelope_sha256"] = "9" * 64
    contract["authority"]["evidence"] = save(witness, raw)
    contract_pin = save(tmp_path / "runtime-contract.json", contract)
    registered = json.loads(registration.read_bytes())
    registered["runtime_contract"] = contract_pin
    registration_pin = save(registration, registered)
    binding = binding.model_copy(update={"registration_sha256": registration_pin["sha256"],
        "source_files": tuple(row.model_copy(update={"sha256": contract_pin["sha256"]})
                              if row.path == contract_pin["path"] else row for row in binding.source_files)})

    with pytest.raises(ValueError, match="AUTHORITY_PROJECTION"):
        module.DiagnosticExecutionContext.load(binding, registration)


@pytest.mark.parametrize("field,value", [("temperature", 1.0), ("top_p", 0.5),
                                        ("native_state", b"foreign checkpoint")])
def test_changed_request_material_refuses(tmp_path: Path, field: str, value) -> None:
    module, binding, registration, _ = fixture(tmp_path)
    context = module.DiagnosticExecutionContext.load(binding, registration)
    with pytest.raises(ValueError, match="MATERIAL_MISMATCH"):
        context.compile("qa-generation", material().model_copy(update={field: value}))


@pytest.mark.parametrize("field,value", [("wire_bytes", b"{}"), ("count_bytes", b"{}"),
                                        ("attempt_id", "f" * 64), ("native_call_id", "foreign")])
def test_forged_compiled_operation_refuses_before_pricing(tmp_path: Path, field: str, value) -> None:
    from dataclasses import replace
    module, binding, registration, _ = fixture(tmp_path)
    context = module.DiagnosticExecutionContext.load(binding, registration)
    compiled = replace(context.compile("qa-generation", material()), **{field: value})
    with pytest.raises(ValueError, match="COMPILED_REQUEST_MISMATCH"):
        context.price_operations(compiled, "e" * 64)


@pytest.mark.parametrize("native_baseline,stage_count", [("bot_style", 3), ("dc_rs", 2)])
def test_plan_runtime_selector_exercises_native_reopen(tmp_path: Path, monkeypatch, native_baseline: str, stage_count: int) -> None:
    from memcontam.experiment import phase13_ordinary_runtime

    from .phase13_recovery_offline import install_offline_denial
    from .test_phase13_diagnostic_native_runtime import (
        test_registered_v4_native_output_reopens_without_v2_relabel as exercise_native_reopen,
    )
    install_offline_denial(monkeypatch)
    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    exercise_native_reopen(tmp_path, native_baseline, stage_count)
