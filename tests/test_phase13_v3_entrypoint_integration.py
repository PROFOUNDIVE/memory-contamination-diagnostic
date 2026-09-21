from __future__ import annotations

import hashlib
import importlib
import json
import os
import socket
import sys
from dataclasses import replace

import pytest

from memcontam.readiness.phase13_v3_entrypoint import SelectedExecutionV3
from .phase13_corrective_identity import corrective_identity

from .test_phase13_v3_entrypoint_fixture import entrypoint_bytes as entrypoint_bytes
from .test_phase13_v3_entrypoint_fixture import entrypoint_fixture as entrypoint_fixture


@pytest.fixture
def deny_external(monkeypatch):
    from memcontam.clients.openai_responses import OpenAIResponsesClient

    counts = {"constructor": 0, "request": 0, "socket": 0, "credential": 0}

    def deny(kind):
        def rejected(*args, **kwargs):
            counts[kind] += 1
            pytest.fail("unexpected external access: " + kind)
        return rejected

    monkeypatch.setattr(OpenAIResponsesClient, "__init__", deny("constructor"))
    monkeypatch.setattr(OpenAIResponsesClient, "send_compiled_v3", deny("request"))
    monkeypatch.setattr(socket, "socket", deny("socket"))
    monkeypatch.setattr(socket, "getaddrinfo", deny("socket"))
    original = type(os.environ).__getitem__

    def environment(environ, name):
        if name in {"OPENAI_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY"}:
            return deny("credential")()
        return original(environ, name)

    monkeypatch.setattr(type(os.environ), "__getitem__", environment)
    yield counts
    assert counts == {"constructor": 0, "request": 0, "socket": 0, "credential": 0}


@pytest.mark.parametrize("module", ["phase13_main_live_cli", "phase13_main_runner_cli"])
def test_cli_validate_uses_selector_before_any_runtime(tmp_path, monkeypatch, module):
    cli = importlib.import_module("memcontam.readiness." + module)
    package = tmp_path / "package.json"
    package.write_text(json.dumps({"schema_version": "phase13_main_execution_freeze_v3",
        "package_id": corrective_identity().package_id, "identity": corrective_identity().model_dump()}))
    monkeypatch.setattr(sys, "argv", [module, "validate", "--repository-root", str(tmp_path),
        "--package", str(package), "--authorization", str(tmp_path / "auth.json"),
        "--expected-authorization-sha256-file", str(tmp_path / "auth.sha256"),
        "--cache-root", str(tmp_path / "cache")])
    with pytest.raises(SystemExit, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        cli.main()


def test_guarded_terminal_store_reopens_and_preserves_events(tmp_path):
    from memcontam.readiness.phase13_v3_entrypoint_paths import private_ledger
    from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3

    binding = {"schema_version": "phase13_main_run_ledger_v3", "unit_ids": ["a" * 64],
               "identity": corrective_identity().model_dump(mode="json"),
               "package_sha256": "b" * 64, "authorization_sha256": "c" * 64}
    with private_ledger(tmp_path / "fixture", create=True) as private:
        ledger = TerminalLedgerV3.create_guarded(private, binding)
        state = ledger.state("a" * 64)
        ledger.append({"schema_version": "phase13_main_dispatch_evidence_v3",
            "kind": "DISPATCH_INTENT", "unit_id": "a" * 64,
            "revision": 1, "previous_hash": state.event_hash, "compiled": None})
    with private_ledger(tmp_path / "fixture", create=False) as private:
        reopened = TerminalLedgerV3.open_guarded(private, ledger.binding)
        assert reopened.state("a" * 64).kind == "DISPATCH_INTENT_PERSISTED"
        (private.directory / "main_run_ledger_v3.sqlite3").chmod(0o644)
        with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
            reopened.rows()


def test_legacy_active_runner_rejects_before_reading_package(tmp_path):
    from memcontam.readiness.phase13_main_runner import MainRunRequest, prepare_main_run

    request = MainRunRequest(tmp_path, tmp_path / "absent", tmp_path / "absent-auth",
                             "a" * 64, tmp_path, "old-v2", 0)
    with pytest.raises(ValueError, match="MAIN_CORRECTED_RUN_ID_MISMATCH"):
        prepare_main_run(request)


@pytest.mark.parametrize("module", ["phase13_main_live_cli", "phase13_main_runner_cli"])
def test_exact_v3_cli_validate_is_provider_free(entrypoint_fixture, monkeypatch, capsys, module, deny_external):
    request = entrypoint_fixture
    cli = importlib.import_module("memcontam.readiness." + module)
    monkeypatch.setattr(sys, "argv", [module, "validate", "--repository-root", str(request.repository_root),
        "--package", str(request.package_path), "--authorization", str(request.authorization_path),
        "--authority-root", str(request.authority_root), "--expected-authorization-sha256-file",
        str(request.expected_authorization_sha256_file), "--cache-root", str(request.repository_root / "cache")])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "READY_NO_CALLS"


@pytest.mark.parametrize("command", ["run", "resume"])
def test_both_active_runner_apis_reject_old_identity(entrypoint_fixture, command, deny_external):
    from memcontam.readiness.phase13_main_runner import (
        MainRunRequest,
        open_main_run,
        prepare_main_run,
    )

    request = entrypoint_fixture
    active = MainRunRequest(request.repository_root, request.package_path, request.authorization_path, "",
        request.repository_root, "old-run", 0, request.authority_root, request.expected_authorization_sha256_file)
    with pytest.raises(ValueError, match="MAIN_CORRECTED_RUN_ID_MISMATCH"):
        (prepare_main_run if command == "run" else open_main_run)(active)


def test_valid_v3_guarded_run_and_resume_without_calls(entrypoint_fixture, deny_external):
    from memcontam.readiness.phase13_main_v3_runner import V3MainRun
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    directory = entrypoint_fixture.repository_root / "fixture-ledger"
    selected = select_execution(entrypoint_fixture, "run")
    assert isinstance(selected, SelectedExecutionV3)
    run = V3MainRun.open(selected, directory, create=True, seed=0)
    try:
        assert run.execute(directory / "cache", max_units=0, tranche_ceiling_krw=450000).provider_calls_issued == 0
    finally:
        run.close()
    selected = select_execution(entrypoint_fixture, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    resumed = V3MainRun.open(selected, directory, create=False, seed=0)
    try:
        assert resumed.status().pending_count == 1
    finally:
        resumed.close()


@pytest.mark.parametrize("change", ["bytes", "symlink"])
def test_authorized_resources_cannot_change_preflight(entrypoint_fixture, deny_external, change):
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    selected = select_execution(entrypoint_fixture, "validate")
    assert isinstance(selected, SelectedExecutionV3)
    path = entrypoint_fixture.repository_root / selected.package.resources[0].path
    original = selected.resource("common_checkpoint_registry")
    try:
        if change == "bytes":
            path.write_bytes(b"tampered")
        else:
            path.unlink()
            path.symlink_to(entrypoint_fixture.package_path)
        with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
            selected.preflight(entrypoint_fixture.repository_root)
        assert selected.resource("common_checkpoint_registry") == original
    finally:
        selected.close()


@pytest.mark.parametrize("field,value", [("authority_root", None), ("expected_authorization_sha256_file", None)])
def test_v3_requires_explicit_authority_and_sidecar(entrypoint_fixture, deny_external, field, value):
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        select_execution(replace(entrypoint_fixture, **{field: value}), "validate")


def test_v3_execution_uses_guarded_requests_and_real_ordinary_runtime(entrypoint_fixture, deny_external, monkeypatch):
    import memcontam.readiness.phase13_main_request_dispatch as dispatch
    from memcontam.clients.base import LLMResponse
    from memcontam.readiness.phase13_main_v3_runner import V3MainRun
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: 1)
    calls = []

    class FakeProvider:
        def send_compiled_v3(self, compiled, before_request):
            assert compiled.native_state == b"{}"
            before_request()
            calls.append(compiled.key.dispatch_id)
            return LLMResponse("final: 0", {
                "usage": {"input_tokens": 1, "output_tokens": 0}, "attempts": 1,
                "authoritative_provider_cost_usd": "0.0000002", "currency": "USD",
                "status": "completed", "response_id": f"fake-{compiled.key.dispatch_id}",
                "model": "gpt-5.6-luna", "service_tier": "default",
            }, {"prompt_tokens": 1, "completion_tokens": 0}, 0)

    selected = select_execution(entrypoint_fixture, "run")
    assert isinstance(selected, SelectedExecutionV3)
    run = V3MainRun.open(selected, entrypoint_fixture.repository_root / "fake-run", create=True, seed=0)
    try:
        report = run.execute(entrypoint_fixture.repository_root / "cache", max_units=1, tranche_ceiling_krw=450000,
                             provider_factory=lambda _: FakeProvider())
        assert report.completed_count == 1
        assert report.provider_calls_issued == 50
        assert len(set(calls)) == 50
        assert all(state.kind == "COMPLETED" for state in run.ledger.states().values())
    finally:
        run.close()


def _seal(payload, field):
    body = {key: value for key, value in payload.items() if key != field}
    payload[field] = hashlib.sha256((json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_self_rehashed_foreign_cost_table_never_reaches_provider(entrypoint_fixture, deny_external):
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    request = entrypoint_fixture
    proof_path = request.repository_root / "data/phase13/main/cost_envelope_v3/cost_proof_v3.json"
    proof = json.loads(proof_path.read_bytes())
    proof["projected_krw"][0]["projected_krw"] += 1
    proof_path.write_bytes(_seal(proof, "proof_hash"))
    package = json.loads(request.package_path.read_bytes())
    package["cost_proof_hash"] = proof["proof_hash"]
    package_raw = _seal(package, "package_hash")
    request.package_path.write_bytes(package_raw)
    authorization = json.loads(request.authorization_path.read_bytes())
    authorization["execution_package_hash"] = package["package_hash"]
    authorization["execution_package_sha256"] = hashlib.sha256(package_raw).hexdigest()
    raw = _seal(authorization, "authorization_hash")
    request.authorization_path.write_bytes(raw)
    request.expected_authorization_sha256_file.write_text(hashlib.sha256(raw).hexdigest() + "\n")
    with pytest.raises(ValueError, match="MAIN_COST_PROOF_MISMATCH"):
        select_execution(request, "validate")


@pytest.mark.parametrize("target", ["package", "authorization", "sidecar", "resource"])
def test_symlinked_bound_input_is_rejected_before_provider(entrypoint_fixture, deny_external, target):
    from memcontam.readiness.phase13_main_resource_contract import RESOURCE_PATHS
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    request = entrypoint_fixture
    path = {"package": request.package_path, "authorization": request.authorization_path,
            "sidecar": request.expected_authorization_sha256_file,
            "resource": request.repository_root / RESOURCE_PATHS["candidate_registry"]}[target]
    saved = path.with_name(path.name + ".saved")
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
        select_execution(request, "validate")


@pytest.mark.parametrize("field,value", [("executable", "/usr/bin/python3"), ("version", "3.11.14"),
    ("prefix", "/wrong-prefix"), ("requirements_sha256", "0" * 64),
    ("requirements_dev_sha256", "0" * 64), ("package_paths", ["/sibling/memcontam"])])
def test_rehashed_runtime_drift_rejects_before_provider(entrypoint_fixture, deny_external, field, value):
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    request = entrypoint_fixture
    package = json.loads(request.package_path.read_bytes())
    package["runtime_identity"][field] = value
    raw_package = _seal(package, "package_hash")
    request.package_path.write_bytes(raw_package)
    authorization = json.loads(request.authorization_path.read_bytes())
    authorization["execution_package_sha256"] = hashlib.sha256(raw_package).hexdigest()
    authorization["execution_package_hash"] = package["package_hash"]
    raw = _seal(authorization, "authorization_hash")
    request.authorization_path.write_bytes(raw)
    request.expected_authorization_sha256_file.write_text(hashlib.sha256(raw).hexdigest() + "\n")
    with pytest.raises(ValueError, match="MAIN_RUNTIME_IDENTITY_DRIFT"):
        select_execution(request, "validate")


def test_proof_bytes_must_be_canonical_not_just_semantically_equal(entrypoint_fixture, deny_external):
    from memcontam.readiness.phase13_v3_entrypoint import select_execution

    path = entrypoint_fixture.repository_root / "data/phase13/main/cost_envelope_v3/cost_proof_v3.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="MAIN_COST_PROOF_MISMATCH"):
        selected = select_execution(entrypoint_fixture, "validate")
        assert isinstance(selected, SelectedExecutionV3)
        selected.close()
