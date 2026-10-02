from __future__ import annotations

import hashlib
import importlib
import json
import os
import socket
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.readiness.phase13_v3_entrypoint import SelectedExecutionV3

from .phase13_corrective_identity import corrective_identity
from .test_phase13_v3_entrypoint_fixture import (
    AUTHORITY, build_entrypoint_bytes, entrypoint_bytes, entrypoint_fixture, seal_fixture_closure,
)

__all__ = ["entrypoint_bytes", "entrypoint_fixture"]


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
        entitlements = run.dispatcher().retry_entitlements
        assert {row.dispatch_id for row in selected.costs.resources.phase4.base.retry_reservations} == entitlements
        assert entitlements
        assert entitlements == run.dispatcher().retry_entitlements
        assert entitlements <= set(run.ledger.binding.unit_ids)
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


def test_current_selected_resources_reject_historical_observability_packet(entrypoint_fixture, deny_external):
    from memcontam.readiness.phase13_main_preloaded_resources import PreloadedMainResources
    from memcontam.readiness.phase13_v3_entrypoint import EntrypointError, select_execution

    selected = select_execution(entrypoint_fixture, "validate")
    assert isinstance(selected, SelectedExecutionV3)
    try:
        historical = (Path(__file__).resolve().parents[1]
                      / "data/phase13/observability/registration_packet_v1.json").read_bytes()
        role = selected.resource_binding("observability_packet")
        tampered = replace(selected, resources=tuple(
            replace(row, raw=historical) if row.binding.path == role.path else row
            for row in selected.resources))

        with pytest.raises(EntrypointError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
            PreloadedMainResources(tampered)
    finally:
        selected.close()


@pytest.mark.parametrize("role,key", (("implementation_identities", "registration"),
                                      ("implementation_identities", "sequence"),
                                     ("applicability_identities", "game24"),
                                     ("implementation_identities", "authority_state")))
def test_rehashed_current_packet_cannot_forge_bound_identity(entrypoint_fixture, deny_external, role, key):
    from memcontam.readiness.phase13_main_preloaded_resources import PreloadedMainResources
    from memcontam.readiness.phase13_v3_entrypoint import EntrypointError, select_execution

    selected = select_execution(entrypoint_fixture, "validate")
    assert isinstance(selected, SelectedExecutionV3)
    try:
        payload = json.loads(selected.resource("observability_packet"))
        payload[role][key]["sha256"] = "0" * 64
        raw = json.dumps(payload).encode()
        bound = selected.resource_binding("observability_packet")
        changed = bound.model_copy(update={"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})
        package = selected.package.model_copy(update={"resources": tuple(
            changed if row.role == "observability_packet" else row for row in selected.package.resources)})
        resources = tuple(replace(row, binding=row.binding.model_copy(update={
            "sha256": changed.sha256, "size": changed.size}), raw=raw)
            if row.binding.path == bound.path else row for row in selected.resources)

        with pytest.raises(EntrypointError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
            PreloadedMainResources(replace(selected, package=package, resources=resources))
    finally:
        selected.close()


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
        parent_id = selected.package.production[0].unit_id
        parent = json.loads(run.ledger.read_record(f"{parent_id}.parent.json"))
        authority_contract = parent["unit_evidence"]["provider_calls"][0]["provider_authority_contract"]
        assert authority_contract["execution_envelope_id"] == selected.package.authority.registry.registry_id
        assert authority_contract["execution_envelope_sha256"] == selected.package.authority.registry.sha256
        assert authority_contract["terminal_failure_contract_id"] == selected.package.authority.terminal.contract_id
    finally:
        run.close()

    original = (entrypoint_fixture.repository_root / "fake-run" / f"{parent_id}.parent.json").read_bytes()
    parent_path = entrypoint_fixture.repository_root / "fake-run" / f"{parent_id}.parent.json"
    database = entrypoint_fixture.repository_root / "fake-run" / "main_run_ledger_v3.sqlite3"
    source = json.loads(original)
    records = source["unit_evidence"]["evidence"]["runtime_evidence"]["production_observability_archive"]["records"]
    assert len(records) == 50
    for mutation in ("schema", "method_call", "sample_order", "omitted_record", "identity"):
        changed = json.loads(original)
        archive = changed["unit_evidence"]["evidence"]["runtime_evidence"]["production_observability_archive"]
        rows = archive["records"]
        if mutation == "schema":
            archive["schema_version"] = "phase13_production_observability_archive_v1"
            for row in rows:
                row["scientific_result"] = False
        elif mutation == "method_call":
            rows[0]["method_calls"][0]["call_id"] = "invented-archive-call"
        elif mutation == "sample_order":
            rows[0], rows[1] = rows[1], rows[0]
        elif mutation == "omitted_record":
            rows.pop()
        else:
            rows[0]["execution_template_id"] = "invented-template"
        raw = json.dumps(changed, sort_keys=True, allow_nan=False).encode()
        parent_path.write_bytes(raw)
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE parents SET raw=?, sha256=? WHERE unit_id=?",
                               (raw, hashlib.sha256(raw).hexdigest(), parent_id))
        try:
            selected = select_execution(entrypoint_fixture, "resume")
            assert isinstance(selected, SelectedExecutionV3)
            with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
                reopened = V3MainRun.open(selected, entrypoint_fixture.repository_root / "fake-run", create=False, seed=0)
                reopened.close()
        finally:
            parent_path.write_bytes(original)
            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE parents SET raw=?, sha256=? WHERE unit_id=?",
                                   (original, hashlib.sha256(original).hexdigest(), parent_id))

    selected = select_execution(entrypoint_fixture, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    reopened = V3MainRun.open(selected, entrypoint_fixture.repository_root / "fake-run", create=False, seed=0)
    reopened.close()


def test_runner_parent_calls_preserve_sparse_scheduled_request_keys(entrypoint_fixture, deny_external):
    from memcontam.baselines.contracts import BaselineExecutionOutcome
    from memcontam.clients.base import LLMResponse
    from memcontam.experiment.phase12.runtime_registry import NOMEM_SINGLETON, RuntimeTrialResult
    from memcontam.logging.schema import MethodCall
    from memcontam.readiness.phase13_main_request_client import MainRequestClientV3
    from memcontam.readiness.phase13_main_v3_runner import V3MainRun
    from memcontam.readiness.phase13_v3_entrypoint import select_execution
    from memcontam.readiness.phase13_v3_request import RequestKeyV3

    class EligibleTimeout(TimeoutError):
        phase13_retry_class = "TIMEOUT_BEFORE_SEMANTIC_PAYLOAD"

    attempts = 0

    class Provider:
        def send_compiled_v3(self, compiled, before_request):
            nonlocal attempts
            before_request()
            attempts += 1
            if attempts == 1:
                raise EligibleTimeout()
            return LLMResponse("final: 0", {"status": "completed", "usage": {"input_tokens": 1,
                "output_tokens": 0}, "authoritative_provider_cost_usd": "0.0000002", "currency": "USD"}, {}, 0)

    selected = select_execution(entrypoint_fixture, "run")
    assert isinstance(selected, SelectedExecutionV3)
    run = V3MainRun.open(selected, entrypoint_fixture.repository_root / "sparse-run", create=True, seed=0)
    try:
        parent_id = selected.package.production[0].unit_id
        client = MainRequestClientV3(run.dispatcher(lambda _binding: Provider()), parent_id,
                                     lambda: selected.preflight(selected.repository_root))
        messages = [{"role": "user", "content": "fixture"}]

        def execute():
            client.chat(messages, "gpt-5.6-luna", {"method_stage": "no_memory_generate"})
            return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), NOMEM_SINGLETON)

        client.trial(execute, lambda: b"{}", ordinal_base=0)
        first_key = RequestKeyV3(parent_id=parent_id, stage="no_memory_generate", ordinal=0)
        run.ledger.reconcile_cost(first_key.dispatch_id,
            {"usage": {"input_tokens": 1, "output_tokens": 0}}, "f" * 64, attempt_index=0)
        client.trial(execute, lambda: b"{}", ordinal_base=2)
        calls = tuple(MethodCall(call_id=f"sparse:trial:{index}:call:1", stage="no_memory_generate",
                                 messages=messages, raw_response="final: 0", model="gpt-5.6-luna",
                                 temperature=0.0, top_p=1.0)
                      for index in (1, 2))

        enriched = run._strict_calls(client, calls)

        assert tuple(call.dispatch_id for call in enriched) == tuple(RequestKeyV3(
            parent_id=parent_id, stage="no_memory_generate", ordinal=ordinal).dispatch_id
            for ordinal in (0, 2))
        run._validate_parent_calls(parent_id, enriched)
        from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
        replayed = TerminalLedgerV3.open_guarded(run.private, run.ledger.binding)
        try:
            state = replayed.state(first_key.dispatch_id)
            assert state.kind == "COMPLETED" and state.completion_hash != state.event_hash
            replace(run, ledger=replayed)._validate_parent_calls(parent_id, enriched)
        finally:
            replayed.close()
    finally:
        run.close()


def test_runner_reconciles_crashed_retry_before_parent_publication(entrypoint_fixture, deny_external, monkeypatch):
    import memcontam.readiness.phase13_main_request_dispatch as dispatch
    from memcontam.clients.base import LLMResponse
    from memcontam.readiness.phase13_main_v3_runner import V3MainRun
    from memcontam.readiness.phase13_v3_entrypoint import select_execution
    from memcontam.readiness.phase13_v3_request import RequestKeyV3

    class EligibleTimeout(TimeoutError):
        phase13_retry_class = "TIMEOUT_BEFORE_SEMANTIC_PAYLOAD"

    class Crash(BaseException):
        pass

    requests: list[str] = []

    class Provider:
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            requests.append(compiled.key.dispatch_id)
            if len(requests) == 1:
                raise EligibleTimeout()
            return LLMResponse("final: 0", {
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "attempts": 1,
                "authoritative_provider_cost_usd": "0.0000014", "currency": "USD",
                "status": "completed", "response_id": f"fake-{len(requests)}",
                "model": "gpt-5.6-luna", "service_tier": "default",
            }, {"prompt_tokens": 1, "completion_tokens": 1}, 0)

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_args: 1)
    selected = select_execution(entrypoint_fixture, "run")
    assert isinstance(selected, SelectedExecutionV3)
    directory = selected.repository_root / "retry-crash-run"
    first = V3MainRun.open(selected, directory, create=True, seed=0)
    parent_id = selected.package.production[0].unit_id
    original = dispatch.ProductionRequestDispatcherV3._append

    def crash_after_retryable(self, key, kind, extra=None):
        original(self, key, kind, extra)
        if kind == "RETRYABLE_ATTEMPT_FAILURE":
            raise Crash()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(dispatch.ProductionRequestDispatcherV3, "_append", crash_after_retryable)
            with pytest.raises(Crash):
                first.execute(selected.repository_root / "cache", max_units=1,
                              tranche_ceiling_krw=450000, provider_factory=lambda _: Provider())
    finally:
        first.close()
    selected = select_execution(entrypoint_fixture, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    reopened = V3MainRun.open(selected, directory, create=False, seed=0)
    try:
        assert len(requests) == 1
        key = RequestKeyV3(parent_id=parent_id, stage="no_memory_generate", ordinal=0)
        assert reopened.ledger.state(key.dispatch_id).kind == "RETRYABLE_ATTEMPT_FAILURE"
        reopened.ledger.reconcile_cost(key.dispatch_id,
            {"usage": {"input_tokens": 1, "output_tokens": 1}}, "f" * 64, attempt_index=0)
        report = reopened.execute(selected.repository_root / "cache", max_units=1,
                                  tranche_ceiling_krw=450000, provider_factory=lambda _: Provider())
        assert report.completed_count == 1
        assert report.provider_calls_issued == len(requests) == 51
        parent = json.loads(reopened.ledger.read_record(f"{parent_id}.parent.json"))
        assert parent["unit_evidence"]["realized_cost_krw"] == reopened.ledger.realized_cost_krw()
    finally:
        reopened.close()


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


def test_live_cli_resume_enforces_cumulative_tranche_then_continues_without_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], deny_external,
) -> None:
    from memcontam.clients.base import LLMResponse
    from memcontam.readiness import phase13_main_live_cli
    from memcontam.readiness.phase13_main_v3_runner import V3MainRun

    for path, raw in build_entrypoint_bytes((0, 1)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    package = json.loads((tmp_path / "package.json").read_bytes())
    first, second = package["production"]
    calls: list[str] = []

    class Provider:
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            calls.append(compiled.key.dispatch_id)
            return LLMResponse("final: 0", {
                "usage": {"input_tokens": 1, "output_tokens": 1}, "attempts": 1,
                "authoritative_provider_cost_usd": "0.0000014", "currency": "USD",
                "status": "completed", "response_id": f"fake-{len(calls)}",
                "model": "gpt-5.6-luna", "service_tier": "default",
            }, {"prompt_tokens": 1, "completion_tokens": 1}, 0)

    original_execute = V3MainRun.execute

    def fake_execute(self, cache, *, max_units, tranche_ceiling_krw):
        return original_execute(self, cache, max_units=max_units,
            tranche_ceiling_krw=tranche_ceiling_krw, provider_factory=lambda _binding: Provider())

    monkeypatch.setattr(V3MainRun, "execute", fake_execute)
    arguments = ["--repository-root", str(tmp_path), "--package", str(tmp_path / "package.json"),
        "--authorization", str(tmp_path / "authorization.json"), "--authority-root", str(AUTHORITY),
        "--expected-authorization-sha256-file", str(tmp_path / "authorization.sha256"),
        "--run-root", str(tmp_path), "--run-id", corrective_identity().run_id,
        "--cache-root", str(tmp_path / "cache"), "--max-units", "1", "--allow-live-calls"]

    def invoke(command: str, seed: int, ceiling: int) -> dict:
        monkeypatch.setattr(sys, "argv", ["phase13-main-a-live", command, *arguments,
            "--seed", str(seed), "--tranche-ceiling-krw", str(ceiling)])
        phase13_main_live_cli.main()
        return json.loads(capsys.readouterr().out)

    completed = invoke("run", 0, 450000)
    assert completed["completed_count"] == 1
    assert len(calls) == 50
    paused = invoke("resume", 1, second["projected_cost_krw"] - 1)
    assert paused["session_state"] == "PAUSED_BEFORE_DISPATCH"
    database = tmp_path / corrective_identity().run_id / "main_run_ledger_v3.sqlite3"
    with sqlite3.connect(database) as connection:
        last_event = connection.execute("SELECT raw FROM run_journal ORDER BY sequence DESC LIMIT 1").fetchone()[0]
    assert json.loads(last_event)["outer_code"] == "MAIN_TRANCHE_CEILING_EXCEEDED"
    assert paused["provider_calls_issued"] == 50
    assert len(calls) == 50
    continued = invoke("resume", 1, 450000)
    assert continued["completed_count"] == 2
    assert continued["provider_calls_issued"] == len(calls) == len(set(calls)) == 100
    assert first["unit_id"] != second["unit_id"]
