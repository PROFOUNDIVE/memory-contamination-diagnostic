from __future__ import annotations

import pytest
import json
import hashlib
import shutil
import sys
from pathlib import Path
from dataclasses import replace

from memcontam.readiness.phase13_v3_authority_models import V3Identity

from .phase13_corrective_identity import corrective_identity
from .test_phase13_v3_entrypoint_fixture import AUTHORITY, build_entrypoint_bytes, seal_fixture_closure
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external
from .test_phase13_v3_entrypoint_integration import _seal
from memcontam.readiness.phase13_v3_entrypoint import SelectionRequest, SelectedExecutionV3, select_execution
from memcontam.readiness.phase13_main_v3_runner import V3MainRun


def test_runtime_root_is_current_source_not_retired_checkout() -> None:
    from memcontam.readiness.phase13_v3_runtime_identity import ROOT
    assert ROOT == Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("generation", ["disposable-alpha", "disposable-beta"])
def test_fresh_identity_serializes_when_generation_is_explicit(generation: str) -> None:
    identity = corrective_identity(generation)
    reopened = V3Identity.model_validate_json(identity.model_dump_json())
    assert reopened == identity
    assert generation in reopened.run_id


def test_identity_rejects_implicit_old_generation() -> None:
    with pytest.raises(ValueError):
        V3Identity.model_validate({})


@pytest.mark.parametrize("field", ["run_id", "package_id", "authorization_id", "cost_proof_id"])
@pytest.mark.parametrize("value", ["", "../escape", "nested/component", "back\\slash", ".", "..", "bad\n", 3])
def test_identity_rejects_malformed_component(field: str, value: str | int) -> None:
    payload = corrective_identity().model_dump()
    payload[field] = value
    with pytest.raises(ValueError):
        V3Identity.model_validate(payload)


@pytest.mark.parametrize("field,value", [
    ("run_id", "phase13-main-a-corrected-20260905-v3"),
    ("package_id", "phase13-main-a-corrected-execution-freeze-v3"),
    ("authorization_id", "phase13-main-a-corrected-authorized-execution-v3"),
    ("cost_proof_id", "phase13-main-a-corrected-cost-proof-v3"),
])
def test_identity_rejects_old_component_as_new(field: str, value: str) -> None:
    payload = corrective_identity().model_dump()
    payload[field] = value
    with pytest.raises(ValueError):
        V3Identity.model_validate(payload)


@pytest.fixture(scope="session", params=["disposable-alpha", "disposable-beta"])
def fresh_source(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest) -> Path:
    tmp_path = tmp_path_factory.mktemp(request.param)
    identity = corrective_identity(request.param)
    for path, raw in build_entrypoint_bytes((0,), execution_identity=identity).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    return tmp_path


@pytest.fixture
def fresh_selection(tmp_path: Path, fresh_source: Path) -> SelectionRequest:
    shutil.copytree(fresh_source, tmp_path, dirs_exist_ok=True)
    identity = V3Identity.model_validate(json.loads((tmp_path / "package.json").read_bytes())["identity"])
    return SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
        AUTHORITY, tmp_path / "authorization.sha256", identity.run_id)


def test_fresh_generation_selects_and_reopens_zero_call_ledger(fresh_selection, deny_external) -> None:
    directory = fresh_selection.repository_root / "ledger"
    validated = select_execution(fresh_selection, "validate")
    assert isinstance(validated, SelectedExecutionV3)
    identity = validated.package.identity
    assert validated.costs.resources.proof.proof_id == identity.cost_proof_id
    validated.close()
    for command in ("run", "resume", "status"):
        selected = select_execution(fresh_selection, command)
        assert isinstance(selected, SelectedExecutionV3)
        run = V3MainRun.open(selected, directory, create=command == "run", seed=0)
        try:
            assert run.dispatcher().binding.identity == identity
            report = run.execute(directory / "cache", max_units=0, tranche_ceiling_krw=450000)
            assert report.provider_calls_issued == 0
            assert report.pending_count == 1
            assert run.ledger.rows() == ()
        finally:
            run.close()


@pytest.mark.parametrize("command", ["validate", "run", "resume", "status"])
@pytest.mark.parametrize("run_id", ["phase13-main-a-corrected-20260905-v3", "foreign-v3", "../bad"])
def test_selector_rejects_request_generation_before_provider(fresh_selection, deny_external, command, run_id) -> None:
    with pytest.raises(ValueError, match="MAIN_CORRECTED_RUN_ID_MISMATCH"):
        select_execution(replace(fresh_selection, run_id=run_id), command)


@pytest.mark.parametrize("target", ["authorization", "package", "authority", "proof", "contract", "sidecar"])
def test_rehashed_cross_generation_substitution_fails_before_provider(fresh_selection, deny_external, target) -> None:
    request = fresh_selection
    other = corrective_identity("foreign-disposable")
    package = json.loads(request.package_path.read_bytes())
    authorization = json.loads(request.authorization_path.read_bytes())
    match target:
        case "authorization":
            authorization.update(identity=other.model_dump(), authorization_id=other.authorization_id,
                                 execution_package_id=other.package_id)
        case "package":
            package["package_id"] = other.package_id
        case "authority":
            package["authority"]["identity"] = other.model_dump()
        case "proof":
            path = request.repository_root / "data/phase13/main/cost_envelope_v3/cost_proof_v3.json"
            proof = json.loads(path.read_bytes())
            proof["proof_id"] = other.cost_proof_id
            path.write_bytes(_seal(proof, "proof_hash"))
            package["cost_proof_hash"] = proof["proof_hash"]
        case "contract":
            path = request.repository_root / "data/phase13/main/main_live_contract_v3.json"
            contract = json.loads(path.read_bytes())
            contract["identity"] = other.model_dump()
            path.write_bytes(_seal(contract, "contract_hash"))
            package["live_contract_hash"] = contract["contract_hash"]
            from memcontam.readiness.phase13_v3_source_closure import freeze_resources
            closure = freeze_resources(request.repository_root,
                tuple(row["path"] for row in package["generated_closure"]["rows"]))
            package["generated_closure"] = closure.model_dump(mode="json")
            package["generated_closure_hash"] = closure.resource_closure_sha256
        case "sidecar":
            request.expected_authorization_sha256_file.write_text("0" * 64 + "\n")
    package_raw = _seal(package, "package_hash")
    request.package_path.write_bytes(package_raw)
    authorization["execution_package_hash"] = package["package_hash"]
    authorization["execution_package_sha256"] = hashlib.sha256(package_raw).hexdigest()
    raw = _seal(authorization, "authorization_hash")
    request.authorization_path.write_bytes(raw)
    if target != "sidecar":
        request.expected_authorization_sha256_file.write_text(hashlib.sha256(raw).hexdigest() + "\n")
    with pytest.raises(ValueError, match="MAIN_(AUTHORIZATION_BINDING|COST_PROOF|GOVERNED_SOURCE)_MISMATCH|MAIN_GOVERNED_SOURCE_DRIFT"):
        select_execution(request, "validate")


def test_request_identity_mismatch_rejected_even_with_same_hashes(fresh_selection, deny_external) -> None:
    from memcontam.readiness.phase13_main_request_dispatch import ProductionRequestDispatcherV3
    selected = select_execution(fresh_selection, "run")
    assert isinstance(selected, SelectedExecutionV3)
    run = V3MainRun.open(selected, fresh_selection.repository_root / "ledger", create=True, seed=0)
    try:
        dispatcher = run.dispatcher()
        binding = dispatcher.binding.model_copy(update={"identity": corrective_identity("foreign-disposable")})
        with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
            ProductionRequestDispatcherV3(run.ledger, binding, dispatcher.parents)
    finally:
        run.close()


def test_cli_accepts_fresh_identity_without_source_changes(fresh_selection, monkeypatch, capsys, deny_external) -> None:
    from memcontam.readiness.phase13_main_runner_cli import main
    request = fresh_selection
    (request.repository_root / "cli-runs").mkdir()
    common = ["--repository-root", str(request.repository_root), "--package", str(request.package_path),
        "--authorization", str(request.authorization_path), "--authority-root", str(request.authority_root),
        "--expected-authorization-sha256-file", str(request.expected_authorization_sha256_file)]
    for command in ("validate", "run", "resume", "status"):
        active = [] if command == "validate" else ["--run-root", str(request.repository_root / "cli-runs"),
            "--run-id", request.run_id, "--seed", "0", "--max-units", "0"]
        monkeypatch.setattr(sys, "argv", ["phase13-main", command, *common, *active])
        main()
        assert json.loads(capsys.readouterr().out)["provider_calls_issued"] == 0


def test_foreign_generation_cannot_reopen_existing_ledger(fresh_selection, deny_external) -> None:
    from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
    selected = select_execution(fresh_selection, "run")
    assert isinstance(selected, SelectedExecutionV3)
    run = V3MainRun.open(selected, fresh_selection.repository_root / "ledger", create=True, seed=0)
    try:
        foreign = run.ledger.binding.model_copy(update={"identity": corrective_identity("foreign")})
        with pytest.raises(ValueError, match="MAIN_TERMINAL_EVIDENCE_CONFLICT"):
            TerminalLedgerV3.open_guarded(run.private, foreign)
    finally:
        run.close()
