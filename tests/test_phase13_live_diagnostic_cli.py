"""Local QA only; ephemeral signatures are not operational scientific authority."""
from __future__ import annotations

import importlib
import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_phase13_live_diagnostic.py"


def admission_api():
    name = "memcontam.readiness.phase13_diagnostic_trusted_admission"
    assert importlib.util.find_spec(name) is not None, "fixed trusted admission verifier missing"
    return importlib.import_module(name)


def test_cli_help_exposes_required_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert all(command in output for command in ("preflight", "register", "run", "verify", "report"))


def test_trusted_admission_does_not_accept_unsigned_fingerprint(tmp_path: Path) -> None:
    import pytest

    module = admission_api()
    unsigned = tmp_path / "admission.json"
    unsigned.write_text('{"accepted": true, "credential_fingerprint": "' + "a" * 64 + '"}')
    with pytest.raises(ValueError):
        module.AdmissionEvidence.model_validate_json(unsigned.read_bytes())


@pytest.mark.parametrize("mutation", ["science", "launch", "credential", "subject", "account", "expired", "financial_key"])
def test_role_separated_signatures_refuse_forgery(tmp_path: Path, monkeypatch, mutation: str) -> None:
    signed_evidence = importlib.import_module("tests.phase13_cli_fixtures").signed_evidence

    module = admission_api()
    evidence, policy = signed_evidence(module, tmp_path)
    monkeypatch.setattr(module, "load_policy", lambda: policy)
    changed = evidence.model_dump(mode="json")
    if mutation in {"science", "launch", "credential"}:
        changed[mutation]["signature"] = "00" * 64
    elif mutation == "subject":
        changed["science"]["subject_sha256"] = "b" * 64
    elif mutation == "account":
        changed["credential"]["account_id"] = "foreign-account"
    elif mutation == "expired":
        changed["launch"]["expires"] = "2000-01-01"
    else:
        policy = policy.model_copy(update={"science": policy.credential})
        monkeypatch.setattr(module, "load_policy", lambda: policy)

    with pytest.raises(ValueError):
        module.authenticate(module.AdmissionEvidence.model_validate_json(__import__("json").dumps(changed)), "a" * 64, "qa-account")


def test_authentic_synthetic_role_signatures_are_not_permanently_refused(tmp_path: Path, monkeypatch) -> None:
    signed_evidence = importlib.import_module("tests.phase13_cli_fixtures").signed_evidence

    module = admission_api()
    evidence, policy = signed_evidence(module, tmp_path)
    monkeypatch.setattr(module, "load_policy", lambda: policy)

    module.authenticate(evidence, "a" * 64, "qa-account")

    assert evidence.credential.credential_fingerprint == module.credential_fingerprint("qa-not-an-operational-secret")


def cli_api():
    name = "memcontam.readiness.phase13_diagnostic_cli"
    assert importlib.util.find_spec(name) is not None, "registered native CLI missing"
    return importlib.import_module(name)


def test_registered_sixteen_native_observations_reopen_one_parent(tmp_path: Path, monkeypatch) -> None:
    module = cli_api()
    registered_fixture = importlib.import_module("tests.phase13_cli_native_fixtures").registered_fixture

    registration, parent, sent = registered_fixture(tmp_path, monkeypatch)
    prepared = module.preflight(registration)
    assert prepared["observations"] == 16 and sent == [] and not (parent / "ledger.sqlite3").exists()

    for child in ("qa-bot", "qa-dc"):
        module.run_registered(registration, child, "qa-not-an-operational-secret")

    status = module.verify_registered(registration, parent)
    assert status["completed"] == 16 and status["pending"] == 0
    assert len(sent) == 80
    with pytest.raises(ValueError):
        module.run_registered(registration, "qa-bot", "qa-not-an-operational-secret")
    assert len(sent) == 80


@pytest.mark.parametrize("failure", ["incomplete", "count_timeout", "after_receipt"])
def test_terminal_or_ambiguous_execution_never_redispatches(tmp_path: Path, monkeypatch, failure: str) -> None:
    module = cli_api()
    registered_fixture = importlib.import_module("tests.phase13_cli_native_fixtures").registered_fixture

    registration, parent, sent = registered_fixture(tmp_path, monkeypatch, failure=failure)
    try:
        module.run_registered(registration, "qa-bot", "qa-not-an-operational-secret")
    except SystemExit:
        assert failure == "after_receipt"
    before = len(sent)

    with pytest.raises(ValueError):
        module.run_registered(registration, "qa-bot", "qa-not-an-operational-secret")

    status = module.verify_registered(registration, parent)
    assert len(sent) == before
    assert status["terminal"] + status["in_flight"] == 1


@pytest.mark.parametrize("mutation", ["science", "launch", "proposal128", "price", "credential", "child", "source", "cost"])
def test_cli_refuses_before_unauthorized_egress(tmp_path: Path, monkeypatch, mutation: str) -> None:
    module = cli_api()
    fixtures = importlib.import_module("tests.phase13_cli_native_fixtures")
    registered_fixture, mutate_registration = fixtures.registered_fixture, fixtures.mutate_registration

    registration, parent, sent = registered_fixture(tmp_path, monkeypatch)
    changed = mutate_registration(registration, mutation)

    with pytest.raises(ValueError):
        module.run_registered(changed, "foreign" if mutation == "child" else "qa-bot",
                              "wrong-key" if mutation == "credential" else "qa-not-an-operational-secret")

    assert sent == [] and not (parent / "ledger.sqlite3").exists()


def test_registration_intake_accepts_plan_arguments_without_authorizing_missing_facts(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "register", "--scope", str(tmp_path / "scope.json"),
        "--eligibility", str(tmp_path / "eligibility.json"), "--budget-binding", str(tmp_path / "budget.json"),
        "--execution-approval", str(tmp_path / "approval.json"), "--output-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert stopped.value.code == 2
    import json
    assert json.loads(capsys.readouterr().out)["status"] == "REFUSED"


@pytest.mark.parametrize("failure", ["qa_client", "cost_contradiction", "after_output"])
def test_cli_preserves_native_guards_and_output_recovery(tmp_path: Path, monkeypatch, failure: str) -> None:
    registered_fixture = importlib.import_module("tests.phase13_cli_native_fixtures").registered_fixture

    module = cli_api()
    registration, parent, sent = registered_fixture(tmp_path, monkeypatch, failure=failure)
    with pytest.raises((ValueError, SystemExit)):
        module.run_registered(registration, "qa-bot", "qa-not-an-operational-secret")
    initial = module.verify_registered(registration, parent)
    assert initial["completed"] == 0
    if failure == "after_output":
        monkeypatch.undo()
        reopened = module.recover_registered(registration, parent)
        assert reopened["completed"] == 1 and reopened["in_flight"] == 0
    else:
        assert (len(sent) == 0) == (failure == "qa_client")


def test_preflight_and_register_never_construct_client(tmp_path: Path, monkeypatch) -> None:
    import httpx
    registered_fixture = importlib.import_module("tests.phase13_cli_native_fixtures").registered_fixture

    def forbidden(**kwargs):
        pytest.fail("preflight/register constructed provider client")
    monkeypatch.setattr(httpx, "Client", forbidden)
    registration, parent, sent = registered_fixture(tmp_path, monkeypatch)

    assert cli_api().preflight(registration)["provider_calls"] == 0
    assert sent == [] and not (parent / "ledger.sqlite3").exists()


def test_signed_operation_plan_is_checked_at_last_pre_egress_boundary(tmp_path: Path, monkeypatch) -> None:
    fixtures = importlib.import_module("tests.phase13_cli_native_fixtures")
    registration, parent, sent = fixtures.registered_fixture(tmp_path, monkeypatch, failure="unplanned_wire")

    status = cli_api().run_registered(registration, "qa-bot", "qa-not-an-operational-secret")

    assert sent == [] and status["in_flight"] == 1
    assert cli_api().verify_registered(registration, parent)["unresolved_requests"] == 1


@pytest.mark.parametrize("missing", ["science", "launch"])
def test_missing_role_is_not_authority(tmp_path: Path, missing: str) -> None:
    fixtures = importlib.import_module("tests.phase13_cli_fixtures")
    module = admission_api()
    evidence, _ = fixtures.signed_evidence(module, tmp_path)
    raw = evidence.model_dump(mode="json")
    del raw[missing]
    with pytest.raises(ValueError):
        module.AdmissionEvidence.model_validate(raw)
