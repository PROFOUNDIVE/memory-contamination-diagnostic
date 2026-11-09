from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_phase13_live_diagnostic.py"
EVIDENCE = Path("/home/hyunwoo/phase13_artifacts/phase13-results-and-main-entry-20261022-001")


def test_real_proposal_reports_prerequisites_instead_of_subject_schema_error(tmp_path, monkeypatch, capsys):
    if not (EVIDENCE / "diagnostic/scope-proposal.json").exists():
        pytest.skip("Task11 frozen external evidence not installed")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "register",
        "--scope", str(EVIDENCE / "diagnostic/scope-proposal.json"),
        "--eligibility", str(EVIDENCE / "diagnostic/eligibility.json"),
        "--budget-binding", str(EVIDENCE / "budget/binding.json"),
        "--execution-approval", str(EVIDENCE / "EXECUTION_APPROVAL.json"),
        "--output-dir", str(tmp_path)])

    result = runpy.run_path(str(SCRIPT))["main"]()

    output = json.loads(capsys.readouterr().out)
    assert result == 2
    assert output["status"] == "PENDING_PREREQUISITES"
    assert len(output["schedule"]) == 16
    assert {row["path"] for row in output["prerequisites"]} >= {
        "checkpoint_groups.0.prefix_id", "checkpoint_groups.0.state_binding",
        "ordinary_units.1.render.render_id", "eligibility.protocol_owner_disposition",
        "preparation.subject", "admission.science", "admission.launch", "admission.credential"}
    assert not (tmp_path / "registration.json").exists()


@pytest.mark.parametrize("mutation,code", [
    ("128", "DIAGNOSTIC_PROPOSAL_SCHEDULE_MISMATCH"),
    ("order", "DIAGNOSTIC_PROPOSAL_SCHEDULE_MISMATCH"),
    ("dc", "DIAGNOSTIC_PROPOSAL_DC_CONTRACT_MISMATCH"),
    ("input", "DIAGNOSTIC_PROPOSAL_COMMON_INPUT_MISMATCH"),
])
def test_proposal_contract_refuses_unsupported_shapes(tmp_path, monkeypatch, capsys, mutation, code):
    if not (EVIDENCE / "diagnostic/scope-proposal.json").exists():
        pytest.skip("Task11 frozen external evidence not installed")
    proposal = json.loads((EVIDENCE / "diagnostic/scope-proposal.json").read_bytes())
    if mutation == "128":
        proposal["ordinary_units"] *= 8
    elif mutation == "order":
        proposal["ordinary_units"][0:2] = reversed(proposal["ordinary_units"][0:2])
    elif mutation == "dc":
        proposal["dc_contract"]["native_order"].reverse()
    else:
        proposal["ordinary_units"][0]["sample_id"] = "foreign-input"
    scope = tmp_path / "proposal.json"
    scope.write_text(json.dumps(proposal))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "register", "--scope", str(scope),
        "--eligibility", str(EVIDENCE / "diagnostic/eligibility.json"),
        "--budget-binding", str(tmp_path / "absent-budget"),
        "--execution-approval", str(tmp_path / "absent-approval"), "--output-dir", str(tmp_path)])

    result = runpy.run_path(str(SCRIPT))["main"]()

    assert result == 2
    assert json.loads(capsys.readouterr().out)["code"] == code


def test_complete_metadata_prepares_exact_subject_without_adopting_it(tmp_path, monkeypatch):
    from memcontam.readiness.phase13_diagnostic_proposal import prepare_proposal
    from memcontam.readiness.phase13_diagnostic_proposal_models import PreparationMetadata
    from .phase13_proposal_fixtures import metadata_fixture

    proposal, metadata, _, subject, _ = metadata_fixture(tmp_path, monkeypatch)

    result = prepare_proposal(proposal, PreparationMetadata.model_validate_json(metadata.read_bytes()))

    assert result.status == "PREPARED_NOT_ADMITTED"
    assert result.subject == subject
    assert result.prerequisites == ()
    assert not Path(subject.registration_path).exists()
    assert not Path(subject.binding.ledger_path).exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_complete_signed_intake_uses_existing_registrar_without_clients_or_ledgers(tmp_path, monkeypatch, capsys, legacy):
    import httpx
    from memcontam.readiness import phase13_diagnostic_cli as cli
    from memcontam.readiness.phase13_shared_provider_budget import SharedProviderBudget
    from .phase13_proposal_fixtures import metadata_fixture

    proposal, metadata, eligibility, subject, _ = metadata_fixture(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("intake constructed a client, embedding or ledger")
    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(cli.LiveDiagnosticLedger, "create", forbidden)
    monkeypatch.setattr(SharedProviderBudget, "create", forbidden)
    monkeypatch.setattr(cli.phase13_diagnostic_cli_native, "load_embedding", forbidden)
    arguments = [str(SCRIPT), "register", "--scope", str(proposal), "--preparation", str(metadata),
        "--eligibility", str(eligibility), "--budget-binding", subject.context().contract.budget_binding.path,
        "--execution-approval", str(tmp_path / "synthetic-owner.json"), "--output-dir", str(tmp_path)]
    if legacy:
        scope = tmp_path / "legacy-subject.json"
        scope.write_text(subject.model_dump_json())
        arguments[3] = str(scope)
        del arguments[4:6]
    monkeypatch.setattr(sys, "argv", arguments)

    result = runpy.run_path(str(SCRIPT))["main"]()

    output = json.loads(capsys.readouterr().out)
    assert result == 0, output
    assert cli.load_registration(Path(subject.registration_path)).subject == subject
    assert cli.preflight(Path(subject.registration_path))["provider_calls"] == 0
    assert not Path(subject.binding.ledger_path).exists()


@pytest.mark.parametrize("mutation", ["signature", "subject", "proposal", "native", "render", "missing_group", "pending"])
def test_preparation_cannot_substitute_for_exact_admission(tmp_path, monkeypatch, capsys, mutation):
    from .phase13_proposal_fixtures import metadata_fixture
    proposal, metadata, eligibility, subject, _ = metadata_fixture(tmp_path, monkeypatch)
    if mutation in {"signature", "subject"}:
        value = json.loads(eligibility.read_bytes())
        value["science"]["signature" if mutation == "signature" else "subject_sha256"] = "00" * (64 if mutation == "signature" else 32)
        eligibility.write_text(json.dumps(value))
    elif mutation == "pending":
        eligibility = EVIDENCE / "diagnostic/eligibility.json"
    else:
        value = json.loads(metadata.read_bytes())
        if mutation == "proposal":
            value["proposal"]["sha256"] = "0" * 64
        elif mutation == "native":
            value["groups"][0]["prefix_id"] = "invented-prefix"
        elif mutation == "render":
            value["renders"] = []
        else:
            del value["groups"][0]["state_binding"]
        metadata.write_text(json.dumps(value))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "register", "--scope", str(proposal),
        "--preparation", str(metadata), "--eligibility", str(eligibility),
        "--budget-binding", subject.context().contract.budget_binding.path,
        "--execution-approval", str(tmp_path / "synthetic-owner.json"), "--output-dir", str(tmp_path)])

    result = runpy.run_path(str(SCRIPT))["main"]()

    assert result == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == ("PENDING_PREREQUISITES" if mutation == "pending" else "REFUSED")
    if mutation == "missing_group":
        assert output["code"] == "DIAGNOSTIC_INPUT_SCHEMA_INVALID"
        assert {"path": "groups.0.state_binding", "type": "missing"} in output["fields"]
    assert not Path(subject.registration_path).exists()
    assert not Path(subject.binding.ledger_path).exists()


@pytest.mark.parametrize("mutation", ["account", "launch", "credential", "cost", "budget", "approval", "source"])
def test_complete_metadata_preserves_existing_admission_gates(tmp_path, monkeypatch, capsys, mutation):
    from memcontam.readiness import phase13_diagnostic_trusted_admission as trust
    from memcontam.readiness.phase13_diagnostic_cli_models import RegistrationSubject
    from .phase13_cli_fixtures import signed_evidence
    from .phase13_proposal_fixtures import metadata_fixture

    proposal, metadata, eligibility, subject, keys = metadata_fixture(tmp_path, monkeypatch)
    budget = subject.context().contract.budget_binding.path
    approval = str(tmp_path / "synthetic-owner.json")
    if mutation in {"account", "launch", "credential"}:
        value = json.loads(eligibility.read_bytes())
        if mutation == "account":
            value["credential"]["account_id"] = "foreign-account"
        else:
            value[mutation]["signature"] = "00" * 64
        eligibility.write_text(json.dumps(value))
    elif mutation == "budget":
        budget = str(tmp_path / "foreign-budget")
    elif mutation == "approval":
        approval = str(tmp_path / "native-metadata-qa.json")
    elif mutation == "source":
        (tmp_path / "native-metadata-qa.json").write_text('{"changed":true}')
    else:
        value = json.loads(metadata.read_bytes())
        value["subject"]["operations"][0]["bound_usd_micros"] = 1
        altered = RegistrationSubject.model_validate_json(json.dumps(value["subject"]))
        metadata.write_text(json.dumps(value))
        admitted, _ = signed_evidence(trust, tmp_path, altered.sha256(), keys)
        eligibility.write_text(admitted.model_dump_json())
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "register", "--scope", str(proposal),
        "--preparation", str(metadata), "--eligibility", str(eligibility), "--budget-binding", budget,
        "--execution-approval", approval, "--output-dir", str(tmp_path)])

    result = runpy.run_path(str(SCRIPT))["main"]()

    assert result == 2
    assert json.loads(capsys.readouterr().out)["status"] == "REFUSED"
    assert not Path(subject.registration_path).exists()
    assert not Path(subject.binding.ledger_path).exists()
