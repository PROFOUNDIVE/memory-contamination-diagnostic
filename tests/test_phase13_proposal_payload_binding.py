from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import httpx
import pytest

from memcontam.readiness import phase13_diagnostic_cli as cli
from memcontam.readiness.phase13_diagnostic_proposal import prepare_proposal
from memcontam.readiness.phase13_diagnostic_proposal_models import PreparationMetadata
from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticError
from memcontam.readiness.phase13_shared_provider_budget import SharedProviderBudget

from . import phase13_proposal_fixtures as fixtures

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_phase13_live_diagnostic.py"
CASES = (
    ("query", "DIAGNOSTIC_PREPARATION_DC_PAYLOAD_MISMATCH"),
    ("response", "DIAGNOSTIC_PREPARATION_DC_PAYLOAD_MISMATCH"),
    ("serialized_hash", "DIAGNOSTIC_PROPOSAL_DC_BYTES_MISMATCH"),
    ("candidate_id", "DIAGNOSTIC_PREPARATION_DC_PAYLOAD_MISMATCH"),
    ("canonical_signature", "DIAGNOSTIC_PROPOSAL_INPUT_SIGNATURE_MISMATCH"),
    ("inventory_index", "DIAGNOSTIC_PROPOSAL_INPUT_INVENTORY_MISMATCH"),
    ("dimension", "DIAGNOSTIC_PROPOSAL_EMBEDDING_CONTRACT_MISMATCH"),
    ("embedding", "DIAGNOSTIC_PROPOSAL_EMBEDDING_CONTRACT_MISMATCH"),
    ("normalize", "DIAGNOSTIC_PROPOSAL_EMBEDDING_CONTRACT_MISMATCH"),
)


def contradicting_fixture(root, monkeypatch, mutation):
    original_save = fixtures.save

    def save(path, value):
        if path.name == "synthetic-proposal.json":
            render = next(row["render"] for row in value["ordinary_units"]
                          if row["task"] == "math_equation_balancer" and row["baseline"] == "dc_rs"
                          and row["arm"] == "correct")
            match mutation:
                case "query" | "response":
                    render[mutation] = "INDEPENDENT_SYNTHETIC_FOREIGN_" + mutation.upper()
                    render["query_sha256"] = hashlib.sha256(render["query"].encode()).hexdigest()
                    render["response_sha256"] = hashlib.sha256(render["response"].encode()).hexdigest()
                    content = json.dumps({"input": render["query"], "raw_output": render["response"]},
                                         sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                    render["serialized_content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
                case "serialized_hash":
                    render["serialized_content_sha256"] = "0" * 64
                case "candidate_id":
                    render["candidate_id"] = "foreign-candidate"
                case "canonical_signature":
                    value["inputs"][0]["canonical_signature"] = "0" * 64
                case "inventory_index":
                    value["inputs"][0]["inventory_index"] = 16
                case "dimension":
                    value["resources"]["dimension"] = 1
                case "embedding":
                    value["resources"]["embedding"] = "foreign-embedding"
                case "normalize":
                    value["resources"]["normalize"] = False
        return original_save(path, value)

    monkeypatch.setattr(fixtures, "save", save)
    return fixtures.metadata_fixture(root, monkeypatch)


@pytest.mark.parametrize("mutation,code", CASES)
def test_preparation_refuses_contradiction_before_returning_subject(tmp_path, monkeypatch, mutation, code):
    proposal, metadata, _, _, _ = contradicting_fixture(tmp_path, monkeypatch, mutation)

    with pytest.raises(LiveDiagnosticError, match=code):
        prepare_proposal(proposal, PreparationMetadata.model_validate_json(metadata.read_bytes()))


@pytest.mark.parametrize("mutation,code", CASES)
def test_signed_cli_refuses_contradiction_before_publication(tmp_path, monkeypatch, capsys, mutation, code):
    proposal, metadata, eligibility, subject, _ = contradicting_fixture(tmp_path, monkeypatch, mutation)

    def forbidden(*args, **kwargs):
        pytest.fail("metadata intake reached forbidden execution seam")

    monkeypatch.setattr(httpx, "Client", forbidden)
    monkeypatch.setattr(httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(cli.LiveDiagnosticLedger, "create", forbidden)
    monkeypatch.setattr(SharedProviderBudget, "create", forbidden)
    monkeypatch.setattr(cli.phase13_diagnostic_cli_native, "load_embedding", forbidden)
    monkeypatch.setattr(cli.phase13_diagnostic_cli_native, "build_native", forbidden)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "register", "--scope", str(proposal),
        "--preparation", str(metadata), "--eligibility", str(eligibility),
        "--budget-binding", subject.context().contract.budget_binding.path,
        "--execution-approval", str(tmp_path / "synthetic-owner.json"), "--output-dir", str(tmp_path)])

    result = runpy.run_path(str(SCRIPT))["main"]()

    assert result == 2
    assert json.loads(capsys.readouterr().out)["code"] == code
    assert not Path(subject.registration_path).exists()
    assert not Path(subject.binding.ledger_path).exists()


@pytest.mark.parametrize("mutation,code", [
    ("missing_entry", "DIAGNOSTIC_PREPARATION_DC_PAYLOAD_FACT_REQUIRED"),
    ("archive_bytes", "DIAGNOSTIC_PREPARATION_DC_ARCHIVE_MISMATCH"),
    ("entry_render", "DIAGNOSTIC_PREPARATION_DC_PAYLOAD_MISMATCH"),
    ("triplet", "DIAGNOSTIC_PREPARATION_DC_PAYLOAD_MISMATCH"),
])
def test_payload_join_refuses_missing_or_contradicting_native_facts(tmp_path, monkeypatch, mutation, code):
    from memcontam.readiness.phase13_diagnostic_cli_models import NativeInput
    from memcontam.readiness.phase13_diagnostic_proposal import validate_dc_payload
    from memcontam.readiness.phase13_diagnostic_proposal_models import Task11Proposal
    from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticScope

    proposal_path, _, _, subject, _ = fixtures.metadata_fixture(tmp_path, monkeypatch)
    proposal = Task11Proposal.model_validate_json(proposal_path.read_bytes())
    unit = next(row for row in proposal.ordinary_units
                if (row.task, row.baseline, row.arm) == ("math_equation_balancer", "dc_rs", "correct"))
    scope = next(scope for scope in subject.scopes
                 if (native := NativeInput.model_validate_json(scope.input_utf8)).task_name == unit.task
                 and (native.baseline, native.arm) == (unit.baseline, unit.arm))
    raw = scope.model_dump(mode="json")
    state = json.loads(scope.state_utf8)
    native_raw = json.loads(scope.input_utf8)
    match mutation:
        case "missing_entry":
            state["entries"] = []
        case "archive_bytes":
            state["native_state"]["archive"][0]["metadata"]["generated_output"] = "foreign-response"
        case "entry_render":
            state["entries"][0]["render_id"] = "foreign-render"
        case "triplet":
            native_raw["branch"]["candidate_triplet_id"] = "foreign-triplet"
    raw["state_utf8"] = json.dumps(state)
    altered = LiveDiagnosticScope.model_validate_json(json.dumps(raw))
    native = NativeInput.model_validate_json(json.dumps(native_raw))

    with pytest.raises(LiveDiagnosticError, match=code):
        validate_dc_payload(unit, native, altered)
