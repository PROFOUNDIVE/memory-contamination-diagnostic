from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from memcontam.experiment import phase13_ordinary_runtime
from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_dispatch
from memcontam.readiness.phase13_main_live_dispatch import persist_unit_dispatch
from memcontam.readiness.phase13_main_runner_ledger import MainRunLedger

from .phase13_snapshot_durable_fixtures import native_snapshot
from .phase13_v3_fixtures import SyntheticFixture


def test_packet_implementation_is_available():
    assert importlib.util.find_spec("memcontam.readiness.phase13_snapshot_packet") is not None


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture(scope="module")
def native_inputs(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("packet-native")
    reopens = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
        for baseline in ("bot_style", "dc_rs"):
            target = root / baseline
            target.mkdir()
            snapshot = native_snapshot(target, baseline)
            unit, output = snapshot_dispatch(snapshot)
            binding = replace(SyntheticFixture().binding(), core_authorization_gate_krw=0)
            ledger = MainRunLedger.create(target / "ledger.sqlite3", binding, (unit,))
            assert ledger.claim_dispatch(unit.unit_id, 0, 0)
            receipt = persist_unit_dispatch(target, unit, output)
            ledger.persist_completed(unit.unit_id, receipt)
            descriptor = target / "reopen.json"
            descriptor.write_text(json.dumps({"unit": asdict(unit), "binding": asdict(binding)}))
            reopens.append(pin(descriptor))
    return reopens


@pytest.fixture
def proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_inputs):
    import memcontam.readiness.phase13_snapshot_packet as packet
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    authority = inputs / "MANIFEST.json"
    authority.write_text(json.dumps({"as_of": "fixture", "current_freshness": "unknown",
        "source_generation": "unknown", "enforced_immutability": False, "entries": []}))
    financial = inputs / "financial.md"
    financial.write_text("synthetic financial boundary, not launch authority")
    monkeypatch.setattr(packet, "AUTHORITY_SHA256", pin(authority)["sha256"])
    monkeypatch.setattr(packet, "FINANCIAL_SHA256", pin(financial)["sha256"])
    pending = inputs / "pending.json"
    pending.write_text(json.dumps({"gates": {"human_panel": {"status": "UNVERIFIED",
        "authentic_rating_rows": 0}, "owner_off_registry_fact": {
        "status": "PENDING", "owner_statement_received": False}},
        "decisions": {"scientific_adoption": False, "paid_launch": False},
        "financial_boundary": {"combined_incremental_provider_budget": 45}}))
    suite = inputs / "green.xml"
    suite.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>')
    claim = inputs / "DoneClaim.json"
    claim.write_text(json.dumps({"task_complete": True, "artifacts_sha256": {
        str(suite): pin(suite)["sha256"]}}))
    monkeypatch.setattr(packet, "MECHANICAL_CLAIM_SHA256", {
        name: pin(claim)["sha256"] for name in ("accounting", "durable", "semantics")})
    raw = {"authority_manifest": pin(authority), "financial_authorization": pin(financial),
        "pending": pin(pending), "handoff": pin(financial),
        "mechanics": [{"name": name, "claim": pin(claim), "suite": pin(suite)}
            for name in ("accounting", "durable", "semantics")],
        "qa_reopens": native_inputs}
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(raw))
    return path


def test_create_and_read_only_reopen_when_mechanics_pass(proposal: Path, tmp_path: Path, native_inputs):
    import memcontam.readiness.phase13_snapshot_packet as packet
    output = tmp_path / "packet"
    packet.prepare_packet(proposal, output)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    ledgers = [Path(row["path"]).parent / "ledger.sqlite3" for row in native_inputs]
    ledger_before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in ledgers]
    result = packet.verify_packet(output)
    assert result["status"] == "DIAGNOSTIC_READY_FOR_SEPARATE_AUTHORIZATION"
    assert result["launch_eligible"] is False
    assert result["paid_launch"] is False
    assert result["real_spend_cap_usd"] == 0
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    assert ledger_before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in ledgers]
    manifest = json.loads((output / "input-manifest.json").read_text())
    assert {row["baseline"] for row in manifest["qa_native_objects"]} == {"bot_style", "dc_rs"}
    assert all(row["state_sha256"] and row["input_sha256"] for row in manifest["qa_native_objects"])
    assert json.loads((output / "registration-proposal.json").read_text())["scientific_schedule"] is None


@pytest.mark.parametrize("filename", ["proposal.json", "readiness.json", "input-manifest.json", "SHA256SUMS"])
def test_refuse_tampered_packet_when_reopened(proposal: Path, tmp_path: Path, filename: str):
    import memcontam.readiness.phase13_snapshot_packet as packet
    output = tmp_path / "packet"
    packet.prepare_packet(proposal, output)
    with (output / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(packet.PacketError):
        packet.verify_packet(output)


def test_refuse_resealed_readiness_when_authority_is_forged(proposal: Path, tmp_path: Path):
    import memcontam.readiness.phase13_snapshot_packet as packet
    output = tmp_path / "packet"
    packet.prepare_packet(proposal, output)
    path = output / "readiness.json"
    raw = json.loads(path.read_text())
    raw["launch_eligible"] = True
    path.write_text(json.dumps(raw))
    (output / "SHA256SUMS").write_bytes(packet.checksums(output))
    with pytest.raises(packet.PacketError):
        packet.verify_packet(output)


@pytest.mark.parametrize("kind", ["collision", "symlink", "parent_alias", "nested"])
def test_refuse_unsafe_output_before_writes(proposal: Path, tmp_path: Path, kind: str):
    import memcontam.readiness.phase13_snapshot_packet as packet
    output = tmp_path / "packet"
    if kind == "collision":
        output.mkdir()
    if kind == "symlink":
        output.symlink_to(tmp_path / "absent", target_is_directory=True)
    if kind == "parent_alias":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        output = alias / "packet"
    if kind == "nested":
        output = proposal.parent / "inputs/packet"
    with pytest.raises((packet.PacketError, FileExistsError)):
        packet.prepare_packet(proposal, output)


@pytest.mark.parametrize("field", ["owner_fact", "human_panel", "dc_family", "scientific_schedule",
    "run_id", "stage_limits", "count_limits", "attempt_retry_limits", "price_risk_acceptance",
    "launch_instruction", "scientific_adoption", "paid_launch", "real_spend_cap_usd"])
def test_reject_authorizing_proposal_fields_when_supplied(proposal: Path, tmp_path: Path, field: str):
    import memcontam.readiness.phase13_snapshot_packet as packet
    raw = json.loads(proposal.read_text())
    raw[field] = True
    proposal.write_text(json.dumps(raw))
    with pytest.raises((packet.PacketError, ValueError)):
        packet.prepare_packet(proposal, tmp_path / "packet")


def test_refuse_changed_dependency_even_when_outer_packet_is_intact(proposal: Path, tmp_path: Path):
    import memcontam.readiness.phase13_snapshot_packet as packet
    output = tmp_path / "packet"
    packet.prepare_packet(proposal, output)
    (proposal.parent / "inputs/green.xml").write_text("corrupt")
    with pytest.raises(packet.PacketError):
        packet.verify_packet(output)


@pytest.mark.parametrize("gate", ["human_panel", "owner_off_registry_fact"])
def test_reject_forged_external_fact_when_pending_inventory_is_rebound(proposal: Path, tmp_path: Path, gate: str):
    import memcontam.readiness.phase13_snapshot_packet as packet
    raw = json.loads(proposal.read_text())
    pending_path = Path(raw["pending"]["path"])
    pending = json.loads(pending_path.read_text())
    pending["gates"][gate]["status"] = "PASS"
    pending_path.write_text(json.dumps(pending))
    raw["pending"] = pin(pending_path)
    proposal.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        packet.prepare_packet(proposal, tmp_path / "packet")


def test_reject_forged_qa_receipt_when_rebound(proposal: Path, tmp_path: Path):
    import memcontam.readiness.phase13_snapshot_packet as packet
    raw = json.loads(proposal.read_text())
    claim = Path(raw["mechanics"][0]["claim"]["path"])
    claim.write_text(json.dumps({"task_complete": True, "artifacts_sha256": {}}))
    raw["mechanics"][0]["claim"] = pin(claim)
    proposal.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        packet.prepare_packet(proposal, tmp_path / "packet")


def test_preparation_never_constructs_clients(proposal: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import memcontam.readiness.phase13_snapshot_packet as packet
    from memcontam.clients.openai_compatible import OpenAICompatibleClient
    from memcontam.clients.openai_responses import OpenAIResponsesClient

    def deny(*args, **kwargs):
        pytest.fail("provider client constructed")

    monkeypatch.setattr(OpenAICompatibleClient, "__init__", deny)
    monkeypatch.setattr(OpenAIResponsesClient, "__init__", deny)
    result = packet.prepare_packet(proposal, tmp_path / "packet")
    assert result["launch_eligible"] is False


@pytest.mark.parametrize("flag", ["--launch", "--live", "--allow-live-calls"])
def test_cli_refuses_live_options_when_invoked(tmp_path: Path, flag: str):
    script = tmp_path / "cli-negative.py"
    entry = Path(__file__).resolve().parents[1] / "scripts/prepare_phase13_snapshot_diagnostic.py"
    script.write_text(f"import runpy, sys\nsys.argv = [{str(entry)!r}, {flag!r}]\nrunpy.run_path({str(entry)!r}, run_name='__main__')\n")
    result = subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline", str(script)],
        env=os.environ, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


@pytest.fixture
def successor_proposal() -> Path:
    root = Path(os.environ.get("RECOVERY_EVIDENCE", "/home/hyunwoo/phase13_artifacts/phase13-parallel-recovery-20261021-001"))
    return root / "d2/proposal-f2checkpoint-001.json"


def test_actual_successor_claims_create_read_only_packet(successor_proposal: Path, tmp_path: Path):
    import memcontam.readiness.phase13_snapshot_packet as packet
    output = tmp_path / "packet"
    packet.prepare_packet(successor_proposal, output)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    result = packet.verify_packet(output)
    assert result["launch_eligible"] is False and result["real_spend_cap_usd"] == 0
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    assert len(json.loads((output / "input-manifest.json").read_text())["qa_native_objects"]) == 5


@pytest.mark.parametrize("mutation", ["changed", "resealed", "unrecognized", "historical"])
def test_successor_claim_trust_refuses_substitution(successor_proposal: Path, tmp_path: Path, mutation: str):
    import memcontam.readiness.phase13_snapshot_packet as packet
    raw = json.loads(successor_proposal.read_text())
    receipt = next(row for row in raw["mechanics"] if row["name"] == "durable")
    claim = tmp_path / "claim.json"
    original = Path(receipt["claim"]["path"]).read_bytes()
    claim.write_bytes(original + b" ")
    receipt["claim"]["path"] = str(claim)
    reason = "PACKET_BINDING_MISMATCH"
    if mutation != "changed":
        reason = "PACKET_QA_CLAIM_PIN_INVALID"
        if mutation == "unrecognized":
            claim.write_text(json.dumps({"task_complete": True, "artifacts_sha256": {}}))
        if mutation == "historical":
            claim.write_bytes((successor_proposal.parents[1] / "task-14/DoneClaim.json").read_bytes())
        receipt["claim"] = pin(claim)
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(raw))
    with pytest.raises(packet.PacketError, match=reason):
        packet.prepare_packet(proposal_path, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()
