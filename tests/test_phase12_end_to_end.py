from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memcontam.experiment.phase12 import cli as phase12_cli
from memcontam.experiment.phase12.contracts import FidelityCertificate
from memcontam.logging.writer_v3 import Phase12RunWriter
from memcontam.readiness import phase12_certificate
from memcontam.readiness.phase12_replay import (
    P12IReplayResult,
    P12IReplaySpec,
    run_p12i_replay,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12"
READINESS_CONFIG = ROOT / "configs" / "phase12" / "readiness.yaml"
_GATE_REASON_CODES = {
    "prefix_checkpoint": "P12I_PREFIX_CHECKPOINT_GATE_FAILED",
    "five_arm_branch": "P12I_FIVE_ARM_BRANCH_GATE_FAILED",
    "nomem_alias": "P12I_NOMEM_ALIAS_GATE_FAILED",
    "filter_information_boundary": "P12I_FILTER_INFORMATION_BOUNDARY_GATE_FAILED",
    "logging_v3_join": "P12I_LOGGING_V3_JOIN_GATE_FAILED",
    "model_behavior_denominator": "P12I_MODEL_BEHAVIOR_DENOMINATOR_GATE_FAILED",
    "eligibility_recomputation": "P12I_ELIGIBILITY_RECOMPUTATION_GATE_FAILED",
    "run_archive_reconstruction": "P12I_RUN_ARCHIVE_RECONSTRUCTION_GATE_FAILED",
}


@dataclass(frozen=True)
class EndToEndResult:
    prefix_dir: Path
    branch_dir: Path
    plan: dict[str, Any]
    aggregate: dict[str, Any]
    archive: dict[str, Any]
    replay: P12IReplayResult
    scientific_admission: bool


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _bfv2_certificate() -> FidelityCertificate:
    p12i = _fixture("FX-P12I-001.json")
    return FidelityCertificate(
        certificate_id="baseline-fidelity-v2-f1c-blocked",
        protocol_version="baseline_fidelity_v2",
        git_commit=str(p12i["repository_commit"]),
        overall_status="blocked",
        issued_at="2026-07-23T00:00:00Z",
    )


def run_complete_replay_fixture(tmp_path: Path) -> EndToEndResult:
    run_root = tmp_path / "runs"
    phase12_cli._validate_config(READINESS_CONFIG)
    plan = phase12_cli._plan(FIXTURE_ROOT / "FX-CONFIG-001.json")
    prefix = phase12_cli._run_prefix(
        Namespace(
            fixture_root=FIXTURE_ROOT,
            replay="FX-BRANCH-001",
            run_root=run_root,
            run_id="prefix",
        )
    )
    branch = phase12_cli._run_branch(
        Namespace(
            fixture_root=FIXTURE_ROOT,
            replay="FX-BRANCH-001",
            run_root=run_root,
            run_id="branch",
        )
    )
    branch_dir = Path(str(branch["run_dir"]))
    aggregate = phase12_cli._aggregate(Namespace(replay=None, run_dir=branch_dir))
    archive = phase12_cli._validate_archive(Namespace(replay=None, run_dir=branch_dir))
    replay = run_p12i_replay(P12IReplaySpec(FIXTURE_ROOT), tmp_path / "p12i")
    certificate = phase12_certificate.issue_p12i(replay, _bfv2_certificate())
    loaded = phase12_certificate.load_p12i(phase12_certificate.serialize_p12i(certificate))
    validation = phase12_certificate.validate_p12i(
        loaded, phase12_certificate.artifacts_for(replay, _bfv2_certificate())
    )
    return EndToEndResult(
        prefix_dir=Path(str(prefix["run_dir"])),
        branch_dir=branch_dir,
        plan=plan,
        aggregate=aggregate,
        archive=archive,
        replay=replay,
        scientific_admission=validation.scientific_admission,
    )


def test_complete_non_scientific_replay_archive(tmp_path: Path) -> None:
    e2e = _fixture("FX-E2E-001.json")
    result = run_complete_replay_fixture(tmp_path / "first")
    evidence_bytes = {
        gate.gate_id: gate.evidence_path.read_bytes() for gate in result.replay.subgates
    }
    reconstructed = run_p12i_replay(P12IReplaySpec(FIXTURE_ROOT), tmp_path / "first" / "p12i")

    assert result.plan == {
        "candidate_routes": ["3w", "5w"],
        "registry_ids": ["run-templates-3w-64bcbfd5a148", "run-templates-5w-343ccedd227e"],
        "scientific_result": False,
    }
    assert result.aggregate["trial_count"] > 0
    assert result.archive["archive_valid"] is e2e["expected"]["archive_valid"]
    assert result.replay.overall_status == "pass"
    assert result.replay.scientific_result is False
    assert result.scientific_admission is e2e["expected"]["scientific_admission"]
    assert (
        len(Phase12RunWriter.read_jsonl(result.prefix_dir, "checkpoint_events.jsonl"))
        == e2e["expected"]["prefix_memory_event_count"]
    )

    expected_files = e2e["expected"]["artifact_files"]
    for filename in expected_files:
        assert (result.branch_dir / filename).is_file()

    manifest = json.loads((result.branch_dir / "public_artifact_manifest.json").read_text())
    assert (
        sum(filename.endswith(".jsonl") for filename in manifest["artifacts"])
        == e2e["expected"]["public_stream_count"]
    )
    assert (
        len(Phase12RunWriter.read_jsonl(result.branch_dir / "audit", "audit_labels.jsonl"))
        == e2e["expected"]["audit_stream_count"]
    )
    assert [gate.gate_id for gate in result.replay.subgates] == list(_GATE_REASON_CODES)
    assert reconstructed == result.replay
    assert all(
        gate.evidence_hash == hashlib.sha256(gate.evidence_path.read_bytes()).hexdigest()
        and gate.evidence_path.read_bytes() == evidence_bytes[gate.gate_id]
        for gate in result.replay.subgates
    )


def test_every_registered_evidence_mutation_fails_closed(tmp_path: Path) -> None:
    result = run_complete_replay_fixture(tmp_path)

    for gate in result.replay.subgates:
        original = gate.evidence_path.read_bytes()
        gate.evidence_path.write_bytes(b'{"mutated":true}')
        failed = run_p12i_replay(P12IReplaySpec(FIXTURE_ROOT), tmp_path / "p12i")

        assert failed.overall_status == "fail"
        assert failed.reason_code == _GATE_REASON_CODES[gate.gate_id]
        assert (
            next(item for item in failed.subgates if item.gate_id == gate.gate_id).reason_code
            == (_GATE_REASON_CODES[gate.gate_id])
        )
        gate.evidence_path.write_bytes(original)
