from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12"
GATE_REASON_CODES = {
    "prefix_checkpoint": "P12I_PREFIX_CHECKPOINT_GATE_FAILED",
    "five_arm_branch": "P12I_FIVE_ARM_BRANCH_GATE_FAILED",
    "nomem_alias": "P12I_NOMEM_ALIAS_GATE_FAILED",
    "filter_information_boundary": "P12I_FILTER_INFORMATION_BOUNDARY_GATE_FAILED",
    "logging_v3_join": "P12I_LOGGING_V3_JOIN_GATE_FAILED",
    "model_behavior_denominator": "P12I_MODEL_BEHAVIOR_DENOMINATOR_GATE_FAILED",
    "eligibility_recomputation": "P12I_ELIGIBILITY_RECOMPUTATION_GATE_FAILED",
    "run_archive_reconstruction": "P12I_RUN_ARCHIVE_RECONSTRUCTION_GATE_FAILED",
}


def _replay_module():
    return importlib.import_module("memcontam.readiness.phase12_replay")


def test_produces_passing_replay_without_certificate(tmp_path: Path) -> None:
    replay = _replay_module()

    result = replay.run_p12i_replay(replay.P12IReplaySpec(FIXTURE_ROOT), tmp_path)

    assert result.overall_status == "pass"
    assert result.reason_code is None
    assert result.scientific_result is False
    assert [gate.gate_id for gate in result.subgates] == list(GATE_REASON_CODES)
    assert all(gate.status == "pass" and gate.reason_code is None for gate in result.subgates)
    assert all(
        gate.evidence_hash == hashlib.sha256(gate.evidence_path.read_bytes()).hexdigest()
        for gate in result.subgates
    )
    assert not list(tmp_path.rglob("*certificate*"))
    fixture = json.loads((FIXTURE_ROOT / "FX-P12I-001.json").read_text(encoding="utf-8"))
    assert fixture["bfv2"]["f1c"] == "blocked"


def test_failed_subgate_blocks_replay(tmp_path: Path) -> None:
    replay = _replay_module()
    spec = replay.P12IReplaySpec(FIXTURE_ROOT)
    passing = replay.run_p12i_replay(spec, tmp_path)

    for gate in passing.subgates:
        original = gate.evidence_path.read_bytes()
        gate.evidence_path.write_text('{"mutated":true}', encoding="utf-8")

        failed = replay.run_p12i_replay(spec, tmp_path)

        assert failed.overall_status == "fail"
        assert failed.reason_code == GATE_REASON_CODES[gate.gate_id]
        failed_gate = next(item for item in failed.subgates if item.gate_id == gate.gate_id)
        assert failed_gate.status == "fail"
        assert failed_gate.reason_code == GATE_REASON_CODES[gate.gate_id]
        gate.evidence_path.write_bytes(original)
