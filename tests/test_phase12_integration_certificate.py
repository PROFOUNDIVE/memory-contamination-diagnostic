from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

from memcontam.experiment.phase12.contracts import FidelityCertificate


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12"
EXPECTED_COMMIT = "830b89c8c169ffa9cdea472887fdae134dbae7cf"


def _replay_module():
    return importlib.import_module("memcontam.readiness.phase12_replay")


def _certificate_module():
    return importlib.import_module("memcontam.readiness.phase12_certificate")


def _bfv2() -> FidelityCertificate:
    payload = json.loads((FIXTURE_ROOT / "FX-P12I-001.json").read_text(encoding="utf-8"))
    return FidelityCertificate(
        certificate_id="baseline-fidelity-v2-f1c-blocked",
        protocol_version="baseline_fidelity_v2",
        git_commit=payload["repository_commit"],
        overall_status="blocked",
        issued_at="2026-07-23T00:00:00Z",
    )


def test_issues_hash_linked_p12i_with_truthful_blocked_f1c(tmp_path: Path) -> None:
    replay = _replay_module()
    certificate = _certificate_module()
    result = replay.run_p12i_replay(replay.P12IReplaySpec(FIXTURE_ROOT), tmp_path)
    bfv2 = _bfv2()

    issued = certificate.issue_p12i(result, bfv2)
    encoded = certificate.serialize_p12i(issued)
    loaded = certificate.load_p12i(encoded)
    report = certificate.validate_p12i(loaded, certificate.artifacts_for(result, bfv2))

    assert issued.overall_status == "pass"
    assert loaded == issued
    assert report.valid is True
    assert report.scientific_admission is False


def test_rejects_failed_stale_foreign_or_falsified_inputs(tmp_path: Path) -> None:
    replay = _replay_module()
    certificate = _certificate_module()
    spec = replay.P12IReplaySpec(FIXTURE_ROOT)
    result = replay.run_p12i_replay(spec, tmp_path)
    bfv2 = _bfv2()
    issued = certificate.issue_p12i(result, bfv2)

    cases = (
        ("resolved_config_hash", "stale", "P12I_RESOLVED_CONFIG_HASH_MISMATCH"),
        ("repository_commit", "foreign", "P12I_REPOSITORY_COMMIT_MISMATCH"),
        ("f1c_contract_refs", {"verifier_blob_sha": "stale"}, "P12I_F1C_CONTRACT_HASH_MISMATCH"),
        ("f1c_status", "pass", "P12I_F1C_STATUS_MISMATCH"),
    )
    for field, value, reason in cases:
        artifacts = certificate.artifacts_for(result, bfv2)
        if field == "f1c_contract_refs":
            refs = dict(artifacts[field])
            refs["verifier_blob_sha"] = "stale"
            artifacts[field] = refs
        else:
            artifacts[field] = value
        with pytest.raises(certificate.CertificateError, match=reason):
            certificate.validate_p12i(issued, artifacts)

    failed_result = copy.copy(result)
    failed_result = failed_result.__class__(
        replay_id=result.replay_id,
        scientific_result=False,
        subgates=tuple(
            gate.__class__(
                gate.gate_id, "fail", "P12I_GATE_FAILED", gate.evidence_path, gate.evidence_hash
            )
            for gate in result.subgates
        ),
        overall_status="fail",
        reason_code="P12I_PREFIX_CHECKPOINT_GATE_FAILED",
        archive_run_dir=result.archive_run_dir,
    )
    with pytest.raises(certificate.CertificateError, match="P12I_REPLAY_FAILED"):
        certificate.issue_p12i(failed_result, bfv2)
