from __future__ import annotations

import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.bct_archive import validate_evidence_bundle
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    approval_descriptor_path,
    approved_plan_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs" / "evidence" / "phase12-filter-v5-bct-v1"
PLAN = ROOT / ".omo" / "plans" / "phase12-post-filter-v5-calibration-readiness.md"
VERIFY = ROOT / "scripts" / "verify_phase12_filter_v5_bct_evidence.py"


def _verify(bundle: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "--through",
            "readiness",
            "--bundle",
            str(bundle),
            "--plan",
            str(PLAN),
            "--artifact-root",
            str(ROOT / "runs" / "phase12-filter-v5-bct-live-v1"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _copied_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    return bundle


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _reseal_backward_hashes(bundle: Path) -> None:
    report_ids = (
        "authority-transition",
        "methods-lock",
        "freeze-a",
        "screening",
        "freeze-b-search-config",
        "bct-execution",
        "archive-validation",
        "claim-scope",
    )
    for index, report_id in enumerate(report_ids[4:], start=4):
        path = bundle / f"{report_id.replace('-', '_')}_report.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        upstream_ids = report_ids[:4] if report_id == "freeze-b-search-config" else report_ids[:5]
        payload["upstream_report_sha256"] = {
            prior: sha256((bundle / f"{prior.replace('-', '_')}_report.json").read_bytes()).hexdigest()
            for prior in upstream_ids
        }
        _write_report(path, payload)
    readiness = bundle / "pilot_b_readiness_report.json"
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["prior_report_sha256"] = {
        report_id: sha256((bundle / f"{report_id.replace('-', '_')}_report.json").read_bytes()).hexdigest()
        for report_id in report_ids
    }
    _write_report(readiness, payload)


def test_readiness_verifier_rejects_report_one_provider_counter_reseal(tmp_path: Path) -> None:
    # Given: a copied report chain with a false paid-call counter in report 1.
    bundle = _copied_bundle(tmp_path)
    report = bundle / "authority_transition_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["provider_calls_issued"] = 1
    _write_report(report, payload)
    _reseal_backward_hashes(bundle)

    # When: the final readiness verifier examines the modified chain.
    result = _verify(bundle)

    # Then: a downstream-resealed self-assertion cannot turn a zero-call branch into success.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_REPORT_CONTRACT_INVALID\n"


def test_readiness_verifier_rejects_misleading_report_nine_success_fields(tmp_path: Path) -> None:
    # Given: the raw readiness stage remains waiting but its report claims completion.
    bundle = _copied_bundle(tmp_path)
    report = bundle / "pilot_b_readiness_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["stage_disposition"] = "completed"
    payload["all_passed"] = True
    _write_report(report, payload)

    # When: the final verifier recomputes report 9 from the stage result.
    result = _verify(bundle)

    # Then: undeclared success fields and a mismatched disposition are rejected.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_REPORT_CONTRACT_INVALID\n"


def test_readiness_verifier_rejects_symlinked_report_substitution(tmp_path: Path) -> None:
    # Given: a required report path is replaced by a symlink to an otherwise valid report.
    bundle = _copied_bundle(tmp_path)
    report = bundle / "claim_scope_report.json"
    report.unlink()
    report.symlink_to(bundle / "archive_validation_report.json")

    # When: the final verifier opens every report.
    result = _verify(bundle)

    # Then: descriptor no-follow opening rejects the substitution.
    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_REPORT_INVALID\n"


def test_evidence_validator_rejects_current_source_universe_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "data" / "phase12" / "filter_v5_bct_v1" / "source_universe_v1.json"
    source.parent.mkdir(parents=True)
    source.write_bytes((ROOT / source.relative_to(tmp_path)).read_bytes().replace(b"7a7a39", b"000000", 1))
    monkeypatch.chdir(tmp_path)
    digest = approved_plan_sha256(PLAN, approval_descriptor_path(PLAN))
    result = validate_evidence_bundle(_copied_bundle(tmp_path), digest, "freeze-a")
    assert result.valid is False
    assert result.reason_code == "EVIDENCE_SOURCE_UNIVERSE_INVALID"
