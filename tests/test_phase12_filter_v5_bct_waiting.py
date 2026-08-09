from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge import bct_live
from memcontam.experiment.phase12.filter_challenge import registry_calibration
from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / ".omo" / "plans" / "phase12-post-filter-v5-calibration-readiness.md"
BUNDLE = ROOT / "docs" / "evidence" / "phase12-filter-v5-bct-v1"
LIVE_ROOT = ROOT / "runs" / "phase12-filter-v5-bct-live-v1"
CONFIG = ROOT / "configs" / "phase12" / "filter_v5_bct_calibration.yaml"
BCT_REQUEST = ROOT / "data" / "phase12" / "filter_v5_bct_v1" / "bct_authorization_request.json"
EVIDENCE_SCRIPT = ROOT / "scripts" / "build_phase12_filter_v5_bct_evidence.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_phase12_filter_v5_bct_evidence.py"
EXPERIMENT_PACKAGE = ROOT / "src" / "memcontam" / "experiment" / "__init__.py"
PYRIGHT_CONFIG = ROOT / "pyrightconfig.json"


def _verify(bundle: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--through",
            "bct",
            "--bundle",
            str(bundle),
            "--plan",
            str(PLAN),
            "--artifact-root",
            str(LIVE_ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_bct_waiting_evidence_has_a_regular_experiment_package_boundary() -> None:
    assert EXPERIMENT_PACKAGE.is_file()


def test_pyrightconfig_resolves_the_src_package_root() -> None:
    settings = json.loads(PYRIGHT_CONFIG.read_text(encoding="utf-8"))
    assert settings["include"] == ["src", "scripts", "tests"]
    assert settings["executionEnvironments"] == [
        {"root": "src", "extraPaths": ["src"]},
        {"root": "scripts", "extraPaths": ["src"]},
        {"root": "tests", "extraPaths": ["src"]},
    ]


def _upstream_bundle(path: Path) -> Path:
    shutil.copytree(BUNDLE, path)
    for name in ("bct_execution", "archive_validation", "claim_scope"):
        (path / f"{name}_report.json").unlink(missing_ok=True)
    return path


def _waiting_bundle(tmp_path: Path) -> tuple[Path, Path]:
    stage_result = tmp_path / "bct-stage-result.json"
    CalibrationStageResult.waiting("bct", "AWAITING_SCREENING_AUTHORIZATION").write_atomic(
        stage_result
    )
    bundle = _upstream_bundle(tmp_path / "bundle")
    built = subprocess.run(
        [
            sys.executable,
            str(EVIDENCE_SCRIPT),
            "--report-set",
            "bct",
            "--bundle",
            str(bundle),
            "--plan",
            str(PLAN),
            "--artifact-root",
            str(LIVE_ROOT),
            "--authorization-request",
            str(BCT_REQUEST),
            "--stage-result",
            str(stage_result),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    return bundle, stage_result


def test_bct_waiting_branch_uses_raw_screening_terminal_and_seals_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: report 5 binds the raw Task-4 missing-screening-authorization stage.
    stage_result = tmp_path / "bct-stage-result.json"
    arguments = argparse.Namespace(
        config=CONFIG,
        artifact_root=LIVE_ROOT,
        run_id="filter-v5-bct-v1-attempt-001",
        stage_result=stage_result,
        authorization=None,
        expected_authorization_sha256=None,
        authorization_request=tmp_path / "not-created-bct-authorization-request.json",
    )
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", LIVE_ROOT)
    monkeypatch.setattr(bct_live, "_build_live_factory", pytest.fail)

    # When: the BCT CLI stage is reached without authorization.
    result = bct_live._run_cli_stage(arguments, "bct")

    # Then: the earlier raw screening terminal prevents client construction and BCT attempts.
    assert result.stage == "bct"
    assert result.disposition == "blocked_before_stage"
    assert result.terminal_status == "AWAITING_SCREENING_AUTHORIZATION"
    assert result.reason_code == "AWAITING_SCREENING_AUTHORIZATION"
    assert result.provider_calls_issued == 0
    assert not LIVE_ROOT.exists()

    bundle = _upstream_bundle(tmp_path / "bundle")
    built = subprocess.run(
        [
            sys.executable,
            str(EVIDENCE_SCRIPT),
            "--report-set",
            "bct",
            "--bundle",
            str(bundle),
            "--plan",
            str(PLAN),
            "--artifact-root",
            str(LIVE_ROOT),
            "--authorization-request",
            str(BCT_REQUEST),
            "--stage-result",
            str(stage_result),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr

    upstream = {
        report_id: hashlib.sha256((bundle / f"{report_id.replace('-', '_')}_report.json").read_bytes()).hexdigest()
        for report_id in (
            "authority-transition",
            "methods-lock",
            "freeze-a",
            "screening",
            "freeze-b-search-config",
        )
    }
    for report_id in ("bct_execution", "archive_validation", "claim_scope"):
        report = json.loads((bundle / f"{report_id}_report.json").read_text(encoding="utf-8"))
        assert report["terminal_status"] == "AWAITING_SCREENING_AUTHORIZATION"
        assert report["stage_disposition"] == "blocked_before_stage"
        assert report["provider_calls_issued"] == 0
        assert report["upstream_report_sha256"] == upstream
        assert report["input_digests"]["freeze_b"] is None
        assert report["input_digests"]["search_config"] is None
        assert report["input_digests"]["bct_authorization_request"] is None

    approved = _verify(bundle)
    assert approved.returncode == 0, approved.stdout + approved.stderr
    assert approved.stdout == "APPROVE\n"

    report_path = bundle / "bct_execution_report.json"
    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["provider_calls_issued"] = 1
    report_path.write_text(json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    rejected = _verify(bundle)
    assert rejected.returncode != 0
    assert rejected.stdout == "EVIDENCE_REPORT_CONTRACT_INVALID\n"


def test_bct_waiting_verifier_rejects_stage_terminal_and_hash_tampering(tmp_path: Path) -> None:
    bundle, stage_result = _waiting_bundle(tmp_path / "stage")
    stage = json.loads(stage_result.read_text(encoding="utf-8"))
    stage["terminal_status"] = "AWAITING_BCT_AUTHORIZATION"
    stage_result.write_text(json.dumps(stage, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    assert _verify(bundle).stdout == "EVIDENCE_STAGE_DIGEST_MISMATCH\n"

    bundle, _ = _waiting_bundle(tmp_path / "terminal")
    report_path = bundle / "claim_scope_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["terminal_status"] = "AWAITING_BCT_AUTHORIZATION"
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    assert _verify(bundle).stdout == "EVIDENCE_REPORT_CONTRACT_INVALID\n"


def test_bct_waiting_branch_rejects_an_existing_live_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "runs" / "phase12-filter-v5-bct-live-v1"
    artifact_root.mkdir(parents=True)
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", artifact_root)
    arguments = argparse.Namespace(
        config=CONFIG,
        artifact_root=artifact_root,
        run_id="filter-v5-bct-v1-attempt-001",
        stage_result=tmp_path / "bct-stage-result.json",
        authorization=None,
        expected_authorization_sha256=None,
        authorization_request=tmp_path / "not-created-bct-authorization-request.json",
    )
    monkeypatch.setattr(bct_live, "_build_live_factory", pytest.fail)

    result = bct_live._run_cli_stage(arguments, "bct")

    assert result.disposition == "blocked_before_stage"
    assert result.terminal_status == "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE"
    assert result.reason_code == "LIVE_ARTIFACT_ROOT_EXISTS"
    assert result.provider_calls_issued == 0


def test_bct_waiting_verifier_rejects_malformed_report_bytes(tmp_path: Path) -> None:
    bundle, _ = _waiting_bundle(tmp_path)
    (bundle / "claim_scope_report.json").write_text("{\n", encoding="utf-8")

    result = _verify(bundle)

    assert result.returncode != 0
    assert result.stdout == "EVIDENCE_REPORT_INVALID\n"

    bundle, _ = _waiting_bundle(tmp_path / "hash")
    report_path = bundle / "archive_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["upstream_report_sha256"]["screening"] = "0" * 64
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    assert _verify(bundle).stdout == "EVIDENCE_REPORT_CONTRACT_INVALID\n"
