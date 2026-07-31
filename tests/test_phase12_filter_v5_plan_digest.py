from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.bct_archive import build_evidence_report
from memcontam.experiment.phase12.filter_challenge.bct_live import (
    CalibrationAuthorizationError,
    _validate_config,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / ".omo" / "plans" / "phase12-post-filter-v5-calibration-readiness.md"
DESCRIPTOR = ROOT / ".omo" / "approvals" / "phase12-post-filter-v5-calibration-readiness.plan.sha256"
METHODS = ROOT / "docs" / "phase12-filter-v5-bct-methods-lock.md"
CONFIG = ROOT / "configs" / "phase12" / "filter_v5_bct_calibration.yaml"
SCRIPT = ROOT / "scripts" / "validate_phase12_filter_v5_methods_lock.py"
EVIDENCE_SCRIPT = ROOT / "scripts" / "build_phase12_filter_v5_bct_evidence.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_phase12_filter_v5_bct_evidence.py"
FREEZE_A = ROOT / "data" / "phase12" / "filter_v5_bct_v1" / "freeze_a.json"
SCREENING_REQUEST = ROOT / "data" / "phase12" / "filter_v5_bct_v1" / "screening_authorization_request.json"
APPROVED_DIGEST = "e8d44600fb3a9177ae691fd8f49ac1c06305b004db7ccd50d391c9876356a230"


def _plan_and_descriptor(tmp_path: Path, plan_bytes: bytes, descriptor_bytes: bytes) -> Path:
    plan = tmp_path / ".omo" / "plans" / PLAN.name
    descriptor = tmp_path / ".omo" / "approvals" / DESCRIPTOR.name
    plan.parent.mkdir(parents=True)
    descriptor.parent.mkdir(parents=True)
    plan.write_bytes(plan_bytes)
    descriptor.write_bytes(descriptor_bytes)
    return plan


def _validate_methods(plan: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--document",
            str(METHODS),
            "--config",
            str(CONFIG),
            "--plan",
            str(plan),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_methods_lock_accepts_only_operational_checkbox_progress(tmp_path: Path) -> None:
    # Given: the current plan differs from approval only in task-progress checkbox state.
    plan = _plan_and_descriptor(tmp_path, PLAN.read_bytes(), DESCRIPTOR.read_bytes())

    # When: the Methods validator checks the descriptor-bound plan.
    result = _validate_methods(plan)

    # Then: it remains the independently approved original digest.
    assert result.returncode == 0, result.stdout + result.stderr
    assert APPROVED_DIGEST in CONFIG.read_text(encoding="utf-8")


def test_methods_lock_rejects_substantive_plan_edit(tmp_path: Path) -> None:
    # Given: an approved descriptor and a plan with a substantive sentence mutation.
    plan = _plan_and_descriptor(
        tmp_path,
        PLAN.read_bytes().replace(b"Freeze strict inventory", b"Freeze changed inventory", 1),
        DESCRIPTOR.read_bytes(),
    )

    # When: the Methods validator checks the substantive mutation.
    # Then: operational normalization cannot accept the substantive change.
    result = _validate_methods(plan)
    assert result.returncode != 0
    assert "METHODS_PLAN_BINDING_MISMATCH" in result.stdout


def test_methods_lock_rejects_descriptor_without_exact_lowercase_newline(tmp_path: Path) -> None:
    # Given: a descriptor without the required trailing newline.
    plan = _plan_and_descriptor(tmp_path, PLAN.read_bytes(), APPROVED_DIGEST.encode("ascii"))

    # When: the Methods validator checks malformed independent approval bytes.
    # Then: malformed independent approval bytes are rejected.
    result = _validate_methods(plan)
    assert result.returncode != 0
    assert "METHODS_PLAN_BINDING_MISMATCH" in result.stdout


def test_calibration_config_rejects_approved_plan_digest_mutation(tmp_path: Path) -> None:
    # Given: a calibration config whose approved-plan field differs from the descriptor.
    config = tmp_path / "filter_v5_bct_calibration.yaml"
    config.write_bytes(CONFIG.read_bytes().replace(APPROVED_DIGEST.encode("ascii"), b"0" * 64))

    # When: a zero-call calibration command validates the config.
    # Then: it rejects the drift before a preview can construct a request.
    with pytest.raises(CalibrationAuthorizationError, match="CALIBRATION_CONFIG_PLAN_MISMATCH"):
        _validate_config(config)


def test_reseal_replaces_only_stale_task3_reports_with_approved_digest(tmp_path: Path) -> None:
    # Given: a structurally valid Task-3 report set bound to a stale digest.
    bundle = tmp_path / "bundle"
    stage = CalibrationStageResult.waiting("screening", "AWAITING_SCREENING_AUTHORIZATION")
    stage_path = tmp_path / "screening-stage.json"
    stage.write_atomic(stage_path)
    for report_id in ("authority-transition", "methods-lock", "freeze-a"):
        build_evidence_report(bundle, report_id, None, "0" * 64)
    build_evidence_report(bundle, "screening", stage_path, "0" * 64)

    # When: the deterministic repair path reseals the four Task-3 reports.
    common = [
        "--bundle",
        str(bundle),
        "--plan",
        str(PLAN),
        "--artifact-root",
        str(tmp_path / "absent-live-root"),
        "--reseal-existing",
    ]
    for report, extra in (
        ("authority-transition", []),
        ("methods-lock", []),
        ("freeze-a", ["--freeze-a", str(FREEZE_A)]),
        (
            "screening",
            [
                "--freeze-a",
                str(FREEZE_A),
                "--authorization-request",
                str(SCREENING_REQUEST),
                "--stage-result",
                str(stage_path),
            ],
        ),
    ):
        result = subprocess.run(
            [sys.executable, str(EVIDENCE_SCRIPT), "--report", report, *common, *extra],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    # Then: the aggregate verifier accepts only the descriptor-bound approved digest.
    verified = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--through",
            "screening",
            "--bundle",
            str(bundle),
            "--plan",
            str(PLAN),
            "--artifact-root",
            str(tmp_path / "absent-live-root"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert verified.stdout == "APPROVE\n"
