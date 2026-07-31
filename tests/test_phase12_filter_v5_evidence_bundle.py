from __future__ import annotations

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    build_evidence_report,
    validate_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult


def test_evidence_report_binds_stage_result_digest(tmp_path) -> None:
    stage = CalibrationStageResult.waiting("screening", "AWAITING_SCREENING_AUTHORIZATION")
    stage_path = tmp_path / "stage.json"
    stage.write_atomic(stage_path)
    bundle = tmp_path / "bundle"

    build_evidence_report(bundle, "screening", stage_path, "a" * 64)
    assert validate_evidence_bundle(bundle, "a" * 64).valid is False
    for report_id in ("authority-transition", "methods-lock", "freeze-a"):
        build_evidence_report(bundle, report_id, stage_path, "a" * 64)

    assert validate_evidence_bundle(bundle, "a" * 64).valid is True
