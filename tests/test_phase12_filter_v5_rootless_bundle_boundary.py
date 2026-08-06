from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge import pilot_b_readiness
from memcontam.experiment.phase12.filter_challenge.bct_archive_models import ArchiveValidation


ROOTLESS_FORBIDDEN = "ROOTLESS_PROFILE_FORBIDDEN"
ROOTLESS_RECEIPT = {
    "schema_version": "rootless_local_receipt_v1",
    "profile": "local_rootless_non_authoritative",
    "kind": "rootless_local_receipt",
    "terminal": "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED",
}


@pytest.mark.parametrize(
    ("report_name", "rootless_stage"),
    (("screening", False), ("freeze_b_search_config", False), ("bct_execution", False), ("screening", True)),
)
def test_readiness_bundle_rejects_rootless_reports_and_stage_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_name: str,
    rootless_stage: bool,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("screening", "freeze_b_search_config", "bct_execution"):
        (bundle / f"{name}_report.json").write_text("{}", encoding="utf-8")
    target = bundle / f"{report_name}_report.json"
    if rootless_stage:
        stage = tmp_path / "stage-result.json"
        stage.write_text(json.dumps(ROOTLESS_RECEIPT), encoding="utf-8")
        target.write_text(json.dumps({"stage_result_path": str(stage)}), encoding="utf-8")
    else:
        target.write_text(json.dumps(ROOTLESS_RECEIPT), encoding="utf-8")
    monkeypatch.setattr(
        pilot_b_readiness, "validate_evidence_bundle", lambda *_: ArchiveValidation(True)
    )

    with pytest.raises(ValueError, match=ROOTLESS_FORBIDDEN):
        pilot_b_readiness.readiness_from_bundle(bundle, "a" * 64)
