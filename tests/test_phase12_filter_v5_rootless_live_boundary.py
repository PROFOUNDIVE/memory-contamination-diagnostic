from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge import bct_live, registry_calibration
from memcontam.experiment.phase12.filter_challenge.bct_live_authorization import (
    CalibrationAuthorizationError,
)


ROOTLESS_FORBIDDEN = "ROOTLESS_PROFILE_FORBIDDEN"
ROOTLESS_RECEIPT = {
    "schema_version": "rootless_local_receipt_v1",
    "profile": "local_rootless_non_authoritative",
    "kind": "rootless_local_receipt",
    "terminal": "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED",
}


def test_rootless_live_authorization_writes_no_stage_or_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "runs" / "phase12-filter-v5-bct-live-v1"
    stage_result = tmp_path / "stage-result.json"
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(ROOTLESS_RECEIPT), encoding="utf-8")
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", artifact_root)

    with pytest.raises(CalibrationAuthorizationError, match=ROOTLESS_FORBIDDEN):
        bct_live._run_cli_stage(
            argparse.Namespace(
                config=tmp_path / "config.yaml",
                freeze_a=tmp_path / "freeze-a.json",
                artifact_root=artifact_root,
                run_id="run-001",
                stage_result=stage_result,
                authorization=authorization,
                expected_authorization_sha256=hashlib.sha256(authorization.read_bytes()).hexdigest(),
                authorization_request=tmp_path / "request.json",
            ),
            "screening",
        )

    assert not stage_result.exists()
    assert not artifact_root.exists()
