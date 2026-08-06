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


def _bct_args(tmp_path: Path, artifact_root: Path, stage_result: Path) -> argparse.Namespace:
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(ROOTLESS_RECEIPT), encoding="utf-8")
    return argparse.Namespace(
        config=tmp_path / "config.yaml",
        freeze_b=tmp_path / "freeze-b.json",
        artifact_root=artifact_root,
        run_id="run-001",
        stage_result=stage_result,
        authorization=authorization,
        expected_authorization_sha256=hashlib.sha256(authorization.read_bytes()).hexdigest(),
        authorization_request=tmp_path / "request.json",
    )


def test_rootless_bct_authorization_precedes_existing_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "runs" / "phase12-filter-v5-bct-live-v1"
    artifact_root.mkdir(parents=True)
    marker = artifact_root / "marker"
    marker.write_bytes(b"unchanged")
    stage_result = tmp_path / "stage-result.json"
    stage_result.write_bytes(b"existing-stage")
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", artifact_root)

    with pytest.raises(CalibrationAuthorizationError, match=ROOTLESS_FORBIDDEN):
        bct_live._run_cli_stage(_bct_args(tmp_path, artifact_root, stage_result), "bct")

    assert marker.read_bytes() == b"unchanged"
    assert tuple(artifact_root.iterdir()) == (marker,)
    assert stage_result.read_bytes() == b"existing-stage"


def test_rootless_bct_authorization_precedes_waiting_screening_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "runs" / "phase12-filter-v5-bct-live-v1"
    stage_result = tmp_path / "stage-result.json"
    stage_result.write_bytes(b"existing-stage")
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(bct_live, "waiting_screening_stage", pytest.fail)

    with pytest.raises(CalibrationAuthorizationError, match=ROOTLESS_FORBIDDEN):
        bct_live._run_cli_stage(_bct_args(tmp_path, artifact_root, stage_result), "bct")

    assert not artifact_root.exists()
    assert stage_result.read_bytes() == b"existing-stage"


def test_nonrootless_bct_preserves_existing_artifact_blocked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "runs" / "phase12-filter-v5-bct-live-v1"
    artifact_root.mkdir(parents=True)
    stage_result = tmp_path / "stage-result.json"
    args = _bct_args(tmp_path, artifact_root, stage_result)
    authorization = tmp_path / "authorization.json"
    authorization.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", artifact_root)
    args.authorization = authorization
    args.expected_authorization_sha256 = hashlib.sha256(authorization.read_bytes()).hexdigest()

    result = bct_live._run_cli_stage(args, "bct")

    assert result.reason_code == "LIVE_ARTIFACT_ROOT_EXISTS"
    assert json.loads(stage_result.read_text(encoding="utf-8"))["reason_code"] == (
        "LIVE_ARTIFACT_ROOT_EXISTS"
    )
