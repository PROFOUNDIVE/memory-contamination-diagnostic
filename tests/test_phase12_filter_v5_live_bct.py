from __future__ import annotations

from pathlib import Path

from memcontam.experiment.phase12.filter_challenge import registry_calibration
from memcontam.experiment.phase12.filter_challenge.bct_live import run_screen_controls


def test_missing_screening_authorization_writes_waiting_stage_without_factory(
    tmp_path: Path, monkeypatch
) -> None:
    calls = 0

    def factory() -> None:
        nonlocal calls
        calls += 1

    artifact_root = tmp_path / "runs" / "phase12-filter-v5-bct-live-v1"
    monkeypatch.setattr(registry_calibration, "ARTIFACT_ROOT", artifact_root)
    result = run_screen_controls(
        artifact_root=artifact_root,
        run_id="filter-v5-screening-v1-attempt-001",
        stage_result=tmp_path / "stage-result.json",
        authorization=None,
        expected_authorization_sha256=None,
        client_factory=factory,
    )

    assert result.terminal_status == "AWAITING_SCREENING_AUTHORIZATION"
    assert result.provider_calls_issued == 0
    assert calls == 0
