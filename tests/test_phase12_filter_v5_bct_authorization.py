from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from memcontam.experiment.phase12.filter_challenge.bct_live import (
    CalibrationAuthorizationError,
    load_authorization,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    ScreeningAuthorizationV1,
)


def test_authorization_requires_matching_descriptor_digest_and_is_unexpired(tmp_path) -> None:
    authorization = ScreeningAuthorizationV1(
        authorization_id="screening-auth-001",
        issued_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        run_id="filter-v5-screening-v1-attempt-001",
        request_sha256="a" * 64,
        implementation_commit="b" * 40,
        artifact_root="/home/hyunwoo/git/memory-contamination-diagnostic-filter-v5/runs/phase12-filter-v5-bct-live-v1",
        ledger_id="filter-v5-bct-budget-v1",
        model_id="gpt-4o-2024-11-20",
        approved_plan_sha256="c" * 64,
        authority_manifest_sha256="d" * 64,
        freeze_sha256="e" * 64,
        provider="openai_responses",
        decoding_sha256="f" * 64,
        maximum_calls=90,
        maximum_input_tokens=368640,
        maximum_output_tokens=57600,
        hard_ceiling_microusd=2_000_000,
        maximum_wall_seconds=3600,
    )
    path = tmp_path / "authorization.json"
    path.write_text(authorization.model_dump_json(), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_authorization(path, expected, ScreeningAuthorizationV1)

    assert loaded.authorization_id == authorization.authorization_id
    with pytest.raises(CalibrationAuthorizationError, match="AUTHORIZATION_DIGEST_MISMATCH"):
        load_authorization(path, "0" * 64, ScreeningAuthorizationV1)
