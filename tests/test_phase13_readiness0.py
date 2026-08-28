from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from memcontam.readiness.phase13_readiness0 import (
    Phase13Readiness0Error,
    validate_readiness0_request,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/phase13/main/mr_p4/readiness0_request_v1.json"


def test_readiness0_request_preserves_authorized_non_scientific_boundary() -> None:
    request = validate_readiness0_request(ARTIFACT)

    assert request.status == "BLOCKED_EXTERNAL_DEPENDENCY"
    assert request.scientific_result is False
    assert request.main_result is False
    assert request.measured_main_a_trajectory_count == 0
    assert request.authorization.allow_live_calls is True
    assert request.authorization.authorizes_mr_p5 is False
    assert request.authorization.authorizes_mr_p6 is False
    assert request.authorization.authorizes_main_a is False
    assert request.case_matrix == ("luna_responses_terminal_success",)
    assert set(request.external_blockers) == {
        "OPENAI_API_KEY_MISSING",
        "F1C_RUNTIME_ENVIRONMENT_NOT_CONFIGURED",
    }


def test_readiness0_request_rejects_scientific_promotion(tmp_path: Path) -> None:
    artifact = tmp_path / ARTIFACT.name
    shutil.copyfile(ARTIFACT, artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["scientific_result"] = True
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase13Readiness0Error, match="READINESS0_REQUEST_INVALID"):
        validate_readiness0_request(artifact)
