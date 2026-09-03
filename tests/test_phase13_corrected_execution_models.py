from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memcontam.readiness.phase13_main_execution_models import (
    AuthorizedExecution,
    MainExecutionFreeze,
)


ROOT = Path(__file__).resolve().parents[1]


def _legacy_package_payload() -> dict:
    return json.loads(
        (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_corrected_freeze_requires_repository_and_run_identity() -> None:
    payload = _legacy_package_payload()
    payload.update(
        schema_version="phase13_main_execution_freeze_v2",
        package_id="phase13-main-a-corrected-execution-freeze-v2",
    )

    with pytest.raises(ValidationError):
        MainExecutionFreeze.model_validate_json(json.dumps(payload))


def test_corrected_freeze_accepts_complete_prospective_identity() -> None:
    payload = _legacy_package_payload()
    payload.update(
        schema_version="phase13_main_execution_freeze_v2",
        package_id="phase13-main-a-corrected-execution-freeze-v2",
        corrected_run_id="phase13-main-a-corrected-20260903-t1",
        repository_commit="d1ac6c84236ec63c367775d24aa953176d321ce0",
        repository_tree_sha256="1" * 64,
    )

    freeze = MainExecutionFreeze.model_validate_json(json.dumps(payload))

    assert freeze.corrected_run_id == "phase13-main-a-corrected-20260903-t1"


def test_corrected_authorization_binds_corrected_package_identity() -> None:
    payload = json.loads(
        (ROOT / "data/phase13/main/mr_p6/authorized_execution_v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload.update(
        schema_version="phase13_main_authorization_v2",
        authorization_id="phase13-main-a-corrected-authorized-execution-v2",
        execution_package_id="phase13-main-a-corrected-execution-freeze-v2",
        corrected_run_id="phase13-main-a-corrected-20260903-t1",
    )

    authorization = AuthorizedExecution.model_validate_json(json.dumps(payload))

    assert authorization.corrected_run_id == "phase13-main-a-corrected-20260903-t1"
