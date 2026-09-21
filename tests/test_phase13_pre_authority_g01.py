from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TypeVar

import pytest

from memcontam.experiment import phase13_ordinary_runtime
from memcontam.readiness import phase13_main_live_cli
from memcontam.readiness.phase13_main_v3_runner import V3MainRun
from memcontam.readiness.phase13_v3_entrypoint import (
    SelectedExecutionV3,
    SelectionRequest,
    select_execution,
)
from memcontam.readiness.phase13_v3_request import RequestKeyV3

from .test_phase13_v3_entrypoint_fixture import entrypoint_bytes as entrypoint_bytes
from .test_phase13_v3_entrypoint_fixture import entrypoint_fixture as entrypoint_fixture
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external

StateT = TypeVar("StateT")


class _PostIntentFailure(ValueError):
    pass


def _fail_native_state(_serialize: Callable[[StateT], StateT], _state: StateT) -> bytes:
    raise _PostIntentFailure("provider-free post-intent runtime failure")


@pytest.mark.usefixtures("deny_external")
def test_post_intent_value_error_preserves_runtime_failure_identity(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint_fixture: SelectionRequest,
) -> None:
    request = entrypoint_fixture
    monkeypatch.setattr(phase13_ordinary_runtime, "native_state_bytes", _fail_native_state)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase13-main-a-live",
            "run",
            "--repository-root",
            str(request.repository_root),
            "--package",
            str(request.package_path),
            "--authorization",
            str(request.authorization_path),
            "--authority-root",
            str(request.authority_root),
            "--expected-authorization-sha256-file",
            str(request.expected_authorization_sha256_file),
            "--run-root",
            str(request.repository_root),
            "--run-id",
            str(request.run_id),
            "--seed",
            "0",
            "--cache-root",
            str(request.repository_root / "cache"),
            "--tranche-ceiling-krw",
            "450000",
            "--max-units",
            "1",
            "--allow-live-calls",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        phase13_main_live_cli.main()

    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    run = V3MainRun.open(selected, request.repository_root / str(request.run_id), create=False, seed=0)
    try:
        unit = selected.package.production[0]
        key = RequestKeyV3(parent_id=unit.unit_id, stage="no_memory_generate", ordinal=0)
        states = run.ledger.states()
        assert sum(state.kind == "DISPATCH_INTENT_PERSISTED" for state in states.values()) == 1
        assert states[key.dispatch_id].kind == "DISPATCH_INTENT_PERSISTED"
        assert states[key.dispatch_id].revision == 1
        assert len(run.ledger.rows()) == 1
        assert run.status().pending_count == 1
        assert run.status().completed_count == 0
        assert run.status().provider_calls_issued == 0
    finally:
        run.close()
    assert raised.value.code == "MAIN_RUN_POST_INTENT_RUNTIME_FAILURE"
