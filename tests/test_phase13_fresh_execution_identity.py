from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest

from memcontam.readiness.phase13_main_v3_runner import DurableParentRecordV3, V3MainRun
from memcontam.readiness.phase13_v3_entrypoint import (
    SelectedExecutionV3,
    SelectionRequest,
    select_execution,
)
from memcontam.readiness.phase13_v3_request import PackageBindingV3

from .test_phase13_corrective_identity import fresh_selection, fresh_source
from .test_phase13_v3_entrypoint_integration import deny_external
from .test_phase13_v3_native_parent_reopen import NativeProvider

__all__ = ["deny_external", "fresh_selection", "fresh_source"]


def test_fresh_identity_parent_reopens_without_redispatch(
    fresh_selection: SelectionRequest, deny_external,
) -> None:
    selected = select_execution(fresh_selection, "run")
    assert isinstance(selected, SelectedExecutionV3)
    identity = selected.package.identity
    directory = fresh_selection.repository_root / identity.run_id
    factories: list[str] = []

    def provider(binding: PackageBindingV3) -> NativeProvider:
        factories.append(binding.identity.run_id)
        assert binding.identity == identity
        return NativeProvider()

    run = V3MainRun.open(selected, directory, create=True, seed=0)
    try:
        report = run.execute(directory / "cache", max_units=1,
            tranche_ceiling_krw=450000, provider_factory=provider)
        assert report.completed_count == 1
        assert report.provider_calls_issued == 50
        assert factories and set(factories) == {identity.run_id}
    finally:
        run.close()

    selected = select_execution(fresh_selection, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    reopened = V3MainRun.open(selected, directory, create=False, seed=0)
    try:
        with reopened.private.connect() as connection:
            unit_id, raw, sha256 = connection.execute(
                "SELECT unit_id, raw, sha256 FROM parents"
            ).fetchone()
        assert hashlib.sha256(raw).hexdigest() == sha256
        assert reopened.ledger.read_record(f"{unit_id}.parent.json") == raw
        parent = reopened._load_parent(unit_id, raw, sha256)
        assert isinstance(parent, DurableParentRecordV3)
        assert parent.unit_evidence.unit_id == unit_id
        before = len(factories)
        report = reopened.execute(directory / "cache", max_units=1,
            tranche_ceiling_krw=450000, provider_factory=provider)
        assert report.provider_calls_issued == 50
        assert len(factories) == before
    finally:
        reopened.close()


def test_fresh_run_rejects_occupied_historical_directory(
    fresh_selection: SelectionRequest, deny_external,
) -> None:
    directory = fresh_selection.repository_root / "occupied"
    directory.mkdir(mode=0o700)
    marker = directory / "historical-marker"
    marker.write_bytes(b"immutable historical fixture")
    selected = select_execution(fresh_selection, "run")
    assert isinstance(selected, SelectedExecutionV3)
    try:
        with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
            V3MainRun.open(selected, directory, create=True, seed=0)
        assert marker.read_bytes() == b"immutable historical fixture"
    finally:
        selected.close()


@pytest.mark.parametrize("schema", ["phase13_main_execution_freeze_v1", "phase13_main_execution_freeze_v2"])
@pytest.mark.parametrize("command", ["run", "resume", "status"])
def test_fresh_request_cannot_select_stale_package(
    tmp_path: Path, schema: str, command: Literal["run", "resume", "status"], deny_external,
) -> None:
    package = tmp_path / "package.json"
    package.write_text(json.dumps({"schema_version": schema}))
    request = SelectionRequest(tmp_path, package, tmp_path / "absent-authorization",
        None, None, "phase13-main-a-disposable-fresh-v3")
    with pytest.raises(ValueError, match="MAIN_PACKAGE_VERSION_UNSUPPORTED"):
        select_execution(request, command)
