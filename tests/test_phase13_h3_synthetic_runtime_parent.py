"""Disposable runner checks: synthetic runtime identity is not runtime acceptance."""

from __future__ import annotations

import socket
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from memcontam.readiness.phase13_main_v3_runner import V3MainRun
from memcontam.readiness.phase13_v3_entrypoint import SelectedExecutionV3, select_execution
from memcontam.readiness.phase13_v3_entrypoint_paths import PrivateLedger
from memcontam.readiness.phase13_v3_runtime_identity import RuntimeIdentityV3

from .phase13_runner_safety_fixture import FakeProvider
from .test_phase13_v3_entrypoint_fixture import (
    AUTHORITY, build_entrypoint_bytes, seal_fixture_closure,
)
from memcontam.readiness.phase13_v3_entrypoint import SelectionRequest
from .phase13_corrective_identity import corrective_identity


SYNTHETIC_RUNTIME = RuntimeIdentityV3(
    executable="/synthetic/non-executable", prefix="/synthetic", base_prefix="/synthetic",
    implementation="CPython", version="3.11.15",
    requirements_sha256="a" * 64, requirements_dev_sha256="b" * 64,
    versions=(), pythonpath="/synthetic/src", module_origin="/synthetic/src/memcontam/__init__.py",
    package_paths=("/synthetic/src/memcontam",),
)


@pytest.fixture
def fake_runtime():
    # Keep production's actual-runtime guard unchanged; only disposable fixture
    # building and selection see this fixed, independently selected test value.
    with ExitStack() as stack:
        for module in (
            "memcontam.readiness.phase13_v3_runtime_identity",
            "memcontam.readiness.phase13_v3_builder",
            "tests.test_phase13_v3_entrypoint_fixture",
        ):
            stack.enter_context(patch(f"{module}.freeze_runtime_identity", return_value=SYNTHETIC_RUNTIME))
        yield


def test_published_parent_orphan_reopens_without_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_runtime,
) -> None:
    def denied(*_args, **_kwargs):
        raise AssertionError("external transport forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    for path, raw in build_entrypoint_bytes((0,)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    request = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
                               AUTHORITY, tmp_path / "authorization.sha256", corrective_identity().run_id)
    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    directory = tmp_path / "run"
    run = V3MainRun.open(selected, directory, create=True, seed=0)
    provider = FakeProvider()
    unit_id = selected.package.production[0].unit_id
    publish = PrivateLedger.publish_record
    triggered = False

    def interrupted(self: PrivateLedger, name: str, raw: bytes) -> None:
        nonlocal triggered
        publish(self, name, raw)
        if name == f"{unit_id}.parent.json" and not triggered:
            triggered = True
            raise RuntimeError("synthetic owner loss after durable parent publication")

    monkeypatch.setattr(PrivateLedger, "publish_record", interrupted)
    try:
        with pytest.raises(RuntimeError, match="synthetic owner loss"):
            run.execute(tmp_path / "cache", max_units=1, tranche_ceiling_krw=450000,
                        provider_factory=provider.factory)
        assert triggered
        with run.private.connect() as connection:
            raw, checksum = connection.execute("SELECT raw, sha256 FROM parents WHERE unit_id=?", (unit_id,)).fetchone()
            assert raw is None and checksum is None
        raw = run.private.read_record(f"{unit_id}.parent.json")
        prior_attempts = len(provider.requests)
    finally:
        run.close()
    monkeypatch.setattr(PrivateLedger, "publish_record", publish)
    reopened_selected = select_execution(request, "status")
    assert isinstance(reopened_selected, SelectedExecutionV3)
    reopened = V3MainRun.open(reopened_selected, directory, create=False, seed=0)
    try:
        assert reopened.status().completed_count == 1
        with reopened.private.connect() as connection:
            assert connection.execute("SELECT raw FROM parents WHERE unit_id=?", (unit_id,)).fetchone()[0] == raw
        reopened.execute(tmp_path / "cache", max_units=1, tranche_ceiling_krw=450000,
                         provider_factory=provider.factory)
        assert len(provider.requests) == prior_attempts
    finally:
        reopened.close()
