from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from memcontam.readiness.phase13_main_v3_runner import V3MainRun
from memcontam.readiness.phase13_main_terminal_partial import TerminalPartialParent
from memcontam.readiness.phase13_v3_entrypoint import EntrypointError, SelectedExecutionV3, SelectionRequest, select_execution
from memcontam.readiness.phase13_v3_entrypoint_paths import PrivateLedger
from memcontam.readiness.phase13_v3_request import CompiledProviderRequestV3, PackageBindingV3
from memcontam.clients.base import LLMResponse

from .phase13_corrective_identity import corrective_identity
from .test_phase13_v3_entrypoint_fixture import build_entrypoint_bytes, seal_fixture_closure
from .test_phase13_v3_native_parent_reopen import NativeProvider, _units
from .test_phase13_v3_native_parent_reopen import deny_main_external as deny_main_external
from .test_phase13_runner_safety import local_authority as local_authority


@pytest.fixture(scope="session")
def c01_entrypoint_bytes() -> dict[str, bytes]:
    return build_entrypoint_bytes((0,), production_units=(replace(_units()[-1], sequence=0),))


@pytest.mark.parametrize("partial", (False, True), ids=("completed", "terminal-partial"))
@pytest.mark.parametrize("boundary,tamper", (
    ("before-publication", None), ("after-publication", None), ("after-sql", None),
    ("after-publication", "corrupt"), ("after-publication", "foreign"),
))
def test_parent_split_crash_reopen_without_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    deny_main_external: Callable[[], None], partial: bool, boundary: str, tamper: str | None,
    local_authority: Path, c01_entrypoint_bytes: dict[str, bytes],
) -> None:
    unit = replace(_units()[-1], sequence=0)
    for path, raw in c01_entrypoint_bytes.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    request = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
        local_authority, tmp_path / "authorization.sha256", corrective_identity().run_id)
    directory = tmp_path / "split-run"

    def open_run(create: bool) -> V3MainRun:
        selected = select_execution(request, "run" if create else "resume")
        assert isinstance(selected, SelectedExecutionV3)
        return V3MainRun.open(selected, directory, create=create, seed=0)

    open_run(True).close()
    pid = os.fork()
    if pid == 0:
        run = open_run(False)
        publish = PrivateLedger.publish_record
        connect = PrivateLedger.connect

        class Provider(NativeProvider):
            def send_compiled_v3(self, compiled: CompiledProviderRequestV3,
                                 before_request: Callable[[], None]) -> LLMResponse:
                if partial and compiled.key.ordinal == 1:
                    before_request()
                    raise RuntimeError("construction-only terminal failure")
                return super().send_compiled_v3(compiled, before_request)

        def interrupted_publish(private: PrivateLedger, name: str, raw: bytes) -> None:
            if name.endswith(".parent.json") and boundary == "before-publication":
                os._exit(77)
            publish(private, name, raw)
            if name.endswith(".parent.json") and boundary == "after-publication":
                os._exit(77)

        @contextmanager
        def interrupted_connect(private: PrivateLedger) -> Iterator[sqlite3.Connection]:
            indexed = False

            def traced(statement: str) -> None:
                nonlocal indexed
                if statement.startswith("UPDATE parents SET raw="):
                    indexed = True

            with connect(private) as connection:
                connection.set_trace_callback(traced)
                yield connection
            if indexed and boundary == "after-sql":
                os._exit(77)

        monkeypatch.setattr(PrivateLedger, "publish_record", interrupted_publish)
        monkeypatch.setattr(PrivateLedger, "connect", interrupted_connect)
        deny_main_external()
        run.execute(tmp_path / "cache", max_units=1, tranche_ceiling_krw=450000,
                    provider_factory=lambda _binding: Provider())
        os._exit(78)
    _, exit_status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(exit_status) == 77
    parent = directory / f"{unit.unit_id}.parent.json"
    with sqlite3.connect(directory / "main_run_ledger_v3.sqlite3") as connection:
        raw, checksum = connection.execute("SELECT raw, sha256 FROM parents").fetchone()
    assert (raw is not None) == (boundary == "after-sql")
    assert parent.exists() == (boundary != "before-publication")
    original = parent.read_bytes() if parent.exists() else None
    if tamper == "corrupt":
        parent.write_bytes(b"{}")
    if tamper == "foreign":
        assert original is not None
        payload = json.loads(original)
        payload["identity"]["run_id"] = "foreign-run"
        parent.write_bytes(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    preserved = parent.read_bytes() if parent.exists() else None

    def forbidden_provider(_binding: PackageBindingV3) -> NativeProvider:
        pytest.fail("reopen must not construct a provider or redispatch")

    for _restart in range(2):
        if tamper is not None or (boundary == "before-publication" and not partial):
            expected = ("MAIN_AUTHORIZATION_BINDING_MISMATCH" if tamper is not None
                        else "MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED")
            with pytest.raises(EntrypointError, match=expected):
                refused = open_run(False)
                refused.close()
            with sqlite3.connect(directory / "main_run_ledger_v3.sqlite3") as connection:
                assert connection.execute("SELECT raw, sha256 FROM parents").fetchone() == (None, None)
            assert (parent.read_bytes() if parent.exists() else None) == preserved
            continue
        reopened = open_run(False)
        try:
            status = reopened.status()
            assert (status.completed_count, status.terminal_technical_missing_count,
                    status.pending_count) == ((0, 1, 0) if partial else (1, 0, 0))
            assert status.provider_calls_issued == (2 if partial else 50)
            with reopened.private.connect() as connection:
                indexed_raw, indexed_sha = connection.execute("SELECT raw, sha256 FROM parents").fetchone()
            if original is not None:
                assert indexed_raw == original == parent.read_bytes()
                assert indexed_sha == hashlib.sha256(original).hexdigest()
                record = reopened._load_parent(unit.unit_id, indexed_raw, indexed_sha)
                if partial:
                    assert isinstance(record, TerminalPartialParent)
                    assert record.terminal_key.ordinal == 1
                    assert record.interrupted_keys == (record.terminal_key,)
                    assert len(record.archive.records) == len(record.observed_calls) == 1
            before = reopened.ledger.rows()
            reopened.execute(tmp_path / "cache", max_units=1, tranche_ceiling_krw=450000,
                             provider_factory=forbidden_provider)
            assert reopened.ledger.rows() == before
        finally:
            reopened.close()
