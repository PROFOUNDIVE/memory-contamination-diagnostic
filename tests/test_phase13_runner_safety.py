from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from .phase13_runner_safety_fixture import FakeProvider, open_run
from .test_phase13_v3_entrypoint_fixture import build_entrypoint_bytes
from .test_phase13_v3_entrypoint_fixture import entrypoint_fixture as _source_selection
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external

source_selection = _source_selection


@pytest.fixture(scope="session")
def entrypoint_bytes() -> dict[str, bytes]:
    return build_entrypoint_bytes((0, 1))


@pytest.fixture(scope="session")
def local_authority(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from memcontam.readiness.phase13_authority_files import (
        PROVENANCE_FILENAME,
        REVISION_MANIFEST_FILENAME,
        ROUTED_DOCUMENTS,
    )
    from .test_phase13_v3_entrypoint_fixture import AUTHORITY

    root = tmp_path_factory.mktemp("safety-authority")
    for filename in [
        *(name for _, name in ROUTED_DOCUMENTS),
        PROVENANCE_FILENAME,
        REVISION_MANIFEST_FILENAME,
    ]:
        shutil.copyfile(AUTHORITY / filename, root / filename)
    return root


@pytest.fixture
def entrypoint_fixture(source_selection, local_authority: Path):
    root = source_selection.repository_root / "authority"
    shutil.copytree(local_authority, root)
    return replace(source_selection, authority_root=root)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch, deny_external) -> FakeProvider:
    import memcontam.readiness.phase13_main_request_dispatch as dispatch

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: 1)
    return FakeProvider()


def test_second_owner_rejected_before_provider(entrypoint_fixture, provider: FakeProvider) -> None:
    run = open_run(entrypoint_fixture, create=True)
    try:
        with pytest.raises(ValueError, match="MAIN_RUN_ALREADY_OWNED"):
            second = open_run(entrypoint_fixture, create=False)
            second.close()
        assert (provider.constructors, len(provider.requests)) == (0, 0)
    finally:
        run.close()


def test_rehashed_parent_evidence_tamper_fails_closed(
    entrypoint_fixture, provider: FakeProvider
) -> None:
    run = open_run(entrypoint_fixture, create=True)
    try:
        assert run.execute(
            Path("unused"),
            max_units=1,
            tranche_ceiling_krw=450000,
            provider_factory=provider.factory,
        ).completed_count == 1
        unit_id = run.selected.package.production[0].unit_id
        with run.private.connect() as connection:
            raw = connection.execute(
                "SELECT raw FROM parents WHERE unit_id=?", (unit_id,)
            ).fetchone()[0]
            payload = json.loads(raw)
            assert run.private.read_record(f"{unit_id}.parent.json") == raw
            call = payload["unit_evidence"]["provider_calls"][0]
            call["messages"][0]["content"] = "tampered"
            call["provider_request_contract"]["input_sha256"] = hashlib.sha256(
                json.dumps(
                    call["messages"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            tampered = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            connection.execute(
                "UPDATE parents SET raw=?, sha256=? WHERE unit_id=?",
                (tampered, hashlib.sha256(tampered).hexdigest(), unit_id),
            )
        with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
            run.status()
        with run.private.connect() as connection:
            connection.execute(
                "UPDATE parents SET raw=?, sha256=? WHERE unit_id=?",
                (raw, hashlib.sha256(raw).hexdigest(), unit_id),
            )
        archive = payload["unit_evidence"]["evidence"][
            "runtime_evidence"
        ]["production_observability_archive"]
        archive["records"][0]["run_id"] = "tampered"
        tampered = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        with run.private.connect() as connection:
            connection.execute(
                "UPDATE parents SET raw=?, sha256=? WHERE unit_id=?",
                (tampered, hashlib.sha256(tampered).hexdigest(), unit_id),
            )
        with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
            run.status()
        payload = json.loads(raw)
        call = payload["unit_evidence"]["provider_calls"][0]
        call["provider_cost_usd"] = 1.0
        call["authoritative_provider_cost_usd"] = 1.0
        payload["unit_evidence"]["realized_cost_krw"] += 1600
        tampered = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        with run.private.connect() as connection:
            connection.execute(
                "UPDATE parents SET raw=?, sha256=? WHERE unit_id=?",
                (tampered, hashlib.sha256(tampered).hexdigest(), unit_id),
            )
        with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
            run.status()
    finally:
        run.close()


def test_malformed_orphan_parent_file_fails_on_reopen_before_provider(
    entrypoint_fixture, provider: FakeProvider
) -> None:
    run = open_run(entrypoint_fixture, create=True)
    unit_id = run.selected.package.production[0].unit_id
    run.close()
    (entrypoint_fixture.repository_root / "safety-run" / f"{unit_id}.parent.json").write_bytes(
        b"{}"
    )

    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE|MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        open_run(entrypoint_fixture, create=False)

    assert (provider.constructors, len(provider.requests)) == (0, 0)


def test_valid_published_parent_reconciles_null_sql_row_without_redispatch(
    entrypoint_fixture, provider: FakeProvider
) -> None:
    run = open_run(entrypoint_fixture, create=True)
    try:
        assert run.execute(
            Path("unused"),
            max_units=1,
            tranche_ceiling_krw=450000,
            provider_factory=provider.factory,
        ).completed_count == 1
        unit_id = run.selected.package.production[0].unit_id
        with run.private.connect() as connection:
            connection.execute(
                "UPDATE parents SET raw=NULL, sha256=NULL WHERE unit_id=?",
                (unit_id,),
            )
    finally:
        run.close()
    provider.constructors = 0
    provider.requests.clear()

    reopened = open_run(entrypoint_fixture, create=False)
    try:
        assert reopened.status().completed_count == 1
        with reopened.private.connect() as connection:
            raw, checksum = connection.execute(
                "SELECT raw, sha256 FROM parents WHERE unit_id=?", (unit_id,)
            ).fetchone()
        assert raw == reopened.private.read_record(f"{unit_id}.parent.json")
        assert checksum == hashlib.sha256(raw).hexdigest()
    finally:
        reopened.close()

    assert (provider.constructors, len(provider.requests)) == (0, 0)


def test_stale_identity_orphan_parent_fails_on_reopen_before_provider(
    entrypoint_fixture, provider: FakeProvider
) -> None:
    run = open_run(entrypoint_fixture, create=True)
    try:
        assert run.execute(
            Path("unused"),
            max_units=1,
            tranche_ceiling_krw=450000,
            provider_factory=provider.factory,
        ).completed_count == 1
        completed_id, orphan_id = (
            unit.unit_id for unit in run.selected.package.production[:2]
        )
        stale = run.private.read_record(f"{completed_id}.parent.json")
    finally:
        run.close()
    provider.constructors = 0
    provider.requests.clear()
    (entrypoint_fixture.repository_root / "safety-run" / f"{orphan_id}.parent.json").write_bytes(
        stale
    )

    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE|MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        open_run(entrypoint_fixture, create=False, seed=1)

    assert (provider.constructors, len(provider.requests)) == (0, 0)


def test_missing_committed_parent_file_fails_on_reopen_before_provider(
    entrypoint_fixture, provider: FakeProvider
) -> None:
    run = open_run(entrypoint_fixture, create=True)
    try:
        assert run.execute(
            Path("unused"),
            max_units=1,
            tranche_ceiling_krw=450000,
            provider_factory=provider.factory,
        ).completed_count == 1
        unit_id = run.selected.package.production[0].unit_id
    finally:
        run.close()
    provider.constructors = 0
    provider.requests.clear()
    (entrypoint_fixture.repository_root / "safety-run" / f"{unit_id}.parent.json").unlink()

    with pytest.raises(ValueError):
        open_run(entrypoint_fixture, create=False)

    assert (provider.constructors, len(provider.requests)) == (0, 0)


def test_self_consistent_stale_identity_parent_fails_on_reopen_before_provider(
    entrypoint_fixture, provider: FakeProvider
) -> None:
    run = open_run(entrypoint_fixture, create=True)
    try:
        assert run.execute(
            Path("unused"),
            max_units=1,
            tranche_ceiling_krw=450000,
            provider_factory=provider.factory,
        ).completed_count == 1
        unit_id = run.selected.package.production[0].unit_id
        with run.private.connect() as connection:
            raw = connection.execute(
                "SELECT raw FROM parents WHERE unit_id=?", (unit_id,)
            ).fetchone()[0]
            payload = json.loads(raw)
            payload["identity"]["run_id"] = "foreign-run"
            malformed = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            connection.execute(
                "UPDATE parents SET raw=?, sha256=? WHERE unit_id=?",
                (malformed, hashlib.sha256(malformed).hexdigest(), unit_id),
            )
        (entrypoint_fixture.repository_root / "safety-run" / f"{unit_id}.parent.json").write_bytes(
            malformed
        )
    finally:
        run.close()
    provider.constructors = 0
    provider.requests.clear()

    with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        open_run(entrypoint_fixture, create=False)

    assert (provider.constructors, len(provider.requests)) == (0, 0)


def test_process_ownership_recovers_after_death(entrypoint_fixture, provider: FakeProvider) -> None:
    run = open_run(entrypoint_fixture, create=True)
    run.close()
    command = [sys.executable, "-c", "from pathlib import Path; from tests.phase13_runner_safety_fixture "
               "import launcher; import sys; launcher(Path(sys.argv[1]))", str(entrypoint_fixture.repository_root)]
    owner = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert owner.stdout is not None
        assert owner.stdout.readline().strip() == "OWNED"
        second = subprocess.run(command, input="\n", capture_output=True, text=True, timeout=60)
        assert second.returncode == 0, second.stderr
        assert second.stdout.strip() == "MAIN_RUN_ALREADY_OWNED"
        owner.kill()
        owner.wait(timeout=10)
        recovered = open_run(entrypoint_fixture, create=False)
        try:
            assert recovered.execute(Path("unused"), max_units=0, tranche_ceiling_krw=450000).provider_calls_issued == 0
        finally:
            recovered.close()
        assert (provider.constructors, len(provider.requests)) == (0, 0)
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=10)


@pytest.mark.parametrize("overflow", [0, 1])
def test_resume_uses_realized_plus_next_projection(entrypoint_fixture, provider: FakeProvider, overflow: int) -> None:
    run = open_run(entrypoint_fixture, create=True)
    try:
        assert run.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000,
                           provider_factory=provider.factory).completed_count == 1
        realized = run.ledger.realized_cost_krw()
        projection = run.selected.package.production[1].projected_cost_krw
        assert realized == 50
    finally:
        run.close()
    resumed = open_run(entrypoint_fixture, create=False, seed=1)
    try:
        before = resumed.ledger.rows()
        report = resumed.execute(Path("unused"), max_units=1, tranche_ceiling_krw=realized + projection - overflow,
                                 provider_factory=provider.factory)
        if overflow:
            assert report.session_state == "PAUSED_BEFORE_DISPATCH"
            assert resumed.ledger.rows() == before
            assert (provider.constructors, len(provider.requests)) == (50, 50)
        else:
            assert report.completed_count == 2
            assert (provider.constructors, len(provider.requests)) == (100, 100)
            assert len(set(provider.requests)) == 100
    finally:
        resumed.close()
    if overflow:
        reopened = open_run(entrypoint_fixture, create=False, seed=1)
        try:
            assert reopened.status().session_state == "PAUSED_BEFORE_DISPATCH"
            assert reopened.execute(Path("unused"), max_units=1, tranche_ceiling_krw=realized + projection,
                                    provider_factory=provider.factory).completed_count == 2
            assert (provider.constructors, len(provider.requests)) == (100, 100)
        finally:
            reopened.close()


def test_unknown_cost_pauses_durably_before_next_provider(entrypoint_fixture, provider: FakeProvider) -> None:
    from memcontam.readiness.phase13_v3_request import MessageV3, RequestKeyV3, RequestMaterialV3
    from memcontam.readiness.phase13_main_request_dispatch import DispatchTechnicalFailureV3
    from memcontam.clients.base import LLMResponse

    class UnknownProvider(FakeProvider):
        def send_compiled_v3(self, compiled, before_request):
            super().send_compiled_v3(compiled, before_request)
            return LLMResponse("final: 0", {}, {}, 0)

    unknown = UnknownProvider()
    run = open_run(entrypoint_fixture, create=True)
    try:
        key = RequestKeyV3(parent_id=run.selected.package.production[0].unit_id, stage="no_memory_generate", ordinal=0)
        with pytest.raises(DispatchTechnicalFailureV3):
            run.dispatcher(unknown.factory).dispatch(key, lambda: RequestMaterialV3(
                messages=(MessageV3(role="user", content="fixture"),), native_state=b"{}"), lambda response: None)
    finally:
        run.close()
    reopened = open_run(entrypoint_fixture, create=False, seed=1)
    try:
        before = reopened.ledger.rows()
        assert reopened.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000,
            provider_factory=provider.factory).session_state == "PAUSED_BEFORE_DISPATCH"
        assert reopened.ledger.rows() == before
        assert (provider.constructors, len(provider.requests)) == (0, 0)
        assert (unknown.constructors, len(unknown.requests)) == (1, 1)
    finally:
        reopened.close()
    reopened = open_run(entrypoint_fixture, create=False, seed=1)
    try:
        assert reopened.status().session_state == "PAUSED_BEFORE_DISPATCH"
    finally:
        reopened.close()


def test_independent_authorization_gate_cannot_be_enlarged(entrypoint_fixture, provider: FakeProvider) -> None:
    from memcontam.readiness.phase13_v3_request import MessageV3, RequestKeyV3, RequestMaterialV3
    from memcontam.readiness.phase13_main_request_dispatch import DispatchTechnicalFailureV3
    from memcontam.clients.base import LLMResponse

    class CostlyFailure(FakeProvider):
        def send_compiled_v3(self, compiled, before_request):
            super().send_compiled_v3(compiled, before_request)
            return LLMResponse("final: 0", {"authoritative_provider_cost_usd": "281.25", "currency": "USD"}, {}, 0)

    costly = CostlyFailure()
    run = open_run(entrypoint_fixture, create=True)
    try:
        key = RequestKeyV3(parent_id=run.selected.package.production[0].unit_id, stage="no_memory_generate", ordinal=0)
        with pytest.raises(DispatchTechnicalFailureV3):
            run.dispatcher(costly.factory).dispatch(key, lambda: RequestMaterialV3(
                messages=(MessageV3(role="user", content="fixture"),), native_state=b"{}"), lambda response: None)
        assert run.ledger.realized_cost_krw() == 450000
    finally:
        run.close()
    reopened = open_run(entrypoint_fixture, create=False, seed=1)
    try:
        assert reopened.execute(Path("unused"), max_units=1, tranche_ceiling_krw=1000000,
            provider_factory=provider.factory).session_state == "PAUSED_BEFORE_DISPATCH"
        assert (provider.constructors, len(provider.requests)) == (0, 0)
    finally:
        reopened.close()
