from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from memcontam.evaluation.phase13_observability_models import Phase13ObservabilityError
from memcontam.readiness.phase13_production_observability import ProductionObservabilityError
from memcontam.readiness.phase13_production_runtime_models import ProductionRuntimeJoinError

from .phase13_runner_safety_fixture import FakeProvider, open_run
from .test_phase13_runner_safety import entrypoint_bytes as entrypoint_bytes
from .test_phase13_runner_safety import provider as provider
from .test_phase13_runner_safety import entrypoint_fixture as entrypoint_fixture
from .test_phase13_runner_safety import local_authority as local_authority
from .test_phase13_runner_safety import source_selection as source_selection
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external


def test_reconstruction_failure_is_sanitized_durable_and_never_retried(
    entrypoint_fixture, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memcontam.readiness.phase13_main_live_runtime as runtime

    def reject(*_args):
        inner = Phase13ObservabilityError("ORDINARY_SEQUENCE_CONTINUITY_MISMATCH")
        inner.args = ("SECRET-response-credential-environment",)
        raise ProductionObservabilityError("PRODUCTION_RECONSTRUCTION_FAILED") from inner

    monkeypatch.setattr(runtime, "validate_production_archive", reject)
    run = open_run(entrypoint_fixture, create=True)
    try:
        with pytest.raises(ProductionObservabilityError):
            run.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000, provider_factory=provider.factory)
        with run.private.connect() as connection:
            raw, checksum = connection.execute("SELECT raw, sha256 FROM run_journal ORDER BY sequence DESC LIMIT 1").fetchone()
        payload = json.loads(raw)
        assert payload["outer_code"] == "PRODUCTION_RECONSTRUCTION_FAILED"
        assert payload["inner_code"] == "ORDINARY_SEQUENCE_CONTINUITY_MISMATCH"
        assert payload["provider_completed"] is True
        assert b"SECRET" not in raw
        assert len(checksum) == 64
        assert (provider.constructors, len(provider.requests)) == (50, 50)
    finally:
        run.close()
    reopened = open_run(entrypoint_fixture, create=False)
    try:
        assert reopened.status().session_state == "RECONSTRUCTION_FAILED"
        with pytest.raises(ValueError, match="MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"):
            reopened.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000, provider_factory=provider.factory)
        assert (provider.constructors, len(provider.requests)) == (50, 50)
    finally:
        reopened.close()
    for field, value in (("metadata", "SECRET"), ("run_id", "foreign"), ("task", "foreign"),
                         ("inner_code", "SECRET"), ("previous_hash", "0" * 64)):
        altered = {**payload, field: value}
        tampered = (json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n").encode()
        from memcontam.readiness.phase13_v3_entrypoint_paths import private_ledger
        with private_ledger(entrypoint_fixture.repository_root / "safety-run", create=False) as private:
            with private.connect() as connection:
                connection.execute("UPDATE run_journal SET raw=?, sha256=? WHERE sequence=2",
                                   (tampered, hashlib.sha256(tampered).hexdigest()))
        with pytest.raises(ValueError):
            invalid = open_run(entrypoint_fixture, create=False)
            invalid.close()
        with private_ledger(entrypoint_fixture.repository_root / "safety-run", create=False) as private:
            with private.connect() as connection:
                connection.execute("UPDATE run_journal SET raw=?, sha256=? WHERE sequence=2", (raw, checksum))


def test_archive_builder_failure_is_sanitized_after_provider_completion(
    entrypoint_fixture, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memcontam.readiness.phase13_main_live_runtime as runtime

    def reject(*_args):
        error = ProductionRuntimeJoinError("PRODUCTION_LINEAGE_PARENT_MISSING")
        error.args = ("SECRET-builder-payload",)
        raise error

    monkeypatch.setattr(runtime, "production_archive_from_ordinary", reject)
    run = open_run(entrypoint_fixture, create=True)
    try:
        with pytest.raises(ProductionObservabilityError):
            run.execute(
                Path("unused"),
                max_units=1,
                tranche_ceiling_krw=450000,
                provider_factory=provider.factory,
            )
        with run.private.connect() as connection:
            raw = connection.execute(
                "SELECT raw FROM run_journal ORDER BY sequence DESC LIMIT 1"
            ).fetchone()[0]
        payload = json.loads(raw)
        assert payload["inner_code"] == "PRODUCTION_LINEAGE_PARENT_MISSING"
        assert payload["provider_completed"] is True
        assert b"SECRET" not in raw
        assert (provider.constructors, len(provider.requests)) == (50, 50)
    finally:
        run.close()
    reopened = open_run(entrypoint_fixture, create=False)
    try:
        with pytest.raises(
            ValueError, match="MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"
        ):
            reopened.execute(
                Path("unused"),
                max_units=1,
                tranche_ceiling_krw=450000,
                provider_factory=provider.factory,
            )
        assert (provider.constructors, len(provider.requests)) == (50, 50)
    finally:
        reopened.close()


def test_parent_finalization_failure_is_durable_and_never_retried(
    entrypoint_fixture, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memcontam.readiness.phase13_main_v3_runner as runner
    from memcontam.readiness.phase13_main_live_evidence import MainEvidenceValidationError

    def reject(*_args):
        raise MainEvidenceValidationError("MAIN_UNIT_EVIDENCE_JOIN_INVALID")

    monkeypatch.setattr(runner, "validate_dispatch_evidence", reject)
    run = open_run(entrypoint_fixture, create=True)
    try:
        with pytest.raises(ProductionObservabilityError):
            run.execute(
                Path("unused"),
                max_units=1,
                tranche_ceiling_krw=450000,
                provider_factory=provider.factory,
            )
        with run.private.connect() as connection:
            raw = connection.execute(
                "SELECT raw FROM run_journal ORDER BY sequence DESC LIMIT 1"
            ).fetchone()[0]
        payload = json.loads(raw)
        assert payload["inner_code"] == "MAIN_UNIT_EVIDENCE_JOIN_INVALID"
        assert payload["provider_completed"] is True
        assert (provider.constructors, len(provider.requests)) == (50, 50)
    finally:
        run.close()
    reopened = open_run(entrypoint_fixture, create=False)
    try:
        with pytest.raises(
            ValueError, match="MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"
        ):
            reopened.execute(
                Path("unused"),
                max_units=1,
                tranche_ceiling_krw=450000,
                provider_factory=provider.factory,
            )
        assert (provider.constructors, len(provider.requests)) == (50, 50)
    finally:
        reopened.close()


def test_reconstruction_sink_failure_leaves_completed_requests_unrepeatable(
    entrypoint_fixture, provider: FakeProvider, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3
    import memcontam.readiness.phase13_main_live_runtime as runtime

    def reject(*_args):
        raise ProductionObservabilityError("PRODUCTION_RECONSTRUCTION_FAILED") from Phase13ObservabilityError("SECRET")

    monkeypatch.setattr(runtime, "validate_production_archive", reject)
    run = open_run(entrypoint_fixture, create=True)
    try:
        with run.private.connect() as connection:
            connection.execute("CREATE TRIGGER reject_telemetry BEFORE INSERT ON run_journal "
                "WHEN json_extract(NEW.raw, '$.kind')='RECONSTRUCTION_FAILED' "
                "BEGIN SELECT RAISE(ABORT, 'sink unavailable'); END")
        with pytest.raises(sqlite3.IntegrityError, match="sink unavailable"):
            run.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000, provider_factory=provider.factory)
        assert run.status().completed_count == 0
        assert (provider.constructors, len(provider.requests)) == (50, 50)
    finally:
        run.close()
    reopened = open_run(entrypoint_fixture, create=False)
    try:
        with pytest.raises(ValueError, match="MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"):
            reopened.execute(Path("unused"), max_units=1, tranche_ceiling_krw=450000, provider_factory=provider.factory)
        assert (provider.constructors, len(provider.requests)) == (50, 50)
    finally:
        reopened.close()
