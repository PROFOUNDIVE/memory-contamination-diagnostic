from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness import phase13_main_readiness
from memcontam.readiness.phase13_cli import add_parser, run
from memcontam.readiness.phase13_main_readiness import (
    Phase13MainReadinessError,
    validate_main_readiness,
)
from memcontam.readiness.phase13_main_readiness_models import MainReadinessReport
from memcontam.readiness.phase13_observability_models import Phase13ObservabilityFixture
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityArchive,
    ProductionTrialRecord,
    ProviderRequestRecord,
    conformance_archive,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/phase13/main/mr_p4"


def _manifest_sha256(root: Path = PACKAGE) -> str:
    return hashlib.sha256((root / "manifest_v1.json").read_bytes()).hexdigest()


def _canonical_hash(value: dict[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_rehashed_manifest(package: Path, manifest: dict[str, JsonValue]) -> None:
    manifest["closure_hash"] = _canonical_hash(
        {key: value for key, value in manifest.items() if key != "closure_hash"}
    )
    (package / "manifest_v1.json").write_text(json.dumps(manifest), encoding="utf-8")


def _conformance_archive() -> ProductionObservabilityArchive:
    manifest = json.loads((PACKAGE / "manifest_v1.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    packet_raw = (ROOT / artifacts["observability_packet"]["path"]).read_bytes()
    fixture_raw = (ROOT / artifacts["observability_fixture"]["path"]).read_bytes()
    ObservabilityRegistrationPacket.model_validate_json(packet_raw)
    fixture = Phase13ObservabilityFixture.model_validate_json(fixture_raw)
    return conformance_archive(fixture, hashlib.sha256(packet_raw).hexdigest())


def test_local_mr_p4_package_materializes_every_policy_fixed_registry() -> None:
    report: MainReadinessReport = validate_main_readiness(PACKAGE, ROOT, _manifest_sha256())

    assert report.execution_template_count == 97
    assert report.level2_interaction_count == 18
    assert report.abstract_seed_slots_per_task == 10
    assert report.H_run == report.H_primary == 50
    assert report.synthetic_observability_conformance_status == "PASS"
    assert report.provider_session_retry_resource_contract_status == "PASS"
    assert report.u_t_status == "NOT_REGISTERED_FOR_CURRENT_MAIN"
    assert report.main_a_measured_scientific_execution_count == 0


def test_local_mr_p4_package_fails_closed_only_on_live_external_dependencies() -> None:
    report: MainReadinessReport = validate_main_readiness(PACKAGE, ROOT, _manifest_sha256())

    assert report.status == "READINESS0_LIVE_EXTERNAL_DEPENDENCY_BLOCKED"
    assert report.blockers == (
        "OPENAI_API_KEY_MISSING",
        "READINESS0_REAUTHORIZATION_REQUIRED",
    )
    assert report.f1c_status == "PASS"
    assert report.provider_calls_issued == 0
    assert report.output_directory_created is False
    assert report.scientific_result is False
    assert report.main_result is False
    assert report.mr_p4_status == "OPEN"
    assert report.mr_p4_closure_claimed is False
    assert report.mr_p5_status == "NOT_STARTED"
    assert report.mr_p6_status == "NOT_AUTHORIZED"
    assert report.main_a_status == "NOT_STARTED"
    assert report.main_execution_authorized is False


def test_mr_p4_manifest_binds_first_frozen_checkpoint_identities() -> None:
    manifest = json.loads((PACKAGE / "manifest_v1.json").read_text(encoding="utf-8"))

    assert manifest["execution_templates"]["concrete_seed_registry_status"] == (
        "CONCRETE_MAIN_SEED_REGISTRY_FROZEN"
    )
    assert {"task_seed_orders", "common_checkpoint_registry"} <= set(manifest["artifacts"])
    assert manifest["gates"]["tau_star_status"] == "PASS"


def test_mr_p4_manifest_binds_direct_safety_dependencies() -> None:
    manifest = json.loads((PACKAGE / "manifest_v1.json").read_text(encoding="utf-8"))

    assert {
        "provider_profile",
        "cost_policy_models",
        "cost_policy_handoff",
    } <= set(manifest["artifacts"])


def test_mr_p4_manifest_binds_current_readiness0_attempt_artifacts() -> None:
    manifest = json.loads((PACKAGE / "manifest_v1.json").read_text(encoding="utf-8"))

    assert {
        "readiness0_live_request",
        "readiness0_live_authorization",
        "readiness0_f1c_registry",
        "readiness0_current_status",
    } <= set(manifest["artifacts"])


def test_historical_readiness0_request_remains_stale_provenance_not_current_status() -> None:
    historical = json.loads(
        (PACKAGE / "readiness0_request_v1.json").read_text(encoding="utf-8")
    )
    report: MainReadinessReport = validate_main_readiness(PACKAGE, ROOT, _manifest_sha256())

    assert historical["external_blockers"] == [
        "OPENAI_API_KEY_MISSING",
        "F1C_RUNTIME_ENVIRONMENT_NOT_CONFIGURED",
    ]
    assert report.blockers == (
        "OPENAI_API_KEY_MISSING",
        "READINESS0_REAUTHORIZATION_REQUIRED",
    )
    assert report.f1c_status == "PASS"


def test_mr_p4_current_status_hash_binding_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads((PACKAGE / "manifest_v1.json").read_text(encoding="utf-8"))
    identity = manifest["artifacts"]["readiness0_current_status"]
    status_path = ROOT / identity["path"]
    tampered = status_path.read_bytes() + b" "
    original_read = phase13_main_readiness.read_regular_nofollow

    def read_with_tampered_status(path: Path) -> bytes:
        return tampered if path == status_path else original_read(path)

    monkeypatch.setattr(phase13_main_readiness, "read_regular_nofollow", read_with_tampered_status)
    with pytest.raises(Phase13MainReadinessError, match="MR_P4_ARTIFACT_HASH_MISMATCH"):
        validate_main_readiness(PACKAGE, ROOT, _manifest_sha256())


def test_mr_p4_current_status_semantic_tamper_fails_after_hash_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "mr_p4"
    shutil.copytree(PACKAGE, package)
    manifest = json.loads((package / "manifest_v1.json").read_text(encoding="utf-8"))
    identity = manifest["artifacts"]["readiness0_current_status"]
    status_path = ROOT / identity["path"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["provider_calls_issued"] = 1
    status["status_hash"] = _canonical_hash(
        {key: value for key, value in status.items() if key != "status_hash"}
    )
    tampered = json.dumps(status).encode()
    identity["sha256"] = hashlib.sha256(tampered).hexdigest()
    _write_rehashed_manifest(package, manifest)
    original_read = phase13_main_readiness.read_regular_nofollow

    def read_with_tampered_status(path: Path) -> bytes:
        return tampered if path == status_path else original_read(path)

    monkeypatch.setattr(phase13_main_readiness, "read_regular_nofollow", read_with_tampered_status)
    with pytest.raises(Phase13MainReadinessError, match="MR_P4_PREREQUISITE_INVALID"):
        validate_main_readiness(package, ROOT, _manifest_sha256(package))


def test_production_contract_rejects_stateful_provider_continuation() -> None:
    request = _conformance_archive().records[0].request.model_dump(mode="json")
    request["previous_response_id"] = "response-from-another-trial"

    with pytest.raises(ValidationError):
        ProviderRequestRecord.model_validate(request)


def test_production_contract_uses_single_transport_attempt() -> None:
    archive = _conformance_archive()
    manifest = json.loads((PACKAGE / "manifest_v1.json").read_text(encoding="utf-8"))

    assert archive.records[0].request.retries_after_initial_attempt == 0
    assert manifest["provider_runtime_contract"]["retries_after_initial_attempt"] == 0


def test_production_contract_rejects_cross_trial_session_reuse() -> None:
    archive = _conformance_archive()
    records = list(archive.records)
    records[1] = records[1].model_copy(update={"session_id": records[0].session_id})

    with pytest.raises(ValidationError, match="CROSS_TRIAL_SESSION_REUSE"):
        ProductionObservabilityArchive(
            schema_version=archive.schema_version,
            registration_packet_sha256=archive.registration_packet_sha256,
            u_t_status=archive.u_t_status,
            records=tuple(records),
        )


def test_production_contract_rejects_mismatched_run_join() -> None:
    record = _conformance_archive().records[0]

    with pytest.raises(ValidationError, match="PRODUCTION_RUN_JOIN_MISMATCH"):
        ProductionTrialRecord(
            execution_template_id=record.execution_template_id,
            run_id="different-run",
            session_id=record.session_id,
            scientific_result=record.scientific_result,
            ordered_sample_ids_sha256=record.ordered_sample_ids_sha256,
            request=record.request,
            evidence=record.evidence,
            terminal_provider_evidence=record.terminal_provider_evidence,
        )


def test_mr_p4_manifest_tamper_fails_even_with_refreshed_outer_hash(tmp_path: Path) -> None:
    package = tmp_path / "mr_p4"
    shutil.copytree(PACKAGE, package)
    manifest_path = package / "manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_templates"]["H_run"] = 49
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Phase13MainReadinessError, match="MR_P4_EXECUTION_CONTRACT_MISMATCH"):
        validate_main_readiness(package, ROOT, _manifest_sha256(package))


def test_mr_p4_manifest_rejects_duplicate_pair_after_all_hashes_are_refreshed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "mr_p4"
    shutil.copytree(PACKAGE, package)
    manifest = json.loads((package / "manifest_v1.json").read_text(encoding="utf-8"))
    pairs = manifest["execution_templates"]["included_task_baseline_pairs"]
    pairs.append(pairs[0])
    _write_rehashed_manifest(package, manifest)

    with pytest.raises(Phase13MainReadinessError, match="MR_P4_EXECUTION_CONTRACT_MISMATCH"):
        validate_main_readiness(package, ROOT, _manifest_sha256(package))


def test_mr_p4_manifest_rejects_call_ceiling_after_all_hashes_are_refreshed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "mr_p4"
    shutil.copytree(PACKAGE, package)
    manifest = json.loads((package / "manifest_v1.json").read_text(encoding="utf-8"))
    manifest["execution_templates"]["call_ceilings"]["fh_bounded"] = {
        "nominal": 0,
        "maximum": 0,
    }
    _write_rehashed_manifest(package, manifest)

    with pytest.raises(Phase13MainReadinessError, match="MR_P4_EXECUTION_CONTRACT_MISMATCH"):
        validate_main_readiness(package, ROOT, _manifest_sha256(package))


@pytest.mark.parametrize(
    ("artifact_name", "internal_hash", "mutation", "expected_error"),
    [
        (
            "track1",
            "checkpoint_hash",
            ("completed_repository_sync", "attempted_seed_count_per_task", 11),
            "MR_P4_TRACK1_CONTRACT_MISMATCH",
        ),
        (
            "track1",
            "checkpoint_hash",
            (None, "schema_version", "phase13_track1_authority_state_sync_checkpoint_v2"),
            "MR_P4_TRACK1_CONTRACT_MISMATCH",
        ),
        (
            "track1",
            "checkpoint_hash",
            ("authority_router", "current_sha256", "0" * 64),
            "MR_P4_TRACK1_CONTRACT_MISMATCH",
        ),
        (
            "package_selection",
            "package_hash",
            ("selected_current_main", "H_run", 49),
            "MR_P4_PACKAGE_SELECTION_MISMATCH",
        ),
        (
            "package_selection",
            "package_hash",
            ("selected_current_main", "package_id", "substituted_package"),
            "MR_P4_PACKAGE_SELECTION_MISMATCH",
        ),
    ],
)
def test_mr_p4_rejects_semantic_artifact_tamper_after_all_hashes_are_refreshed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    internal_hash: str,
    mutation: tuple[str | None, str, JsonValue],
    expected_error: str,
) -> None:
    package = tmp_path / "mr_p4"
    shutil.copytree(PACKAGE, package)
    manifest = json.loads((package / "manifest_v1.json").read_text(encoding="utf-8"))
    identity = manifest["artifacts"][artifact_name]
    artifact_path = ROOT / identity["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    section, field, value = mutation
    if section is None:
        artifact[field] = value
    else:
        artifact[section][field] = value
    artifact[internal_hash] = _canonical_hash(
        {key: value for key, value in artifact.items() if key != internal_hash}
    )
    tampered_raw = json.dumps(artifact).encode()
    identity["sha256"] = hashlib.sha256(tampered_raw).hexdigest()
    _write_rehashed_manifest(package, manifest)
    original_read = phase13_main_readiness.read_regular_nofollow

    def read_with_tampered_artifact(path: Path) -> bytes:
        return tampered_raw if path == artifact_path else original_read(path)

    monkeypatch.setattr(phase13_main_readiness, "read_regular_nofollow", read_with_tampered_artifact)
    with pytest.raises(Phase13MainReadinessError, match=expected_error):
        validate_main_readiness(package, ROOT, _manifest_sha256(package))


def test_phase13_cli_exposes_main_readiness_validation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_parser(subparsers)
    args = parser.parse_args(
        [
            "phase13",
            "validate-main-readiness",
            "--repository-root",
            str(ROOT),
            "--root",
            str(PACKAGE),
            "--expected-manifest-sha256",
            _manifest_sha256(),
        ]
    )

    run(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "READINESS0_LIVE_EXTERNAL_DEPENDENCY_BLOCKED"
    assert payload["mr_p4_closure_claimed"] is False
    assert payload["main_execution_authorized"] is False
