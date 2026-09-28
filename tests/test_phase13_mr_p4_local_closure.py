from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness import phase13_main_readiness
from memcontam.readiness.phase13_main_readiness import (
    Phase13MainReadinessError,
    validate_main_readiness,
)
from memcontam.readiness.phase13_observability_models import Phase13ObservabilityFixture
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityArchive,
    ProductionTrialRecord,
    ProviderRequestRecord,
    conformance_archive,
)
from memcontam.readiness.phase13_v3_builder import validate_mr_p4
from memcontam.readiness.phase13_v3_builder_inputs import production
from memcontam.readiness.phase13_v3_conformance import ConformanceV3
from memcontam.readiness.phase13_v3_cost_models import canonical_bytes, digest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/phase13/main/mr_p4"
pytest_plugins = ("tests.test_phase13_v3_artifact_builder",)
Staged = tuple[ModuleType, Path, Path, Path]


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


def test_local_mr_p4_package_materializes_every_policy_fixed_registry(staged: Staged) -> None:
    _, root, output, authority = staged
    report = validate_mr_p4(root, authority, output)
    units = production(report.first_freeze, report.resources)
    conformance = ConformanceV3.model_validate_json(
        (output / "mr_p4/corrected_v3/provider_free_conformance_v3.json").read_bytes()
    )

    assert len({unit.execution_template_id for unit in units if unit.kind != "CLEAN_PREFIX"}) == 97
    assert len({(unit.task, unit.memory_baseline) for unit in units if unit.kind == "CLEAN_PREFIX"}) == 23
    assert len({(unit.task, unit.memory_baseline) for unit in units
        if unit.kind == "CLEAN_PREFIX" and unit.memory_baseline != "fh_bounded"}) == 18
    assert report.first_freeze.concrete_seed_ids == tuple(range(10))
    assert all(task.H_run == 50 for task in report.first_freeze.registry.tasks.values())
    assert all(predicate.passed for predicate in conformance.predicates)
    assert report.authority.registry.default_max_transport_attempts == 1
    assert report.authority.registry.entitled_eligible_max_transport_attempts == 2
    assert report.authority.registry.maximum_retries_after_initial_attempt == 1
    assert conformance.measured_trajectories == 0
    assert _conformance_archive().u_t_status == "NOT_REGISTERED_FOR_CURRENT_MAIN"


def test_local_mr_p4_package_closes_after_non_scientific_live_readiness(staged: Staged) -> None:
    _, root, output, authority = staged
    report = validate_mr_p4(root, authority, output)
    conformance = ConformanceV3.model_validate_json(
        (output / "mr_p4/corrected_v3/provider_free_conformance_v3.json").read_bytes()
    )

    assert report.status == "CLOSED"
    assert all(predicate.passed for predicate in conformance.predicates)
    assert conformance.real_provider_calls == 0
    assert conformance.scientific_result is False
    assert conformance.measured_trajectories == 0
    assert output.is_dir()
    assert not (output / "mr_p5").exists()
    assert not (output / "mr_p6").exists()
    assert not (output / report.identity.run_id).exists()


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
        "readiness0_live_evidence_manifest",
        "readiness0_live_evidence_cases",
    } <= set(manifest["artifacts"])


def test_historical_readiness0_request_remains_stale_provenance_not_current_status(staged: Staged) -> None:
    historical = json.loads(
        (PACKAGE / "readiness0_request_v1.json").read_text(encoding="utf-8")
    )
    _, root, output, authority = staged
    report = validate_mr_p4(root, authority, output)

    assert historical["external_blockers"] == [
        "OPENAI_API_KEY_MISSING",
        "F1C_RUNTIME_ENVIRONMENT_NOT_CONFIGURED",
    ]
    assert report.status == "CLOSED"
    assert all("readiness0_request_v1.json" not in row.path for row in report.resources)


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
    staged: Staged,
) -> None:
    _, root, output, authority = staged
    manifest = validate_mr_p4(root, authority, output)
    status_path = output / "mr_p4/corrected_v3/provider_free_conformance_v3.json"
    status = ConformanceV3.model_validate_json(status_path.read_bytes()).model_copy(
        update={"real_provider_calls": 1}
    )
    status = status.model_copy(update={"conformance_hash": digest(status, "conformance_hash")})
    tampered = canonical_bytes(status)
    status_path.write_bytes(tampered)
    bindings = tuple(row.model_copy(update={"sha256": hashlib.sha256(tampered).hexdigest(),
        "size": len(tampered)}) if row.path.endswith(status_path.name) else row for row in manifest.artifacts)
    manifest = manifest.model_copy(update={"artifacts": bindings})
    manifest = manifest.model_copy(update={"closure_hash": digest(manifest, "closure_hash")})
    (output / "mr_p4/corrected_v3/manifest_v3.json").write_bytes(canonical_bytes(manifest))

    with pytest.raises(ValueError, match="MAIN_PROVIDER_FREE_CONFORMANCE_FAILED"):
        validate_mr_p4(root, authority, output)


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
            parsed_answer=record.parsed_answer,
            method_calls=record.method_calls,
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
    staged: Staged,
    artifact_name: str,
    internal_hash: str,
    mutation: tuple[str | None, str, JsonValue],
    expected_error: str,
) -> None:
    _, root, output, authority = staged
    manifest = validate_mr_p4(root, authority, output).model_dump(mode="json")
    _, field, value = mutation
    match field:
        case "attempted_seed_count_per_task":
            manifest["first_freeze"]["concrete_seed_ids"] = list(range(11))
            rejection = "MAIN_MR_P4_FIRST_FREEZE_MISMATCH"
        case "schema_version":
            manifest["schema_version"] = "phase13_mr_p4_local_closure_manifest_v2"
            rejection = "MAIN_ARTIFACT_BINDING_MISMATCH"
        case "current_sha256":
            router = next(row for row in manifest["authority"]["documents"] if row["role"] == "router")
            router["sha256"] = value
            rejection = "MAIN_AUTHORITY_BINDING_MISMATCH"
        case "H_run":
            manifest["first_freeze"]["registry"]["tasks"]["game24"]["H_run"] = value
            rejection = "MAIN_ARTIFACT_BINDING_MISMATCH"
        case "package_id":
            manifest["identity"]["package_id"] = value
            rejection = "MAIN_ARTIFACT_BINDING_MISMATCH"
        case _:
            pytest.fail(f"unmapped historical mutation: {artifact_name}/{internal_hash}/{expected_error}")
    manifest["closure_hash"] = hashlib.sha256((json.dumps(
        {key: value for key, value in manifest.items() if key != "closure_hash"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ) + "\n").encode()).hexdigest()
    (output / "mr_p4/corrected_v3/manifest_v3.json").write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    )

    with pytest.raises(ValueError, match=rejection):
        validate_mr_p4(root, authority, output)


def test_phase13_cli_exposes_main_readiness_validation(
    staged: Staged,
) -> None:
    _, root, output, authority = staged
    result = subprocess.run(
        ("bash", "-c", 'source .omo/evidence/phase13_shell_contract.sh; phase13_python "$@"',
         "--", str(ROOT / "scripts/build_phase13_corrected_main_closure.py"), "validate",
         "--repository-root", str(root), "--authority-root", str(authority), "--stage", "mr-p4",
         "--artifact", str(output / "mr_p4/corrected_v3/manifest_v3.json")),
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload == {"status": "CLOSED", "provider_calls": 0, "measured_trajectories": 0}
    assert not (output / "mr_p6").exists()
