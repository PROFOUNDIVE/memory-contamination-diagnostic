from __future__ import annotations

import hashlib
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import memcontam.cli as cli
import pytest

from memcontam.experiment.phase12.contracts import (
    CodeMatrixPlan,
    ExploratoryActivationManifest,
    FidelityCertificate,
    MftManifest,
    Phase12IntegrationCertificate,
    PilotBManifest,
    RouteFeasibilityReport,
    RouteSelectionManifest,
    SeedAllocationManifest,
    SelectedPackageResourceManifest,
    canonical_json_hash,
)
from memcontam.experiment.phase12.planner import (
    validate_exploratory_activation,
    validate_route_selection,
)
from memcontam.manifests.archive_validation import ArchiveValidationReport


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase12"


def _admission_module():
    return importlib.import_module("memcontam.readiness.scientific_admission")


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["memcontam", *args])
    cli.main()


def _certificates(tmp_path: Path, fixture_name: str):
    module = _admission_module()
    fixture = _fixture(fixture_name)
    bfv2 = FidelityCertificate(
        certificate_id=f"bfv2-{fixture['fixture_id']}",
        protocol_version="baseline_fidelity_v2",
        git_commit=fixture["repository_commit"],
        overall_status=fixture["bfv2"]["f1c"],
        issued_at="2026-07-23T00:00:00Z",
    )
    resolved_config = {"fixture_id": fixture["fixture_id"]}
    public_manifest = {"fixture_id": fixture["fixture_id"], "status": "completed"}
    gates = []
    gate_hashes = {}
    for gate_id in fixture["gates"]:
        path = tmp_path / f"{fixture['fixture_id']}-{gate_id}.json"
        path.write_text(json.dumps({"gate_id": gate_id}), encoding="utf-8")
        evidence_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        gates.append(
            {
                "gate_id": gate_id,
                "status": "pass",
                "reason_code": None,
                "evidence_path": str(path),
                "evidence_hash": evidence_hash,
            }
        )
        gate_hashes[f"{gate_id}_gate"] = evidence_hash
    p12i = Phase12IntegrationCertificate(
        certificate_id=f"p12i-{fixture['fixture_id']}",
        protocol_version="phase12_integration_v1",
        git_commit=fixture["repository_commit"],
        resolved_config_hash=canonical_json_hash(resolved_config),
        public_artifact_manifest_hash=canonical_json_hash(public_manifest),
        bfv2_certificate_id=bfv2.certificate_id,
        bfv2_certificate_hash=canonical_json_hash(bfv2.model_dump(mode="json")),
        **gate_hashes,
        overall_status="pass",
        issued_at="2026-07-23T00:00:00Z",
    )
    return module.CertificateBundle(
        bfv2_certificate=bfv2,
        p12i_certificate=p12i,
        p12i_artifacts={
            "repository_commit": fixture["repository_commit"],
            "resolved_config": resolved_config,
            "public_artifact_manifest": public_manifest,
            "f1c_status": fixture["bfv2"]["f1c"],
            "f1c_contract_refs": fixture["f1c_contract_refs"],
            "subgates": gates,
        },
    )


def _write_admission_bundle(
    tmp_path: Path, fixture_name: str, *, exploratory: bool = False
) -> Path:
    certificates = _certificates(tmp_path, fixture_name)
    payload = {
        "bfv2_certificate": certificates.bfv2_certificate.model_dump(mode="json"),
        "p12i_certificate": certificates.p12i_certificate.model_dump(mode="json"),
        "p12i_artifacts": dict(certificates.p12i_artifacts),
        "archive_root": str(tmp_path / "archive"),
        "trajectory_seed": 7,
        "abstract_seed_slot": None,
    }
    if exploratory:
        route = _fixture("FX-ROUTE-001.json")
        resource = deepcopy(route["valid_selected_package_resource_manifest"])
        resource["exploratory_call_budget"] = 0
        payload.update(
            {
                "trajectory_seed": next(
                    iter(route["valid_exploratory_activation_manifest"]["exploratory_slot_to_seed"].values())
                ),
                "abstract_seed_slot": next(
                    iter(route["valid_exploratory_activation_manifest"]["exploratory_slot_to_seed"])
                ),
                "feasibility_reports": route["feasibility_reports"],
                "pilot_b_manifest": route["valid_pilot_b_manifest"],
                "mft_manifest": route["valid_mft_manifest"],
                "route_selection_manifest": route["valid_external_selection_manifest"],
                "seed_allocation_manifest": route["valid_seed_allocation_manifest"],
                "exploratory_plan": route["valid_exploratory_plan"],
                "selected_package_resource_manifest": resource,
                "exploratory_activation_manifest": route["valid_exploratory_activation_manifest"],
            }
        )
    path = tmp_path / f"{fixture_name}.bundle.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _validated_governance():
    route = _fixture("FX-ROUTE-001.json")
    reports = tuple(RouteFeasibilityReport.model_validate(item) for item in route["feasibility_reports"])
    pilot = PilotBManifest.model_validate(route["valid_pilot_b_manifest"])
    mft = MftManifest.model_validate(route["valid_mft_manifest"])
    selection = RouteSelectionManifest.model_validate(route["valid_external_selection_manifest"])
    allocation = SeedAllocationManifest.model_validate(route["valid_seed_allocation_manifest"])
    validated_route = validate_route_selection(reports, pilot, mft, selection, allocation)
    plan = CodeMatrixPlan.model_validate(route["valid_exploratory_plan"])
    resource = SelectedPackageResourceManifest.model_validate(
        route["valid_selected_package_resource_manifest"]
    )
    activation = ExploratoryActivationManifest.model_validate(
        route["valid_exploratory_activation_manifest"]
    )
    return selection, allocation, activation, validated_route, validate_exploratory_activation(
        plan, resource, activation, validated_route
    )


def test_admits_applicable_pre_route_selected_route_and_exploratory_pass_fixtures_without_paid_calls(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _admission_module()
    certificates = _certificates(tmp_path, "FX-P12I-PASS-001.json")
    archive = ArchiveValidationReport(True, _fixture("FX-ARCHIVE-001.json")["expected"]["resolved_edges"])
    selection, allocation, activation, validated_route, validated_activation = _validated_governance()

    pilot = module.evaluate_scientific_admission(
        module.ScientificRunRequest("pilot_a", "3w", "text_only", True, 7, None),
        certificates,
        archive,
        None,
        None,
    )
    main_slot, main_seed = next(iter(allocation.slot_to_seed.items()))
    main = module.evaluate_scientific_admission(
        module.ScientificRunRequest(
            "main_a",
            "3w",
            "text_only",
            True,
            main_seed,
            main_slot,
            selection.manifest_id,
            allocation.manifest_id,
        ),
        certificates,
        archive,
        validated_route,
        None,
    )
    exploratory_slot, exploratory_seed = next(iter(activation.exploratory_slot_to_seed.items()))
    exploratory = module.evaluate_scientific_admission(
        module.ScientificRunRequest(
            "exploratory_code",
            "3w",
            "python_sandbox",
            True,
            exploratory_seed,
            exploratory_slot,
            selection.manifest_id,
            allocation.manifest_id,
            activation.manifest_id,
        ),
        certificates,
        archive,
        validated_route,
        validated_activation,
    )

    assert pilot.scientific_admission_ref["p12i_certificate_id"] == certificates.p12i_certificate.certificate_id
    assert main.scientific_admission_ref == pilot.scientific_admission_ref
    assert exploratory.scientific_admission_ref == pilot.scientific_admission_ref
    assert not (tmp_path / "runs").exists()

    phase12_cli = importlib.import_module("memcontam.experiment.phase12.cli")
    monkeypatch.setattr(phase12_cli, "validate_archive", lambda _: ArchiveValidationReport(True, 11))
    run_root = tmp_path / "cli-runs"
    _run_cli(
        monkeypatch,
        "phase12",
        "run-prefix",
        "--scientific",
        "--admission-only",
        "--run-family",
        "pilot_a",
        "--admission-bundle",
        str(_write_admission_bundle(tmp_path, "FX-P12I-PASS-001.json")),
        "--run-root",
        str(run_root),
    )
    assert json.loads(capsys.readouterr().out)["admitted"] is True
    assert not run_root.exists()


def test_blocks_every_incomplete_or_blocked_fixture(tmp_path: Path, monkeypatch) -> None:
    module = _admission_module()
    archive = ArchiveValidationReport(True, 11)
    blocked = _certificates(tmp_path, "FX-P12I-001.json")
    passed = _certificates(tmp_path, "FX-P12I-PASS-001.json")
    selection, allocation, activation, validated_route, validated_activation = _validated_governance()

    cases = (
        (
            module.ScientificRunRequest("pilot_a", "3w", "text_only", True, 7, None),
            blocked,
            None,
            None,
            "F1C_NOT_PASS",
        ),
        (
            module.ScientificRunRequest("main_a", "3w", "text_only", True, 7, None),
            passed,
            None,
            None,
            "ROUTE_SELECTION_REQUIRED",
        ),
        (
            module.ScientificRunRequest(
                "main_a", "3w", "text_only", True, 7, None, selection.manifest_id
            ),
            passed,
            None,
            None,
            "SEED_ALLOCATION_REQUIRED",
        ),
        (
            module.ScientificRunRequest(
                "exploratory_code",
                "3w",
                "python_sandbox",
                True,
                7,
                None,
                selection.manifest_id,
                allocation.manifest_id,
            ),
            passed,
            validated_route,
            None,
            "EXPLORATORY_ACTIVATION_REQUIRED",
        ),
    )
    for request, certificates, route, activation_result, code in cases:
        with pytest.raises(module.AdmissionDenied, match=code):
            module.evaluate_scientific_admission(
                request, certificates, archive, route, activation_result
            )

    exploratory_slot, exploratory_seed = next(iter(activation.exploratory_slot_to_seed.items()))
    request = module.ScientificRunRequest(
        "exploratory_code",
        "3w",
        "python_sandbox",
        True,
        exploratory_seed,
        exploratory_slot,
        selection.manifest_id,
        allocation.manifest_id,
        activation.manifest_id,
    )
    underfunded = validated_activation.model_copy(
        update={"estimated_exploratory_calls": 2, "exploratory_call_budget": 1}
    )
    insufficient_reserve = validated_activation.model_copy(
        update={
            "estimated_exploratory_calls": 0,
            "exploratory_call_budget": 1,
            "reproducibility_reserve": 1,
            "remaining_call_capacity": 1,
        }
    )
    for activation_result, code in (
        (underfunded, "EXPLORATORY_BUDGET_INSUFFICIENT"),
        (insufficient_reserve, "REPRODUCIBILITY_RESERVE_INSUFFICIENT"),
    ):
        with pytest.raises(module.AdmissionDenied, match=code):
            module.evaluate_scientific_admission(
                request, passed, archive, validated_route, activation_result
            )

    from memcontam.experiment.phase12 import cli as phase12_cli

    route_fixture = _fixture("FX-ROUTE-001.json")
    unfunded_payload = {
        "exploratory_activation_manifest": route_fixture["valid_exploratory_activation_manifest"],
        "exploratory_plan": route_fixture["valid_exploratory_plan"],
        "selected_package_resource_manifest": deepcopy(
            route_fixture["valid_selected_package_resource_manifest"]
        ),
    }
    unfunded_payload["selected_package_resource_manifest"]["mandatory_package_status"] = "not_resourced"
    with pytest.raises(module.AdmissionDenied, match="EXPLORATORY_RESOURCE_RESERVATION_NOT_PASS"):
        phase12_cli._validated_activation(unfunded_payload, validated_route)

    monkeypatch.setattr(phase12_cli, "validate_archive", lambda _: ArchiveValidationReport(True, 11))
    run_root = tmp_path / "cli-runs"
    with pytest.raises(SystemExit, match="EXPLORATORY_BUDGET_INSUFFICIENT"):
        _run_cli(
            monkeypatch,
            "phase12",
            "run-prefix",
            "--scientific",
            "--admission-only",
            "--run-family",
            "exploratory_code",
            "--mode",
            "python_sandbox",
            "--admission-bundle",
            str(_write_admission_bundle(tmp_path, "FX-P12I-PASS-001.json", exploratory=True)),
            "--run-root",
            str(run_root),
        )
    assert not run_root.exists()

    with pytest.raises(SystemExit, match="ADMISSION_EVIDENCE_FORBIDDEN"):
        _run_cli(
            monkeypatch,
            "phase12",
            "run-prefix",
            "--admission-only",
            "--admission-bundle",
            str(_write_admission_bundle(tmp_path, "FX-P12I-PASS-001.json")),
            "--run-root",
            str(run_root),
        )
    assert not run_root.exists()

    assert not (tmp_path / "runs").exists()
