from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import pytest

from memcontam.readiness.phase13_cli import add_parser, run
from memcontam.readiness.phase13_observability_models import Phase13ObservabilityFixture


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "data/phase13/observability"


def _validator():
    return importlib.import_module("memcontam.readiness.phase13_observability_validate")


def _manifest_sha256(root: Path = PACKAGE) -> str:
    return hashlib.sha256((root / "manifest_v1.json").read_bytes()).hexdigest()


def test_published_observability_package_is_deterministic_and_track2_5_complete() -> None:
    report = _validator().validate_phase13_observability_package(
        PACKAGE,
        ROOT,
        _manifest_sha256(),
    )

    assert report.track2_5_status == "TRACK2_5_NOVELTY_OBSERVABILITY_COMPLETE"
    assert report.evidence_scope == "synthetic_contract_fixture_only"
    assert report.mr_p4_prerequisite_status == "OBSERVABILITY_PREREQUISITE_MET"
    assert report.mr_p5_handoff_status == "MEASUREMENT_IDENTITY_HANDOFF_CLOSED"
    assert report.u_t_status == "NOT_REGISTERED_FOR_CURRENT_MAIN"
    assert report.main_a_measured_scientific_execution_count == 0
    assert report.reconstructed_trial_count == 5
    assert report.reconstruction_sha256 == report.repeat_reconstruction_sha256
    assert report.downstream_blockers == ()
    assert report.mr_p4_closure_claimed is False
    assert report.mr_p5_closure_claimed is False
    assert report.main_execution_authorized is False
    _validator().require_mr_p4_observability(report)


def test_observability_package_fails_closed_when_runtime_evidence_is_tampered(
    tmp_path: Path,
) -> None:
    package = tmp_path / "observability"
    shutil.copytree(PACKAGE, package)
    fixture_path = package / "fixture_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["trials"][0]["memory_before_ids"] = []
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(
        _validator().Phase13ObservabilityValidationError,
        match="OBSERVABILITY_ARTIFACT_HASH_MISMATCH",
    ):
        _validator().validate_phase13_observability_package(
            package,
            ROOT,
            _manifest_sha256(package),
        )


def test_packet_verifier_hash_tamper_fails_after_outer_hashes_are_refreshed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "observability"
    shutil.copytree(PACKAGE, package)
    packet_path = package / "registration_packet_v1.json"
    fixture_path = package / "fixture_v1.json"
    manifest_path = package / "manifest_v1.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    packet["verifier_identities"]["game24"]["sha256"] = "0" * 64
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    packet_sha256 = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    fixture["registration_packet_sha256"] = packet_sha256
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    manifest["registration_packet_sha256"] = packet_sha256
    manifest["artifacts"]["observability_registration_packet"]["sha256"] = packet_sha256
    manifest["artifacts"]["synthetic_contract_fixture"]["sha256"] = hashlib.sha256(
        fixture_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        _validator().Phase13ObservabilityValidationError,
        match="OBSERVABILITY_ARTIFACT_HASH_MISMATCH",
    ):
        _validator().validate_phase13_observability_package(
            package,
            ROOT,
            _manifest_sha256(package),
        )


def test_observability_manifest_requires_every_named_governance_binding(tmp_path: Path) -> None:
    package = tmp_path / "observability"
    shutil.copytree(PACKAGE, package)
    manifest_path = package / "manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementations"]["wrong_core_binding"] = manifest["implementations"].pop(
        "core_main_registry"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        _validator().Phase13ObservabilityValidationError,
        match="OBSERVABILITY_IMPLEMENTATION_BINDING_MISMATCH",
    ):
        _validator().validate_phase13_observability_package(
            package,
            ROOT,
            _manifest_sha256(package),
        )


def test_observability_fixture_rejects_empty_seed_identity() -> None:
    fixture = (PACKAGE / "fixture_v1.json").read_text(encoding="utf-8").replace(
        '"fixture-game24-seed-9"', '""', 1
    )

    with pytest.raises(ValueError, match="string_too_short"):
        Phase13ObservabilityFixture.model_validate_json(fixture)


def test_fixture_is_bound_to_the_registration_packet_and_unique_trial_sequence() -> None:
    registration = importlib.import_module(
        "memcontam.evaluation.phase13_observability_registration"
    )
    fixture = Phase13ObservabilityFixture.model_validate_json(
        (PACKAGE / "fixture_v1.json").read_bytes()
    )
    packet = registration.load_registration_packet(PACKAGE / "registration_packet_v1.json")

    assert fixture.registration_packet_sha256 == hashlib.sha256(
        (PACKAGE / "registration_packet_v1.json").read_bytes()
    ).hexdigest()
    second = fixture.trials[1]
    assert second.context is not None
    duplicate = fixture.model_copy(
        update={
            "trials": (
                fixture.trials[0],
                second.model_copy(
                    update={
                        "trial_id": fixture.trials[0].trial_id,
                        "context": second.context.model_copy(
                            update={"trial_id": fixture.trials[0].trial_id}
                        ),
                        "retrievals": tuple(
                            event.model_copy(update={"trial_id": fixture.trials[0].trial_id})
                            for event in second.retrievals
                        ),
                    }
                ),
                *fixture.trials[2:],
            )
        }
    )
    with pytest.raises(ValueError, match="ORDINARY_SEQUENCE_CONTINUITY_MISMATCH"):
        _validator().reconstruct_fixture(duplicate, packet)


def test_phase13_cli_exposes_observability_validation(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_parser(subparsers)
    args = parser.parse_args(
        [
            "phase13",
            "validate-observability",
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
    assert payload["track2_5_status"] == "TRACK2_5_NOVELTY_OBSERVABILITY_COMPLETE"
    assert payload["mr_p4_prerequisite_status"] == "OBSERVABILITY_PREREQUISITE_MET"
    assert payload["mr_p5_handoff_status"] == "MEASUREMENT_IDENTITY_HANDOFF_CLOSED"
    assert payload["main_a_measured_scientific_execution_count"] == 0
