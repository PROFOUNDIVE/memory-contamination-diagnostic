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


def test_published_observability_package_is_deterministic_and_truthfully_blocked() -> None:
    report = _validator().validate_phase13_observability_package(
        PACKAGE,
        ROOT,
        _manifest_sha256(),
    )

    assert report.track2_5_status == "BLOCKED"
    assert report.evidence_scope == "synthetic_contract_fixture_only"
    assert report.mr_p4_prerequisite_status == "BLOCKED"
    assert report.mr_p5_handoff_status == "NOT_AVAILABLE"
    assert report.u_t_status == "NOT_REGISTERED_FOR_CURRENT_MAIN"
    assert report.main_a_measured_scientific_execution_count == 0
    assert report.reconstructed_trial_count == 5
    assert report.reconstruction_sha256 == report.repeat_reconstruction_sha256
    assert set(report.blockers) == {
        "FAILURE_CLASSIFIER_REGISTRY_NOT_REGISTERED",
        "RECURRENCE_LOOKBACK_NOT_REGISTERED",
        "EXPOSURE_CONDITIONING_RULE_NOT_REGISTERED",
        "POST_EVICTION_TIMING_NOT_REGISTERED",
        "RETENTION_DURATION_ENDPOINT_NOT_REGISTERED",
        "PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED",
        "CONCRETE_MAIN_SEED_REGISTRY_NOT_FROZEN",
        "LEVEL2_FH_INTERACTIONS_NOT_MATERIALIZED",
    }

    with pytest.raises(
        _validator().Phase13ObservabilityValidationError,
        match="OBSERVABILITY_PREREQUISITE_BLOCKED",
    ):
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
    assert payload["track2_5_status"] == "BLOCKED"
    assert payload["mr_p4_prerequisite_status"] == "BLOCKED"
