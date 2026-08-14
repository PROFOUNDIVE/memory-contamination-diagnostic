from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_provenance import (
    Phase13ProvenanceError,
    validate_provenance_bundle,
)


def _write_bundle(root: Path) -> tuple[Path, Path]:
    artifact = root / "inputs" / "registry.json"
    artifact.parent.mkdir()
    artifact.write_text('{"registry":"prospective"}\n', encoding="utf-8")
    manifest_payload: dict[str, JsonValue] = {
        "schema_version": "phase13_provenance_manifest_v1",
        "bundle_id": "prospective-bundle",
        "artifacts": [
            {
                "role": "execution_registry",
                "path": "inputs/registry.json",
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    canonical = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()
    manifest_payload["manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    seal = root / "seal.json"
    seal.write_text(
        json.dumps(
            {
                "schema_version": "phase13_provenance_seal_v1",
                "bundle_id": "prospective-bundle",
                "manifest_hash": manifest_payload["manifest_hash"],
            }
        ),
        encoding="utf-8",
    )
    return manifest, seal


def test_provenance_bundle_validates_hash_bound_artifacts(tmp_path: Path) -> None:
    manifest, seal = _write_bundle(tmp_path)

    report = validate_provenance_bundle(tmp_path, manifest, seal)

    assert report.bundle_id == "prospective-bundle"
    assert report.artifact_count == 1


def test_provenance_bundle_rejects_artifact_drift(tmp_path: Path) -> None:
    manifest, seal = _write_bundle(tmp_path)
    (tmp_path / "inputs" / "registry.json").write_text("drift\n", encoding="utf-8")

    with pytest.raises(Phase13ProvenanceError, match="ARTIFACT_HASH_MISMATCH"):
        validate_provenance_bundle(tmp_path, manifest, seal)
