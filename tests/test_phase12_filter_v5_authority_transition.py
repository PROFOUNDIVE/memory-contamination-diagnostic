from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from memcontam.readiness.phase13_v3_builder import validate_mr_p4
from .test_phase12_externalized_provenance import legacy_registry

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_phase12_filter_v5_authority_snapshot.py"
MANIFEST = ROOT / "docs" / "evidence" / "phase12-filter-v5-bct-v1" / "authority_transition_manifest.json"
AUTHORITY_ROOT = Path(
    "/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts"
)
pytest_plugins = ("tests.test_phase13_v3_artifact_builder",)


def _run(manifest: Path, output: Path, authority_root: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--output", str(output)]
    if authority_root is not None:
        command.extend(("--authority-root", str(authority_root)))
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)


def _copy_authorities(tmp_path: Path) -> tuple[Path, Path]:
    authority_root = tmp_path / "authorities"
    manifest = tmp_path / "authority_transition_manifest.json"
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for authority in payload["external_authorities"]:
        source = AUTHORITY_ROOT / authority["relative_path"]
        path = authority_root / authority["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        info = path.stat()
        authority["byte_count"] = info.st_size
        authority["identity"] = {
            "device": info.st_dev,
            "inode": info.st_ino,
            "mode": stat.S_IMODE(info.st_mode),
        }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return authority_root, manifest


def test_authority_snapshot_validates_exact_repository_inputs(
    staged: tuple[ModuleType, Path, Path, Path],
) -> None:
    _, root, output, authority = staged
    legacy = legacy_registry()
    before = {path.name: path.read_bytes() for path in authority.iterdir()}

    snapshot = validate_mr_p4(root, authority, output)

    assert snapshot.status == "CLOSED"
    assert {path.name: path.read_bytes() for path in authority.iterdir()} == before
    serialized = snapshot.model_dump_json()
    assert all(row.sha256 not in serialized for row in legacy.records)
    assert all(not row.current_authorization_member for row in legacy.records)
    conformance = json.loads((output / "mr_p4/corrected_v3/provider_free_conformance_v3.json").read_bytes())
    assert conformance["real_provider_calls"] == 0
    assert not (output / "mr_p6").exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "Phase 12 Filter-v5 Verifier-Backed Challenge Amendment.md",
        "Phase 12 — THEORETICAL ARTIFACT.md",
        "Phase 12-Compatible Baseline Memory and Lightweight Filter Design revised-v3.md",
        "Phase 12-Compatible Contamination Construction Intervention Timing and Sensitivity Protocol.md",
        "Phase 12-Compatible Pilot Main and Exploratory Experiment Design.md",
        "AGENTS.md",
    ],
)
def test_authority_snapshot_rejects_mutated_external_copy(tmp_path: Path, relative_path: str) -> None:
    authority_root, manifest = _copy_authorities(tmp_path)
    target = authority_root / relative_path
    target.write_text(target.read_text(encoding="utf-8") + "\nmutation\n", encoding="utf-8")

    result = _run(manifest, tmp_path / "snapshot.json", authority_root)

    assert result.returncode != 0
    assert "THEORETICAL_AUTHORITY_DRIFT" in result.stdout


def test_authority_snapshot_rejects_manifest_hash_substitution(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["external_authorities"][0]["sha256"] = hashlib.sha256(b"substitution").hexdigest()
    manifest = tmp_path / "authority_transition_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(manifest, tmp_path / "snapshot.json")

    assert result.returncode != 0
    assert "THEORETICAL_AUTHORITY_DRIFT" in result.stdout
