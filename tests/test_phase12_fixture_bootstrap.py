from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12"
APPENDIX_B_MANIFEST_SHA256 = "b129d5a66166b0ec5fc8bb4b32c2ab2a76e09238d074b91687322a97828adda8"


@dataclass(frozen=True)
class FixtureManifest:
    files: dict[str, str]
    fixture_manifest_version: str
    seed: int
    source_sha256: str


@dataclass(frozen=True)
class FixtureCheckReport:
    root: Path
    manifest: FixtureManifest
    codes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.codes


def load_bootstrap_manifest(path: Path) -> FixtureManifest:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return FixtureManifest(
        files=dict(data["files"]),
        fixture_manifest_version=data["fixture_manifest_version"],
        seed=data["seed"],
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def verify_bootstrap_fixtures(root: Path, manifest: FixtureManifest) -> FixtureCheckReport:
    codes: list[str] = []
    if manifest.source_sha256 != APPENDIX_B_MANIFEST_SHA256:
        codes.append("MANIFEST_HASH_MISMATCH")
    fixture_ids: dict[str, str] = {}
    fixture_payloads: dict[str, dict[str, object]] = {}

    for filename, expected_hash in manifest.files.items():
        path = root / filename
        if not path.exists():
            codes.append("FIXTURE_MISSING")
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        fixture_payloads[filename] = payload
        fixture_id = payload.get("fixture_id")
        if isinstance(fixture_id, str):
            fixture_ids[fixture_id] = filename

        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            codes.append("FIXTURE_HASH_MISMATCH")

    for filename, payload in fixture_payloads.items():
        compose = payload.get("compose")
        if not isinstance(compose, list):
            continue

        for referenced_fixture_id in compose:
            if not isinstance(referenced_fixture_id, str):
                continue
            referenced_filename = fixture_ids.get(referenced_fixture_id)
            if referenced_filename is None or not (root / referenced_filename).exists():
                codes.append("FIXTURE_REFERENCE_MISSING")
                break

    return FixtureCheckReport(root=root, manifest=manifest, codes=tuple(dict.fromkeys(codes)))


def test_bootstrap_manifest_loads_and_committed_fixtures_verify() -> None:
    manifest = load_bootstrap_manifest(FIXTURE_ROOT / "manifest.json")

    report = verify_bootstrap_fixtures(FIXTURE_ROOT, manifest)

    assert report.ok


def test_bootstrap_missing_fixture_reports_missing_code(tmp_path: Path) -> None:
    fixture_root = tmp_path / "phase12"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    (fixture_root / "FX-TOOL-001.json").unlink()

    manifest = load_bootstrap_manifest(fixture_root / "manifest.json")
    report = verify_bootstrap_fixtures(fixture_root, manifest)

    assert "FIXTURE_MISSING" in report.codes


def test_bootstrap_hash_mismatch_reports_hash_code(tmp_path: Path) -> None:
    fixture_root = tmp_path / "phase12"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    path = fixture_root / "FX-SCHEMA-001.json"
    path.write_bytes(path.read_bytes().replace(b"phase12", b"phase13", 1))

    manifest = load_bootstrap_manifest(fixture_root / "manifest.json")
    report = verify_bootstrap_fixtures(fixture_root, manifest)

    assert "FIXTURE_HASH_MISMATCH" in report.codes


def test_bootstrap_missing_reference_reports_reference_code(tmp_path: Path) -> None:
    fixture_root = tmp_path / "phase12"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    path = fixture_root / "FX-E2E-001.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["compose"] = ["FX-CONFIG-001", "FX-DOES-NOT-EXIST-001"]
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    manifest = load_bootstrap_manifest(fixture_root / "manifest.json")
    report = verify_bootstrap_fixtures(fixture_root, manifest)

    assert "FIXTURE_REFERENCE_MISSING" in report.codes


def test_bootstrap_manifest_byte_mutation_reports_manifest_hash_code(tmp_path: Path) -> None:
    fixture_root = tmp_path / "phase12"
    shutil.copytree(FIXTURE_ROOT, fixture_root)
    manifest_path = fixture_root / "manifest.json"
    manifest_path.write_bytes(
        manifest_path.read_bytes().replace(b"phase12-fixtures-v13", b"phase12-fixtures-v14", 1)
    )

    manifest = load_bootstrap_manifest(manifest_path)
    report = verify_bootstrap_fixtures(fixture_root, manifest)

    assert "MANIFEST_HASH_MISMATCH" in report.codes
