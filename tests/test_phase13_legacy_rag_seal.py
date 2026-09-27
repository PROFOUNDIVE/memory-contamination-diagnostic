import hashlib
import json
from pathlib import Path

from memcontam.readiness.phase13_legacy_rag_validate import validate_legacy_rag_package


ROOT = Path(__file__).resolve().parents[1]


def _assert_seal(package: Path, seal_path: Path) -> tuple[str, str]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    unsigned = dict(seal)
    seal_hash = unsigned.pop("seal_sha256")

    assert seal["status"] == "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
    assert seal["tasks"] == ["game24", "math_equation_balancer", "word_sorting"]
    assert seal["manifest_sha256"] == hashlib.sha256(
        (package / "manifest.json").read_bytes()
    ).hexdigest()
    assert seal_hash == hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_sha256 = seal["manifest_sha256"]
    status = seal["status"]
    assert isinstance(manifest_sha256, str)
    assert isinstance(status, str)
    return manifest_sha256, status


def test_historical_track2_seal_remains_cryptographically_intact() -> None:
    _assert_seal(
        ROOT / "data/phase13/rag/legacy",
        ROOT / "data/phase13/rag/legacy_seal_v1.json",
    )


def test_current_track2_seal_binds_validated_three_task_package() -> None:
    package = ROOT / "data/phase13/rag/legacy_v2"
    manifest_sha256, status = _assert_seal(
        package, ROOT / "data/phase13/rag/legacy_seal_v2.json"
    )
    report = validate_legacy_rag_package(package, ROOT, manifest_sha256)
    assert report.package_status == status
