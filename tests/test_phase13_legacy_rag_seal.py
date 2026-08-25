import hashlib
import json
from pathlib import Path

from memcontam.readiness.phase13_legacy_rag_validate import validate_legacy_rag_package


ROOT = Path(__file__).resolve().parents[1]


def test_published_track2_seal_binds_validated_three_task_package() -> None:
    package = ROOT / "data/phase13/rag/legacy"
    seal_path = ROOT / "data/phase13/rag/legacy_seal_v1.json"
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
    report = validate_legacy_rag_package(package, ROOT, seal["manifest_sha256"])
    assert report.package_status == seal["status"]
