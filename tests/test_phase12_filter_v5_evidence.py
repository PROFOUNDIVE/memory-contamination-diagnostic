from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.evidence import (
    EVIDENCE_FILENAMES,
    EvidenceBuildError,
    EvidenceBuildRequest,
    build_evidence_bundle,
    validate_evidence_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "phase12" / "filter_v5"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _prepared_request(tmp_path: Path, output_name: str = "evidence") -> EvidenceBuildRequest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repository = tmp_path / "repository"
    repository.mkdir()
    fixture_root = repository / "fixtures"
    shutil.copytree(FIXTURES, fixture_root)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "filter-v5@example.test")
    _git(repository, "config", "user.name", "Filter V5")
    _git(repository, "add", "fixtures")
    _git(repository, "commit", "-qm", "synthetic inputs")
    implementation_commit = _git(repository, "rev-parse", "HEAD")

    plan = tmp_path / "approved-plan.md"
    plan.write_text("# Approved synthetic plan\n", encoding="utf-8")
    plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
    summary = tmp_path / "validation-summary.json"
    summary.write_text(
        json.dumps(
            {
                "implementation_commit": implementation_commit,
                "provider_calls_issued": 0,
                "reviewed_plan_sha256": plan_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return EvidenceBuildRequest(
        repository_root=repository,
        plan=plan,
        expected_plan_sha256=plan_sha256,
        implementation_commit=implementation_commit,
        search_config=fixture_root / "FilterChallengeSearchConfig.yaml",
        fixture_root=fixture_root,
        validation_summary=summary,
        output_root=repository / output_name,
    )


def _bundle_bytes(root: Path) -> dict[str, bytes]:
    return {name: (root / name).read_bytes() for name in EVIDENCE_FILENAMES}


def test_builder_writes_deterministic_non_self_hashing_nine_file_bundle(tmp_path: Path) -> None:
    first = _prepared_request(tmp_path / "first")
    second = _prepared_request(tmp_path / "second")

    build_evidence_bundle(first)
    build_evidence_bundle(second)

    assert {path.name for path in first.output_root.iterdir()} == set(EVIDENCE_FILENAMES)
    assert _bundle_bytes(first.output_root) == _bundle_bytes(second.output_root)
    manifest = json.loads((first.output_root / "implementation_manifest.json").read_text())
    assert set(manifest["reports"]) == set(EVIDENCE_FILENAMES[1:])
    assert "implementation_manifest_sha256" not in manifest["header"]
    assert "evidence_commit" not in manifest["header"]
    for name in EVIDENCE_FILENAMES:
        report = json.loads((first.output_root / name).read_text())
        assert set(report["header"]) >= {
            "amendment",
            "authority_hashes",
            "config_schema_hashes",
            "implementation_commit",
            "plan_sha256",
            "policy",
            "validation_summary_sha256",
        }
        assert "evidence_commit" not in report["header"]
        assert "implementation_manifest_sha256" not in report["header"]
    assert all(
        manifest["reports"][name] == hashlib.sha256((first.output_root / name).read_bytes()).hexdigest()
        for name in EVIDENCE_FILENAMES[1:]
    )


def test_builder_rejects_plan_summary_and_report_drift(tmp_path: Path) -> None:
    request = _prepared_request(tmp_path)
    build_evidence_bundle(request)

    request.plan.write_text("# Mutated\n", encoding="utf-8")
    with pytest.raises(EvidenceBuildError, match="PLAN_SHA256_MISMATCH"):
        build_evidence_bundle(request)
    summary_request = _prepared_request(tmp_path / "summary")
    summary_request.validation_summary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvidenceBuildError, match="VALIDATION_SUMMARY_PLAN_MISMATCH"):
        build_evidence_bundle(summary_request)
    (request.output_root / EVIDENCE_FILENAMES[1]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvidenceBuildError, match="EVIDENCE_GRAPH_MISMATCH"):
        validate_evidence_bundle(request.output_root)


def test_builder_script_accepts_exact_required_arguments(tmp_path: Path) -> None:
    request = _prepared_request(tmp_path, "script-evidence")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_phase12_filter_v5_evidence.py"),
            "--repository-root",
            str(request.repository_root),
            "--plan",
            str(request.plan),
            "--expected-plan-sha256",
            request.expected_plan_sha256,
            "--implementation-commit",
            request.implementation_commit,
            "--search-config",
            str(request.search_config),
            "--fixture-root",
            str(request.fixture_root),
            "--validation-summary",
            str(request.validation_summary),
            "--output-root",
            str(request.output_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)) == {"files", "implementation_manifest_sha256"}
