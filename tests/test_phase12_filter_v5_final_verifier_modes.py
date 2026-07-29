from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

from memcontam.experiment.phase12.filter_challenge.evidence import (
    EvidenceBuildRequest,
    build_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    canonical_json_bytes,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier import (
    FinalVerifierError,
    FinalVerifierRequest,
    verify_final_report,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "phase12" / "filter_v5"
Mutation = Literal["forbidden_diff", "invalid_python", "mft_failure", "source_dirty"] | None


@dataclass(frozen=True, slots=True)
class VerifierFixture:
    base_commit: str
    evidence: EvidenceBuildRequest
    source_repository: Path


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fixture(tmp_path: Path, mutation: Mutation = None) -> VerifierFixture:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    fixture_root = repository / "fixtures"
    shutil.copytree(FIXTURES, fixture_root)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "filter-v5@example.test")
    _git(repository, "config", "user.name", "Filter V5")
    _git(repository, "add", "fixtures")
    _git(repository, "commit", "-qm", "base")
    base_commit = _git(repository, "rev-parse", "HEAD")
    implementation_path = repository / "src" / "filter_v5_marker.py"
    implementation_path.parent.mkdir()
    implementation_path.write_text("MARKER: int = 1\n", encoding="utf-8")
    if mutation == "invalid_python":
        implementation_path.write_text("def broken(:\n", encoding="utf-8")
    if mutation == "forbidden_diff":
        forbidden = repository / "docs" / "forbidden.md"
        forbidden.parent.mkdir()
        forbidden.write_text("forbidden\n", encoding="utf-8")
    _git(repository, "add", "--all")
    _git(repository, "commit", "-qm", "implementation")
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
    evidence = EvidenceBuildRequest(
        repository_root=repository,
        plan=plan,
        expected_plan_sha256=plan_sha256,
        implementation_commit=implementation_commit,
        search_config=fixture_root / "FilterChallengeSearchConfig.yaml",
        fixture_root=fixture_root,
        validation_summary=summary,
        output_root=repository / "evidence",
    )
    build_evidence_bundle(evidence)
    if mutation == "mft_failure":
        _rewrite_mft(evidence.output_root)
    _git(repository, "add", "evidence")
    _git(repository, "commit", "-qm", "evidence")
    source = _source_repository(tmp_path, mutation == "source_dirty")
    return VerifierFixture(base_commit, evidence, source)


def _source_repository(tmp_path: Path, dirty: bool) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "filter-v5@example.test")
    _git(source, "config", "user.name", "Filter V5")
    (source / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-qm", "source")
    name = "unexpected.txt" if dirty else "Pilot-A 관련 기록.md"
    (source / name).write_text("untracked\n", encoding="utf-8")
    return source


def _rewrite_mft(evidence_root: Path) -> None:
    report_path = evidence_root / "mft_fv5_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report"]["all_passed"] = False
    report_path.write_bytes(canonical_json_bytes(report))
    manifest_path = evidence_root / "implementation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reports"][report_path.name] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def _request(
    fixture: VerifierFixture,
    mode: Literal["plan-compliance", "code-quality", "integration", "scope"],
    output: Path,
) -> FinalVerifierRequest:
    request = FinalVerifierRequest(
        mode=mode,
        repository_root=fixture.evidence.repository_root,
        plan=fixture.evidence.plan,
        expected_plan_sha256=fixture.evidence.expected_plan_sha256,
        evidence_root=fixture.evidence.output_root,
        validation_summary=fixture.evidence.validation_summary,
        output=output,
        approval_paths=(),
    )
    match mode:
        case "code-quality":
            return replace(request, base_commit=fixture.base_commit)
        case "integration":
            return replace(
                request,
                search_config=fixture.evidence.search_config,
                fixture_root=fixture.evidence.fixture_root,
                execution_prerequisites=(
                    fixture.evidence.fixture_root / "bct_execution_prerequisites.json"
                ),
                scratch_root=output.parent / "scratch",
            )
        case "scope":
            return replace(
                request,
                base_commit=fixture.base_commit,
                source_repository_root=fixture.source_repository,
            )
        case "plan-compliance":
            return request


def test_plan_compliance_rejects_failed_evidence_clause(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, "mft_failure")

    with pytest.raises(FinalVerifierError, match="PLAN_COMPLIANCE_REJECTED"):
        verify_final_report(_request(fixture, "plan-compliance", tmp_path / "f1.json"))


def test_code_quality_runs_nonempty_commands_and_rejects_invalid_python(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "passing")
    report = verify_final_report(_request(fixture, "code-quality", tmp_path / "passing" / "f2.json"))
    commands = report["commands"]
    assert isinstance(commands, list) and commands
    assert all(isinstance(command, dict) and command["exit_code"] == 0 for command in commands)

    invalid = _fixture(tmp_path / "invalid", "invalid_python")
    with pytest.raises(FinalVerifierError, match="CODE_QUALITY_REJECTED"):
        verify_final_report(_request(invalid, "code-quality", tmp_path / "invalid" / "f2.json"))


def test_integration_reruns_commands_and_rejects_evidence_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "passing")
    report = verify_final_report(_request(fixture, "integration", tmp_path / "passing" / "f3.json"))
    assert report["command_ids"] == [
        "validate-search-config",
        "validate-selected-policy",
        "mft",
        "build-archive",
        "validate-archive",
        "cost-preview",
        "bct-readiness",
    ]
    mutations = report["mutations"]
    assert isinstance(mutations, list) and mutations
    assert all(isinstance(item, dict) and item["observed"] == item["expected"] for item in mutations)

    mismatch = _fixture(tmp_path / "mismatch", "mft_failure")
    with pytest.raises(FinalVerifierError, match="INTEGRATION_EVIDENCE_MISMATCH"):
        verify_final_report(_request(mismatch, "integration", tmp_path / "mismatch" / "f3.json"))


def test_scope_reads_real_diff_authorities_and_source_status(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "passing")
    report = verify_final_report(_request(fixture, "scope", tmp_path / "passing" / "f4.json"))
    assert report["forbidden_diff_count"] == 0
    assert report["authority_status"] == "matched"
    assert report["source_dirty_allowlist"] == ["?? Pilot-A 관련 기록.md"]

    forbidden = _fixture(tmp_path / "forbidden", "forbidden_diff")
    with pytest.raises(FinalVerifierError, match="SCOPE_FORBIDDEN_DIFF"):
        verify_final_report(_request(forbidden, "scope", tmp_path / "forbidden" / "f4.json"))
    dirty_source = _fixture(tmp_path / "source", "source_dirty")
    with pytest.raises(FinalVerifierError, match="SOURCE_DIRTY_ALLOWLIST_MISMATCH"):
        verify_final_report(_request(dirty_source, "scope", tmp_path / "source" / "f4.json"))


def test_descriptor_rejects_traversal_component(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
        EvidenceBuildError,
        descriptor_sha256,
    )

    with pytest.raises(EvidenceBuildError, match="DESCRIPTOR_PATH_COMPONENT_INVALID"):
        descriptor_sha256(Path("/tmp/../must-not-open"))


def test_modes_reject_irrelevant_or_missing_mode_inputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan_request = _request(fixture, "plan-compliance", tmp_path / "f1.json")
    with pytest.raises(FinalVerifierError, match="IRRELEVANT_MODE_ARGUMENTS"):
        verify_final_report(replace(plan_request, base_commit=fixture.base_commit))
