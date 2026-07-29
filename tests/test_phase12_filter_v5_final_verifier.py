from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from tests.phase12_filter_v5_summary_cases import complete_validation_summary

from memcontam.experiment.phase12.filter_challenge.evidence import (
    EVIDENCE_FILENAMES,
    EvidenceBuildRequest,
    build_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier import (
    FinalVerifierMode,
    FinalVerifierError,
    FinalVerifierRequest,
    verify_final_report,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "phase12" / "filter_v5"
FINAL_MODES: tuple[FinalVerifierMode, ...] = (
    "plan-compliance",
)
APPROVAL_MODES: tuple[FinalVerifierMode, ...] = (
    "plan-compliance",
    "code-quality",
    "integration",
    "scope",
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _committed_evidence_request(tmp_path: Path) -> EvidenceBuildRequest:
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
        complete_validation_summary(plan_sha256, implementation_commit).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    request = EvidenceBuildRequest(
        repository_root=repository,
        plan=plan,
        expected_plan_sha256=plan_sha256,
        implementation_commit=implementation_commit,
        search_config=fixture_root / "FilterChallengeSearchConfig.yaml",
        fixture_root=fixture_root,
        validation_summary=summary,
        output_root=repository / "evidence",
    )
    build_evidence_bundle(request)
    _git(repository, "add", "evidence")
    _git(repository, "commit", "-qm", "evidence only")
    return request


def _request(
    request: EvidenceBuildRequest, mode: FinalVerifierMode, output: Path
) -> FinalVerifierRequest:
    return FinalVerifierRequest(
        mode=mode,
        repository_root=request.repository_root,
        plan=request.plan,
        expected_plan_sha256=request.expected_plan_sha256,
        evidence_root=request.output_root,
        validation_summary=request.validation_summary,
        output=output,
        approval_paths=(),
    )


@pytest.mark.parametrize("mode", FINAL_MODES)
def test_final_verifier_reports_exact_post_commit_binding(
    tmp_path: Path, mode: FinalVerifierMode
) -> None:
    evidence = _committed_evidence_request(tmp_path)

    report = verify_final_report(_request(evidence, mode, tmp_path / f"{mode}.json"))

    assert report["verdict"] == "APPROVE"
    bindings = report["bindings"]
    assert isinstance(bindings, dict)
    assert set(bindings) == {
        "plan_sha256",
        "validation_summary_sha256",
        "implementation_manifest_sha256",
        "implementation_commit",
        "evidence_commit",
    }
    assert bindings["implementation_commit"] == evidence.implementation_commit


def test_terminal_refuses_missing_approvals_and_rewritten_evidence(tmp_path: Path) -> None:
    evidence = _committed_evidence_request(tmp_path)
    f1 = tmp_path / "f1.json"
    approved = verify_final_report(_request(evidence, "plan-compliance", f1))
    approved_bindings = approved["bindings"]
    assert isinstance(approved_bindings, dict)
    terminal = _request(evidence, "terminal", tmp_path / "terminal.json")
    with pytest.raises(FinalVerifierError, match="FINAL_APPROVALS_REQUIRED"):
        verify_final_report(terminal)

    approvals: list[Path] = []
    for mode in APPROVAL_MODES:
        path = tmp_path / f"{mode}.json"
        path.write_text(
            json.dumps(
                {"bindings": approved_bindings, "mode": mode, "verdict": "APPROVE"},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        approvals.append(path)
    terminal = replace(
        _request(evidence, "terminal", tmp_path / "terminal.json"),
        approval_paths=tuple(approvals),
    )
    with pytest.raises(FinalVerifierError, match="FINAL_APPROVAL_MISMATCH"):
        verify_final_report(terminal)

    (evidence.output_root / EVIDENCE_FILENAMES[1]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(FinalVerifierError, match="EVIDENCE_BYTES_REWRITTEN"):
        verify_final_report(_request(evidence, "plan-compliance", tmp_path / "rewritten.json"))
