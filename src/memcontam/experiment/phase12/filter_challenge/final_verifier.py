from __future__ import annotations

import subprocess
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.evidence import validate_evidence_bundle
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EVIDENCE_FILENAMES,
    EvidenceBuildError,
    canonical_json_bytes,
    descriptor_sha256,
    sha256_path,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_integration import (
    verify_integration,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_plan import (
    verify_plan_compliance,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_quality import (
    verify_code_quality,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_scope import verify_scope
from memcontam.experiment.phase12.filter_challenge.final_verifier_terminal import (
    build_terminal_report,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import (
    FinalVerifierError,
    FinalVerifierMode,
    FinalVerifierRequest,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.validation_summary import Task17ValidationSummary


def verify_final_report(request: FinalVerifierRequest) -> dict[str, JsonValue]:
    _validate_mode_inputs(request)
    bindings = _post_commit_bindings(request)
    summary = Task17ValidationSummary.model_validate_json(request.validation_summary.read_bytes()).model_dump(mode="json")
    match request.mode:
        case "plan-compliance":
            report = _approval_report(request.mode, bindings, verify_plan_compliance(request.evidence_root, summary))
        case "code-quality":
            assert request.base_commit is not None
            report = _approval_report(request.mode, bindings, verify_code_quality(request.repository_root, request.base_commit, str(bindings["implementation_commit"]), request.evidence_root, request.validation_summary))
        case "integration":
            assert request.search_config is not None and request.fixture_root is not None
            assert request.execution_prerequisites is not None and request.scratch_root is not None
            report = _approval_report(request.mode, bindings, verify_integration(request.repository_root, request.evidence_root, str(bindings["implementation_commit"]), request.search_config, request.fixture_root, request.execution_prerequisites, request.scratch_root, request.validation_summary))
        case "scope":
            assert request.base_commit is not None and request.source_repository_root is not None
            report = _approval_report(request.mode, bindings, verify_scope(request.repository_root, request.source_repository_root, request.base_commit, str(bindings["implementation_commit"])))
        case "terminal":
            report = build_terminal_report(request, bindings)
        case unreachable:
            raise AssertionError(unreachable)
    request.output.parent.mkdir(parents=True, exist_ok=True)
    request.output.write_bytes(canonical_json_bytes(report))
    return report


def _post_commit_bindings(request: FinalVerifierRequest) -> dict[str, JsonValue]:
    if descriptor_sha256(request.plan).sha256 != request.expected_plan_sha256:
        raise FinalVerifierError("PLAN_SHA256_MISMATCH")
    head = _verify_committed_bytes(request.repository_root, request.evidence_root)
    try:
        bundle = validate_evidence_bundle(request.evidence_root)
    except EvidenceBuildError as error:
        raise FinalVerifierError(error.code) from error
    summary_hash = sha256_path(request.validation_summary)
    header = bundle.header
    if header.get("plan_sha256") != request.expected_plan_sha256:
        raise FinalVerifierError("EVIDENCE_PLAN_BINDING_MISMATCH")
    if header.get("validation_summary_sha256") != summary_hash:
        raise FinalVerifierError("EVIDENCE_VALIDATION_SUMMARY_MISMATCH")
    implementation_commit = header.get("implementation_commit")
    if not isinstance(implementation_commit, str):
        raise FinalVerifierError("EVIDENCE_IMPLEMENTATION_COMMIT_INVALID")
    summary = Task17ValidationSummary.model_validate_json(request.validation_summary.read_bytes())
    if summary.implementation_commit != implementation_commit:
        raise FinalVerifierError("VALIDATION_SUMMARY_COMMIT_MISMATCH")
    if _git(request.repository_root, "rev-parse", "HEAD^", "EVIDENCE_PARENT_INVALID") != implementation_commit:
        raise FinalVerifierError("EVIDENCE_PARENT_INVALID")
    if _git(request.repository_root, "status", "--porcelain=v1", "REPOSITORY_STATUS_INVALID"):
        raise FinalVerifierError("REPOSITORY_DIRTY")
    return {
        "plan_sha256": request.expected_plan_sha256,
        "validation_summary_sha256": summary_hash,
        "implementation_manifest_sha256": bundle.implementation_manifest_sha256,
        "implementation_commit": implementation_commit,
        "evidence_commit": head,
    }


def _verify_committed_bytes(repository_root: Path, evidence_root: Path) -> str:
    head = _git(repository_root, "rev-parse", "HEAD", "EVIDENCE_COMMIT_INVALID")
    try:
        relative_root = evidence_root.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise FinalVerifierError("EVIDENCE_ROOT_OUTSIDE_REPOSITORY") from error
    expected_paths = {f"{relative_root}/{name}" for name in EVIDENCE_FILENAMES}
    changed = set(
        _git(repository_root, "diff-tree", "--no-commit-id", "--name-only", "-r", head, "EVIDENCE_COMMIT_INVALID").splitlines()
    )
    if changed != expected_paths:
        raise FinalVerifierError("EVIDENCE_COMMIT_SCOPE_INVALID")
    for name in EVIDENCE_FILENAMES:
        committed = _git_bytes(repository_root, "show", f"{head}:{relative_root}/{name}")
        if committed != (evidence_root / name).read_bytes():
            raise FinalVerifierError("EVIDENCE_BYTES_REWRITTEN")
    return head


def _approval_report(
    mode: FinalVerifierMode, bindings: dict[str, JsonValue], payload: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    return {"bindings": bindings, "mode": mode, "verdict": "APPROVE", **payload}


def _validate_mode_inputs(request: FinalVerifierRequest) -> None:
    integration = (
        request.search_config,
        request.fixture_root,
        request.execution_prerequisites,
        request.scratch_root,
    )
    if request.mode != "terminal" and request.approval_paths:
        raise FinalVerifierError("IRRELEVANT_MODE_ARGUMENTS")
    match request.mode:
        case "plan-compliance" | "terminal":
            if request.base_commit is not None or request.source_repository_root is not None or any(integration):
                raise FinalVerifierError("IRRELEVANT_MODE_ARGUMENTS")
        case "code-quality":
            if request.base_commit is None:
                raise FinalVerifierError("MODE_ARGUMENT_REQUIRED")
            if request.source_repository_root is not None or any(integration):
                raise FinalVerifierError("IRRELEVANT_MODE_ARGUMENTS")
        case "integration":
            if any(value is None for value in integration):
                raise FinalVerifierError("MODE_ARGUMENT_REQUIRED")
            if request.base_commit is not None or request.source_repository_root is not None:
                raise FinalVerifierError("IRRELEVANT_MODE_ARGUMENTS")
        case "scope":
            if request.base_commit is None or request.source_repository_root is None:
                raise FinalVerifierError("MODE_ARGUMENT_REQUIRED")
            if any(integration):
                raise FinalVerifierError("IRRELEVANT_MODE_ARGUMENTS")
        case unreachable:
            raise AssertionError(unreachable)


def _git(root: Path, *arguments: str) -> str:
    *command, error_code = arguments
    result = subprocess.run(
        ["git", "-C", str(root), *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise FinalVerifierError(error_code)
    return result.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *arguments], check=False, capture_output=True)
    if result.returncode != 0:
        raise FinalVerifierError("EVIDENCE_COMMIT_INVALID")
    return result.stdout
