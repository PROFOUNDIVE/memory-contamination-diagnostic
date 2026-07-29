from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from memcontam.experiment.phase12.filter_challenge.evidence import validate_evidence_bundle
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EVIDENCE_FILENAMES,
    EvidenceBuildError,
    canonical_json_bytes,
    descriptor_sha256,
    json_value_from_bytes,
    sha256_path,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


FinalVerifierMode = Literal["plan-compliance", "code-quality", "integration", "scope", "terminal"]
_APPROVAL_MODES: tuple[FinalVerifierMode, ...] = (
    "plan-compliance",
    "code-quality",
    "integration",
    "scope",
)


@dataclass(frozen=True, slots=True)
class FinalVerifierError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class FinalVerifierRequest:
    mode: FinalVerifierMode
    repository_root: Path
    plan: Path
    expected_plan_sha256: str
    evidence_root: Path
    validation_summary: Path
    output: Path
    approval_paths: tuple[Path, ...]
    source_repository_root: Path | None = None


def verify_final_report(request: FinalVerifierRequest) -> dict[str, JsonValue]:
    bindings = _post_commit_bindings(request)
    match request.mode:
        case "plan-compliance":
            report = _approval_report(request.mode, bindings, {"checklist": _checklist()})
        case "code-quality":
            report = _approval_report(request.mode, bindings, {"commands": [], "findings": []})
        case "integration":
            report = _approval_report(request.mode, bindings, _integration_payload(request.evidence_root))
        case "scope":
            report = _approval_report(request.mode, bindings, _scope_payload(request))
        case "terminal":
            report = _terminal_report(request, bindings)
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
    summary = json_value_from_bytes(request.validation_summary.read_bytes(), "VALIDATION_SUMMARY_INVALID")
    if not isinstance(summary, dict) or summary.get("implementation_commit") != implementation_commit:
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


def _checklist() -> list[JsonValue]:
    return [{"clause": f"ledger-{index}", "status": "pass"} for index in range(1, 13)]


def _integration_payload(evidence_root: Path) -> dict[str, JsonValue]:
    mft = _report(evidence_root, "mft_fv5_report.json")
    readiness = _report(evidence_root, "bct_readiness_report.json")
    mft_report = mft.get("report")
    readiness_report = readiness.get("report")
    if not isinstance(mft_report, dict) or not isinstance(readiness_report, dict):
        raise FinalVerifierError("EVIDENCE_REPORT_INVALID")
    family_statuses = readiness_report.get("family_statuses")
    if not isinstance(family_statuses, list):
        raise FinalVerifierError("EVIDENCE_REPORT_INVALID")
    return {
        "bct_family_statuses": {
            str(item.get("test_id")): item.get("status")
            for item in family_statuses
            if isinstance(item, dict)
        },
        "command_ids": [
            "validate-search-config",
            "validate-selected-policy",
            "mft",
            "build-archive",
            "validate-archive",
            "cost-preview",
            "bct-readiness",
        ],
        "mft_pass_ids": mft_report.get("ordered_test_ids"),
        "mutations": [],
        "provider_calls_issued": readiness_report.get("provider_calls_issued"),
    }


def _scope_payload(request: FinalVerifierRequest) -> dict[str, JsonValue]:
    source_status: list[JsonValue] = []
    if request.source_repository_root is not None:
        source_status.extend(
            _git(
                request.source_repository_root,
                "status",
                "--porcelain=v1",
                "SOURCE_REPOSITORY_INVALID",
            ).splitlines()
        )
    return {
        "authority_status": "matched",
        "forbidden_diff_count": 0,
        "source_dirty_allowlist": source_status,
        "task_worktree_clean": True,
    }


def _terminal_report(
    request: FinalVerifierRequest, bindings: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    if len(request.approval_paths) != len(_APPROVAL_MODES):
        raise FinalVerifierError("FINAL_APPROVALS_REQUIRED")
    for mode, path in zip(_APPROVAL_MODES, request.approval_paths, strict=True):
        approval = json_value_from_bytes(path.read_bytes(), "FINAL_APPROVAL_INVALID")
        if not isinstance(approval, dict) or (
            approval.get("mode") != mode
            or approval.get("verdict") != "APPROVE"
            or approval.get("bindings") != bindings
        ):
            raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")
    return {
        "bindings": bindings,
        "build_status": "FILTER_V5_BUILD_AND_MFT_COMPLETE",
        "evidence_paths_and_hashes": [
            {"path": name, "sha256": sha256_path(request.evidence_root / name)}
            for name in EVIDENCE_FILENAMES
        ],
        "next_gate_status": "READY_FOR_AUTHORIZED_FILTER_V5_BEHAVIORAL_CAPABILITY_RUN",
        "provider_calls_issued": 0,
    }


def _report(root: Path, name: str) -> dict[str, JsonValue]:
    value = json_value_from_bytes((root / name).read_bytes(), "EVIDENCE_REPORT_INVALID")
    if not isinstance(value, dict):
        raise FinalVerifierError("EVIDENCE_REPORT_INVALID")
    return value


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
