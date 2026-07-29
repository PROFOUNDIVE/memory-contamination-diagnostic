from __future__ import annotations

import subprocess
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    AUTHORITY_BINDINGS,
    EvidenceBuildError,
    descriptor_sha256,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


_FORBIDDEN_PREFIXES = (".sisyphus/", "data/", "docs/", "configs/", "runs/")


def verify_scope(
    repository_root: Path, source_repository_root: Path, base_commit: str, implementation_commit: str
) -> dict[str, JsonValue]:
    changed = _git(repository_root, "diff", "--name-only", base_commit, implementation_commit)
    forbidden = [path for path in changed.splitlines() if path.startswith(_FORBIDDEN_PREFIXES)]
    if forbidden:
        raise FinalVerifierError("SCOPE_FORBIDDEN_DIFF")
    if _git(repository_root, "status", "--porcelain=v1"):
        raise FinalVerifierError("TASK_WORKTREE_DIRTY")
    source_status = _status(source_repository_root)
    if source_status != ["?? Pilot-A 관련 기록.md"]:
        raise FinalVerifierError("SOURCE_DIRTY_ALLOWLIST_MISMATCH")
    source_dirty_allowlist: list[JsonValue] = [*source_status]
    try:
        matched = all(descriptor_sha256(path).sha256 == digest for _, path, digest in AUTHORITY_BINDINGS)
    except EvidenceBuildError as error:
        raise FinalVerifierError(error.code) from error
    if not matched:
        raise FinalVerifierError("AUTHORITY_MISMATCH")
    return {"authority_status": "matched", "forbidden_diff_count": 0, "source_dirty_allowlist": source_dirty_allowlist, "task_worktree_clean": True}


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(("git", "-C", str(root), *arguments), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise FinalVerifierError("SCOPE_GIT_INVALID")
    return result.stdout.strip()


def _status(root: Path) -> list[str]:
    result = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain=v1", "-z"),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise FinalVerifierError("SCOPE_GIT_INVALID")
    return [entry.decode("utf-8") for entry in result.stdout.split(b"\0") if entry]
