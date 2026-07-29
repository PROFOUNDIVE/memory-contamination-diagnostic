from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    AUTHORITY_BINDINGS,
    EvidenceBuildError,
    descriptor_sha256,
    sha256_bytes,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


_FORBIDDEN_PREFIXES = (".sisyphus/", "data/", "docs/", "configs/", "runs/")
_PILOT_A_PREFIXES = (
    "src/memcontam/readiness/pilot_a_",
    "tests/test_phase12_pilot_a_",
    "configs/phase12/pilot_a",
    "runs/runs/pilot-a-game24-",
    ".sisyphus/evidence/pilot-a-",
)
_FORBIDDEN_EXACT_PATHS = frozenset(
    {
        "src/memcontam/memory/admission.py",
        "src/memcontam/memory/filtered_state.py",
        "src/memcontam/experiment/phase12/filter_mft.py",
        "src/memcontam/experiment/phase12/filter_v4.py",
        "tests/test_phase12_filter_v4.py",
        "tests/test_pilot_a_preflight.py",
        "scripts/inspect_phase12_pilot_a.py",
        "data/phase12/filter_v4/evidence.json",
        "docs/scientific-golden.json",
        "Pilot-A 관련 기록.md",
    }
)
_APPROVED_SCOPE_EXCEPTIONS: Final = (
    (
        "docs/phase12-operator-runbook.md",
        "required docs parity",
        "36358a4eedf99f45a97b97ca926dc95189e931d7b37caf77e0916bcad654b7fc",
        "b207d119d34ba1f25ffa5203871ab54b926d3028a1cf9ce05bcf28fcb1f1523c",
    ),
    (
        "tests/test_phase12_pilot_a_launch.py",
        "Pilot-A test-harness hermeticity",
        "0fc8cf7ab5296e09134db88fe68e11878ca2b5da0ac1a21a5ffd0e4c26c0e272",
        "eb84d622f71174e3c30ca31b9ac02ce9783aef78af792ec447e97c0cfc9e8aa9",
    ),
)


@dataclass(frozen=True, slots=True)
class _ApprovedScopeException:
    path: str
    reason: str
    before_sha256: str
    after_sha256: str


def verify_scope(
    repository_root: Path, source_repository_root: Path, base_commit: str, implementation_commit: str
) -> dict[str, JsonValue]:
    changed = _git(repository_root, "diff", "--name-only", base_commit, implementation_commit)
    approved = [
        exception
        for path in changed.splitlines()
        if _is_forbidden(path)
        and (
            exception := _approved_scope_exception(
                repository_root, base_commit, implementation_commit, path
            )
        )
        is not None
    ]
    forbidden = [
        path
        for path in changed.splitlines()
        if _is_forbidden(path)
        and _approved_scope_exception(repository_root, base_commit, implementation_commit, path) is None
    ]
    if forbidden:
        raise FinalVerifierError("SCOPE_FORBIDDEN_DIFF")
    if _git(repository_root, "status", "--porcelain=v1"):
        raise FinalVerifierError("TASK_WORKTREE_DIRTY")
    source_status = _status(source_repository_root)
    if source_status != ["?? Pilot-A 관련 기록.md"]:
        raise FinalVerifierError("SOURCE_DIRTY_ALLOWLIST_MISMATCH")
    source_dirty_allowlist: list[JsonValue] = [*source_status]
    changed_path_values: list[JsonValue] = [*changed.splitlines()]
    approved_scope_exceptions: list[JsonValue] = [
        {
            "path": exception.path,
            "reason": exception.reason,
            "before_sha256": exception.before_sha256,
            "after_sha256": exception.after_sha256,
        }
        for exception in sorted(approved, key=lambda item: item.path)
    ]
    try:
        matched = all(descriptor_sha256(path).sha256 == digest for _, path, digest in AUTHORITY_BINDINGS)
    except EvidenceBuildError as error:
        raise FinalVerifierError(error.code) from error
    if not matched:
        raise FinalVerifierError("AUTHORITY_MISMATCH")
    return {
        "authority_status": "matched",
        "approved_scope_exceptions": approved_scope_exceptions,
        "base_commit": base_commit,
        "changed_paths": changed_path_values,
        "forbidden_diff_count": 0,
        "implementation_commit": implementation_commit,
        "source_dirty_allowlist": source_dirty_allowlist,
        "task_worktree_clean": True,
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-c", "core.quotepath=false", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FinalVerifierError("SCOPE_GIT_INVALID")
    return result.stdout.strip()


def _is_forbidden(path: str) -> bool:
    return path.startswith((*_FORBIDDEN_PREFIXES, *_PILOT_A_PREFIXES)) or path in _FORBIDDEN_EXACT_PATHS


def _approved_scope_exception(
    repository_root: Path, base_commit: str, implementation_commit: str, path: str
) -> _ApprovedScopeException | None:
    for exception in _APPROVED_SCOPE_EXCEPTIONS:
        candidate = _ApprovedScopeException(*exception)
        if candidate.path != path:
            continue
        before = _git_bytes(repository_root, "show", f"{base_commit}:{path}")
        after = _git_bytes(repository_root, "show", f"{implementation_commit}:{path}")
        if before is None or after is None:
            return None
        if sha256_bytes(before) == candidate.before_sha256 and sha256_bytes(after) == candidate.after_sha256:
            return candidate
    return None


def _git_bytes(root: Path, *arguments: str) -> bytes | None:
    result = subprocess.run(
        ("git", "-c", "core.quotepath=false", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _status(root: Path) -> list[str]:
    result = subprocess.run(
        ("git", "-C", str(root), "status", "--porcelain=v1", "-z"),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise FinalVerifierError("SCOPE_GIT_INVALID")
    return [entry.decode("utf-8") for entry in result.stdout.split(b"\0") if entry]
