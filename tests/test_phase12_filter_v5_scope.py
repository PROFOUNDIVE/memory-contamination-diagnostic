from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.test_phase12_filter_v5_final_verifier_modes import _fixture, _request

from memcontam.experiment.phase12.filter_challenge.final_verifier import (
    FinalVerifierError,
    verify_final_report,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_scope import verify_scope


_ROOT = Path(__file__).resolve().parents[1]
_BASE_COMMIT = "8bd8f4e85ebca919319770e319f0574af32fd458"
_IMPLEMENTATION_COMMIT = "12491c3c60c0e70c1544a9a1bc07c211462a4a61"
_APPROVED_SCOPE_TRANSITIONS = (
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
_DEFAULT_SCOPE_PATHS = tuple(transition[0] for transition in _APPROVED_SCOPE_TRANSITIONS)


@pytest.mark.parametrize(
    "forbidden_path",
    (
        "src/memcontam/memory/admission.py",
        "src/memcontam/experiment/phase12/filter_v4.py",
        "tests/test_pilot_a_preflight.py",
        "tests/test_phase12_pilot_a_invariants.py",
        "tests/test_phase12_pilot_a_launch.py",
        "src/memcontam/readiness/pilot_a_preflight.py",
        "scripts/inspect_phase12_pilot_a.py",
        "configs/phase12/pilot_a.yaml",
        "docs/phase12-pilot-a-operator-checklist.md",
        ".sisyphus/evidence/pilot-a-closeout/pilot_a_execution_manifest.json",
        ".sisyphus/evidence/pilot-a-clean-audit/pilot_a_frozen_evidence_manifest.json",
        "runs/runs/pilot-a-game24-example/public_artifact_manifest.json",
        "runs/runs/pilot-a-game24-example/archive_seal.json",
        "Pilot-A 관련 기록.md",
    ),
)
def test_scope_rejects_actual_pilot_a_and_core_path_families(
    tmp_path: Path, forbidden_path: str
) -> None:
    fixture = _fixture(tmp_path, forbidden_path=forbidden_path)

    with pytest.raises(FinalVerifierError, match="SCOPE_FORBIDDEN_DIFF"):
        verify_final_report(_request(fixture, "scope", tmp_path / "f4.json"))


def test_scope_payload_binds_changed_commit_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = verify_final_report(_request(fixture, "scope", tmp_path / "f4.json"))

    assert report["base_commit"] == fixture.base_commit
    assert report["implementation_commit"] == fixture.evidence.implementation_commit
    assert report["changed_paths"] == ["src/filter_v5_marker.py"]


def test_scope_allows_only_the_two_pinned_content_transitions(tmp_path: Path) -> None:
    repository, source, base_commit, implementation_commit = _scope_transition_repository(tmp_path)

    report = verify_scope(repository, source, base_commit, implementation_commit)

    assert report["approved_scope_exceptions"] == [
        {
            "path": path,
            "reason": reason,
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
        }
        for path, reason, before_sha256, after_sha256 in _APPROVED_SCOPE_TRANSITIONS
    ]


@pytest.mark.parametrize("path", tuple(path for path, *_ in _APPROVED_SCOPE_TRANSITIONS))
def test_scope_rejects_pinned_paths_when_their_content_drifts(tmp_path: Path, path: str) -> None:
    repository, source, base_commit, implementation_commit = _scope_transition_repository(
        tmp_path, paths=(path,), mode="drift"
    )

    with pytest.raises(FinalVerifierError, match="SCOPE_FORBIDDEN_DIFF"):
        verify_scope(repository, source, base_commit, implementation_commit)


@pytest.mark.parametrize("mode", ("deleted", "symlink"))
def test_scope_rejects_missing_or_symlink_pinned_paths(tmp_path: Path, mode: str) -> None:
    path = _APPROVED_SCOPE_TRANSITIONS[0][0]
    repository, source, base_commit, implementation_commit = _scope_transition_repository(
        tmp_path, paths=(path,), mode=mode
    )

    with pytest.raises(FinalVerifierError):
        verify_scope(repository, source, base_commit, implementation_commit)


def _scope_transition_repository(
    tmp_path: Path, paths: tuple[str, ...] = _DEFAULT_SCOPE_PATHS, mode: str = "exact"
) -> tuple[Path, Path, str, str]:
    fixture = _fixture(tmp_path)
    repository = fixture.evidence.repository_root
    for path in paths:
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_git_bytes(_BASE_COMMIT, path))
    _git(repository, "add", "--all")
    _git(repository, "commit", "-qm", "scope base")
    base_commit = _git(repository, "rev-parse", "HEAD")
    for path in paths:
        target = repository / path
        match mode:
            case "exact":
                target.write_bytes(_git_bytes(_IMPLEMENTATION_COMMIT, path))
            case "drift":
                target.write_bytes(b"hash drift\n")
            case "deleted":
                target.unlink()
            case "symlink":
                target.unlink()
                target.symlink_to("scope-replacement")
            case unreachable:
                raise AssertionError(unreachable)
    _git(repository, "add", "--all")
    _git(repository, "commit", "-qm", "scope implementation")
    return repository, fixture.source_repository, base_commit, _git(repository, "rev-parse", "HEAD")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=_ROOT, check=True, capture_output=True
    ).stdout
