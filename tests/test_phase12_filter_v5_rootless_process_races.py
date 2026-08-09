from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCK_HELPER = ROOT / "scripts" / "open_phase12_filter_v5_rootless_orchestration_lock.py"


def _run(qa_root: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        (sys.executable, "-B", "-I", "-S", os.fspath(LOCK_HELPER), "--qa-root", os.fspath(qa_root)),
        check=False,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        close_fds=True,
    )


def test_lock_helper_creates_and_reopens_private_lock(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    qa_root = repository / "runs" / "phase12-filter-v5-rootless-qa"
    qa_root.mkdir(parents=True)
    for path in (repository, repository / "runs", qa_root):
        path.chmod(0o700)

    first = _run(qa_root)
    second = _run(qa_root)

    lock = qa_root / "orchestration.lock"
    assert first.returncode == second.returncode == 0
    assert first.stdout == first.stderr == second.stdout == second.stderr == b""
    assert lock.stat().st_mode & 0o777 == 0o600
    assert lock.stat().st_nlink == 1


def test_lock_helper_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    qa_root = repository / "runs" / "phase12-filter-v5-rootless-qa"
    qa_root.mkdir(parents=True)
    for path in (repository, repository / "runs", qa_root):
        path.chmod(0o700)
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    (qa_root / "orchestration.lock").symlink_to(target)

    completed = _run(qa_root)

    assert completed.returncode == 64
    assert completed.stdout == completed.stderr == b""
    assert target.read_bytes() == b"unchanged"


def test_lock_helper_rejects_wrong_mode_and_multiple_links(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    qa_root = repository / "runs" / "phase12-filter-v5-rootless-qa"
    qa_root.mkdir(parents=True)
    for path in (repository, repository / "runs", qa_root):
        path.chmod(0o700)
    lock = qa_root / "orchestration.lock"
    lock.write_bytes(b"")
    lock.chmod(0o644)

    wrong_mode = _run(qa_root)
    lock.chmod(0o600)
    os.link(lock, qa_root / "second-link")
    multiple_links = _run(qa_root)

    assert wrong_mode.returncode == multiple_links.returncode == 64
    assert wrong_mode.stdout == wrong_mode.stderr == multiple_links.stdout == multiple_links.stderr == b""
