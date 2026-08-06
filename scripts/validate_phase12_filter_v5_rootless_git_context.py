from __future__ import annotations

import argparse
import os
import stat
import subprocess
from pathlib import Path


ALLOWED_PREFIXES = ("remote.", "branch.")
ALLOWED_KEYS = {
    "core.repositoryformatversion",
    "core.filemode",
    "core.bare",
    "core.logallrefupdates",
    "core.worktree",
    "extensions.worktreeconfig",
}


def _safe_directory(path: Path) -> None:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid not in {0, os.getuid()} or stat.S_IMODE(info.st_mode) & 0o022:
        raise RuntimeError


def _read(path: Path) -> bytes:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_nlink != 1 or info.st_uid not in {0, os.getuid()} or stat.S_IMODE(info.st_mode) & 0o022:
        raise RuntimeError
    return path.read_bytes()


def _config(path: Path) -> None:
    if not path.exists():
        return
    section = ""
    for raw_line in _read(path).splitlines():
        if not raw_line or raw_line.startswith(b"#") or raw_line.startswith(b";"):
            continue
        if raw_line.startswith(b"[") and raw_line.endswith(b"]"):
            section = raw_line[1:-1].decode("ascii").lower().replace(' "', ".").replace('"', "")
            continue
        if b"=" not in raw_line or not section:
            raise RuntimeError
        key = f"{section}.{raw_line.split(b'=', 1)[0].strip().decode('ascii').lower()}"
        if key.startswith("include.") or key.startswith("includeif.") or (key not in ALLOWED_KEYS and not key.startswith(ALLOWED_PREFIXES)):
            raise RuntimeError


def _comments_only(path: Path) -> None:
    if not path.exists():
        return
    raw = _read(path)
    if raw and (not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw):
        raise RuntimeError
    if any(line and not line.startswith(b"#") for line in raw.splitlines()):
        raise RuntimeError


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), "--no-optional-locks", "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "-c", "core.hooksPath=/dev/null", "-c", "core.excludesFile=/dev/null", "-c", "core.attributesFile=/dev/null", *arguments],
        check=False,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_ATTR_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_REPLACE_OBJECTS": "1"},
    )
    if result.returncode:
        raise RuntimeError
    return result.stdout


def validate(root: Path) -> None:
    if not root.is_absolute() or root != root.resolve(strict=True):
        raise RuntimeError
    _safe_directory(root)
    git_path = root / ".git"
    if git_path.is_dir():
        git_directory = git_path
    else:
        pointer = _read(git_path)
        if not pointer.startswith(b"gitdir: ") or not pointer.endswith(b"\n"):
            raise RuntimeError
        git_directory = Path(pointer[8:-1].decode("utf-8")).resolve(strict=True)
    _safe_directory(git_directory)
    common_marker = git_directory / "commondir"
    common_directory = (git_directory / _read(common_marker).decode("utf-8").strip()).resolve(strict=True) if common_marker.exists() else git_directory
    _safe_directory(common_directory)
    _config(common_directory / "config")
    _config(git_directory / "config.worktree")
    for directory in {git_directory / "info", common_directory / "info"}:
        if not directory.exists():
            continue
        _safe_directory(directory)
        _comments_only(directory / "exclude")
        _comments_only(directory / "attributes")
    alternates = common_directory / "objects" / "info" / "alternates"
    if alternates.exists() and _read(alternates):
        raise RuntimeError
    if (common_directory / "refs" / "replace").exists() and any((common_directory / "refs" / "replace").iterdir()):
        raise RuntimeError
    if any(line.startswith(b"h ") for line in _git(root, "ls-files", "-v").splitlines()) or any(
        line.startswith(b"S ") for line in _git(root, "ls-files", "-t").splitlines()
    ):
        raise RuntimeError


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        validate(arguments.repo_root)
    except (OSError, RuntimeError, UnicodeDecodeError):
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
