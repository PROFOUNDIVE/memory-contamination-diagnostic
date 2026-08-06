from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Final


GIT: Final = "/usr/bin/git"
ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
SECTION_PATTERN: Final = re.compile(r'\[([A-Za-z][A-Za-z0-9-]*)(?: "([A-Za-z0-9._-]+)")?\]\Z')
KEY_PATTERN: Final = re.compile(r"[ \t]*([A-Za-z][A-Za-z0-9-]*)[ \t]*=[ \t]*(.*?)[ \t]*\Z")
REF_PATTERN: Final = re.compile(r"refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\Z")
ALLOWED_RULES: Final = (
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".ruff_cache/",
    "*.egg-info/",
    ".omo/",
    ".env",
    ".sisyphus/",
    ".opencode/",
    "AGENTS.md",
    "opencode.json",
    "runs/*",
    "!runs/.gitkeep",
    "/docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json",
)
STRUCTURAL_KEYS: Final = {
    "core.repositoryformatversion",
    "core.filemode",
    "core.bare",
    "core.logallrefupdates",
    "core.worktree",
    "extensions.worktreeconfig",
}
DIRECTORY_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
GIT_ENV: Final = {
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
}


class ContextError(Exception):
    pass


def _absolute(path: Path) -> Path:
    text = os.fspath(path)
    if not text.startswith("/") or text != os.path.normpath(text):
        raise ContextError
    return Path(text)


def _safe_directory(info: os.stat_result, *, current_uid_only: bool = False) -> None:
    owners = {os.getuid()} if current_uid_only else {0, os.getuid()}
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in owners or stat.S_IMODE(info.st_mode) & 0o022:
        raise ContextError


def _open_absolute_directory(path: Path, *, current_uid_only: bool = False) -> int:
    target = _absolute(path)
    descriptor = os.open("/", DIRECTORY_FLAGS)
    try:
        root_info = os.fstat(descriptor)
        if root_info.st_uid != 0:
            raise ContextError
        _safe_directory(root_info)
        for component in target.parts[1:]:
            next_descriptor = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            _safe_directory(os.fstat(descriptor))
        if current_uid_only and os.fstat(descriptor).st_uid != os.getuid():
            raise ContextError
        return descriptor
    except (ContextError, OSError):
        os.close(descriptor)
        raise


def _file_bytes(descriptor: int, *, current_uid_only: bool = True) -> bytes:
    info = os.fstat(descriptor)
    owners = {os.getuid()} if current_uid_only else {0, os.getuid()}
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid not in owners
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise ContextError
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1_048_576):
        chunks.append(chunk)
    return b"".join(chunks)


def _optional_file(directory: int, name: str, *, current_uid_only: bool = True) -> bytes | None:
    try:
        descriptor = os.open(name, FILE_FLAGS, dir_fd=directory)
    except FileNotFoundError:
        return None
    try:
        return _file_bytes(descriptor, current_uid_only=current_uid_only)
    finally:
        os.close(descriptor)


def _required_file(directory: int, name: str, *, current_uid_only: bool = True) -> bytes:
    raw = _optional_file(directory, name, current_uid_only=current_uid_only)
    if raw is None:
        raise ContextError
    return raw


def _require_regular_file(directory: int, name: str) -> None:
    descriptor = os.open(name, FILE_FLAGS, dir_fd=directory)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise ContextError
    finally:
        os.close(descriptor)


def _optional_directory(directory: int, name: str, *, current_uid_only: bool = False) -> int | None:
    try:
        descriptor = os.open(name, DIRECTORY_FLAGS, dir_fd=directory)
    except FileNotFoundError:
        return None
    try:
        _safe_directory(os.fstat(descriptor), current_uid_only=current_uid_only)
        return descriptor
    except (ContextError, OSError):
        os.close(descriptor)
        raise


def _optional_nested_file(directory: int, parts: tuple[str, ...]) -> bytes | None:
    current = os.dup(directory)
    try:
        for component in parts[:-1]:
            next_directory = _optional_directory(current, component)
            if next_directory is None:
                return None
            os.close(current)
            current = next_directory
        return _optional_file(current, parts[-1])
    finally:
        os.close(current)


def _nested_directory_has_entries(directory: int, parts: tuple[str, ...]) -> bool:
    current = os.dup(directory)
    try:
        for component in parts:
            next_directory = _optional_directory(current, component)
            if next_directory is None:
                return False
            os.close(current)
            current = next_directory
        return bool(os.listdir(current))
    finally:
        os.close(current)


def _joined_path(base: Path, raw: bytes) -> Path:
    if not raw or b"\x00" in raw or b"\r" in raw or b"\n" in raw:
        raise ContextError
    text = raw.decode("utf-8")
    candidate = text if text.startswith("/") else os.path.join(os.fspath(base), text)
    return _absolute(Path(os.path.normpath(candidate)))


def _git(root: Path, *arguments: str) -> bytes:
    command = [
        GIT,
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.fileMode=true",
        "-c",
        "core.ignoreCase=false",
        "-c",
        "core.precomposeUnicode=false",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.excludesFile=/dev/null",
        "-c",
        "core.attributesFile=/dev/null",
        "-c",
        "core.bare=false",
        "-c",
        f"core.worktree={root}",
        "-c",
        "status.relativePaths=false",
        "-c",
        "submodule.recurse=false",
        "-c",
        "diff.ignoreSubmodules=none",
        "-C",
        os.fspath(root),
        *arguments,
    ]
    result = subprocess.run(command, check=False, capture_output=True, env=GIT_ENV)
    if result.returncode != 0 or result.stderr:
        raise ContextError
    return result.stdout


def _single_path(raw: bytes) -> Path:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ContextError
    return _absolute(Path(raw[:-1].decode("utf-8")))


def _validate_value(
    section: str,
    subsection: str | None,
    name: str,
    value: str,
    root: Path,
) -> None:
    if not value or unicodedata.normalize("NFC", value) != value or any(ord(character) < 0x20 for character in value):
        raise ContextError
    match section, subsection, name:
        case "core", None, "repositoryformatversion":
            if value != "0":
                raise ContextError
        case "core", None, "filemode":
            if value != "true":
                raise ContextError
        case "core", None, "bare":
            if value != "false":
                raise ContextError
        case "core", None, "logallrefupdates":
            if value != "true":
                raise ContextError
        case "core", None, "worktree":
            if value != os.fspath(root):
                raise ContextError
        case "extensions", None, "worktreeconfig":
            if value != "true":
                raise ContextError
        case "remote", identifier, "url":
            if identifier is None or ID_PATTERN.fullmatch(identifier) is None or any(
                character.isspace() for character in value
            ):
                raise ContextError
        case "remote", identifier, "fetch":
            expected = f"+refs/heads/*:refs/remotes/{identifier}/*"
            if identifier is None or ID_PATTERN.fullmatch(identifier) is None or value != expected:
                raise ContextError
        case "branch", identifier, "remote":
            if identifier is None or ID_PATTERN.fullmatch(identifier) is None or (
                value != "." and ID_PATTERN.fullmatch(value) is None
            ):
                raise ContextError
        case "branch", identifier, "merge":
            if (
                identifier is None
                or ID_PATTERN.fullmatch(identifier) is None
                or REF_PATTERN.fullmatch(value) is None
                or ".." in value
                or "//" in value
            ):
                raise ContextError
        case _:
            raise ContextError


def _parse_config(raw: bytes, root: Path, values: dict[str, str]) -> None:
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or b"\r" in raw or not raw.endswith(b"\n"):
        raise ContextError
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContextError from error
    section = ""
    subsection: str | None = None
    for line in text.splitlines():
        if not line or line.lstrip().startswith(("#", ";")):
            continue
        section_match = SECTION_PATTERN.fullmatch(line)
        if section_match is not None:
            section = section_match.group(1).lower()
            subsection = section_match.group(2)
            if section in {"include", "includeif"}:
                raise ContextError
            if subsection is not None and ID_PATTERN.fullmatch(subsection) is None:
                raise ContextError
            continue
        key_match = KEY_PATTERN.fullmatch(line)
        if key_match is None or not section:
            raise ContextError
        name = key_match.group(1).lower()
        value = key_match.group(2)
        key = f"{section}.{name}" if subsection is None else f"{section}.{subsection}.{name}"
        if key in values:
            raise ContextError
        if key not in STRUCTURAL_KEYS and not (
            key.startswith("remote.") and name in {"url", "fetch"}
        ) and not (key.startswith("branch.") and name in {"remote", "merge"}):
            raise ContextError
        _validate_value(section, subsection, name, value, root)
        values[key] = value
    required = {
        "core.repositoryformatversion": "0",
        "core.bare": "false",
        "core.logallrefupdates": "true",
    }
    if any(values.get(key) != value for key, value in required.items()):
        raise ContextError


def _validate_comments(raw: bytes | None) -> None:
    if raw is None:
        return
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or b"\r" in raw or (raw and not raw.endswith(b"\n")):
        raise ContextError
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContextError from error
    if any(line and not line.startswith(b"#") for line in raw.splitlines()):
        raise ContextError


def _validate_info(directory: int) -> None:
    info = _optional_directory(directory, "info")
    if info is None:
        return
    try:
        _validate_comments(_optional_file(info, "exclude"))
        _validate_comments(_optional_file(info, "attributes"))
    finally:
        os.close(info)


def _validate_objects_and_refs(directory: int) -> None:
    alternates = _optional_nested_file(directory, ("objects", "info", "alternates"))
    if alternates:
        raise ContextError
    if _nested_directory_has_entries(directory, ("refs", "replace")):
        raise ContextError
    packed_refs = _optional_file(directory, "packed-refs")
    if packed_refs is not None and any(
        line and not line.startswith((b"#", b"^")) and b" refs/replace/" in line
        for line in packed_refs.splitlines()
    ):
        raise ContextError


def _validate_gitignore(root: Path, root_descriptor: int) -> None:
    raw = _required_file(root_descriptor, ".gitignore")
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or b"\r" in raw or not raw.endswith(b"\n"):
        raise ContextError
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContextError from error
    rules = tuple(line for line in text.splitlines() if line and not line.startswith("#"))
    if rules != ALLOWED_RULES:
        raise ContextError
    if _git(root, "ls-files", "--error-unmatch", "--", ".gitignore") != b".gitignore\n":
        raise ContextError
    if _git(root, "cat-file", "blob", "HEAD:.gitignore") != raw:
        raise ContextError


def _validate_index(root: Path) -> None:
    verbose = _git(root, "ls-files", "-v", "-z", "--").split(b"\x00")
    typed = _git(root, "ls-files", "-t", "-z", "--").split(b"\x00")
    if any(len(record) >= 2 and 0x61 <= record[0] <= 0x7A and record[1] == 0x20 for record in verbose):
        raise ContextError
    if any(record.startswith(b"S ") for record in typed):
        raise ContextError


def validate(root: Path) -> None:
    repository_root = _absolute(root)
    root_descriptor = _open_absolute_directory(repository_root, current_uid_only=True)
    git_descriptor = -1
    common_descriptor = -1
    try:
        try:
            git_descriptor = os.open(".git", DIRECTORY_FLAGS, dir_fd=root_descriptor)
        except NotADirectoryError:
            pointer = _required_file(root_descriptor, ".git")
            if not pointer.startswith(b"gitdir: ") or not pointer.endswith(b"\n") or pointer.count(b"\n") != 1:
                raise ContextError
            git_path = _joined_path(repository_root, pointer[8:-1])
            git_descriptor = _open_absolute_directory(git_path, current_uid_only=True)
        else:
            _safe_directory(os.fstat(git_descriptor), current_uid_only=True)
            git_path = repository_root / ".git"
        commondir = _optional_file(git_descriptor, "commondir")
        if commondir is None:
            common_path = git_path
            common_descriptor = os.dup(git_descriptor)
        else:
            if not commondir.endswith(b"\n") or commondir.count(b"\n") != 1:
                raise ContextError
            common_path = _joined_path(git_path, commondir[:-1])
            common_descriptor = _open_absolute_directory(common_path, current_uid_only=True)

        values: dict[str, str] = {}
        _parse_config(_required_file(common_descriptor, "config"), repository_root, values)
        _require_regular_file(git_descriptor, "index")
        worktree_config = _optional_file(git_descriptor, "config.worktree")
        if worktree_config is not None:
            if values.get("extensions.worktreeconfig") != "true":
                raise ContextError
            _parse_config(worktree_config, repository_root, values)

        directory_keys: set[tuple[int, int]] = set()
        for descriptor in (git_descriptor, common_descriptor):
            info = os.fstat(descriptor)
            key = (info.st_dev, info.st_ino)
            if key in directory_keys:
                continue
            directory_keys.add(key)
            _validate_info(descriptor)
            _validate_objects_and_refs(descriptor)

        if _single_path(_git(repository_root, "rev-parse", "--show-toplevel")) != repository_root:
            raise ContextError
        if _single_path(_git(repository_root, "rev-parse", "--path-format=absolute", "--git-dir")) != git_path:
            raise ContextError
        if _single_path(_git(repository_root, "rev-parse", "--path-format=absolute", "--git-common-dir")) != common_path:
            raise ContextError
        _validate_index(repository_root)
        _validate_gitignore(repository_root, root_descriptor)
    finally:
        if common_descriptor >= 0:
            os.close(common_descriptor)
        if git_descriptor >= 0:
            os.close(git_descriptor)
        os.close(root_descriptor)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--repo-root":
        return 64
    try:
        validate(Path(sys.argv[2]))
    except (ContextError, OSError, UnicodeError):
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
