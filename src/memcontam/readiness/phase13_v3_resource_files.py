from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Final, Literal

from pydantic import Field

from memcontam.readiness.phase13_authority_files import (
    AuthorityFileError, authority_directory,
)
from memcontam.readiness.phase13_cost_policy_models import Sha256
from memcontam.readiness.phase13_v3_authority_models import FrozenModel


class ClosureError(ValueError):
    def __init__(self, code: Literal["MAIN_PATH_UNSAFE", "MAIN_GOVERNED_SOURCE_DRIFT"]) -> None:
        super().__init__(code)
        self.code = code


class FileBinding(FrozenModel):
    path: str
    size: int = Field(ge=0)
    sha256: Sha256


@dataclass(frozen=True, slots=True)
class ValidatedResource:
    binding: FileBinding
    raw: bytes


FIXED_GOVERNED: Final = (
    "pyproject.toml",
    "scripts/build_phase13_corrected_main_closure.py",
    "scripts/diagnose_phase13_mr_p5_closure.py",
    "scripts/build_phase13_main_registries.py",
)


def normalized_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Accept only unambiguous UTF-8 POSIX names, including ancestor spelling."""
    seen: set[str] = set()
    prefixes: dict[str, str] = {}
    for path in paths:
        try:
            path.encode("utf-8", errors="strict")
        except UnicodeError as error:
            raise ClosureError("MAIN_PATH_UNSAFE") from error
        parts = path.split("/")
        if any(part in {"", ".", ".."} for part in parts) or any(
            character in path for character in ("\\", "\x00", ":")
        ) or path in seen:
            raise ClosureError("MAIN_PATH_UNSAFE")
        seen.add(path)
        for depth in range(1, len(parts) + 1):
            prefix = "/".join(parts[:depth])
            folded = prefix.casefold()
            if folded in prefixes and prefixes[folded] != prefix:
                raise ClosureError("MAIN_PATH_UNSAFE")
            prefixes[folded] = prefix
    return tuple(sorted(paths, key=lambda path: path.encode("utf-8")))


def is_governed(path: str) -> bool:
    return path in FIXED_GOVERNED or (
        path.startswith("src/memcontam/") and path.endswith(".py")
    )


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _read_stable(directory: int, filename: str, stack: ExitStack) -> bytes:
    descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    stack.callback(os.close, descriptor)
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ClosureError("MAIN_PATH_UNSAFE")
    reads: list[bytes] = []
    for _ in range(2):
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        reads.append(b"".join(chunks))
    after = os.fstat(descriptor)
    if reads[0] != reads[1] or len(reads[0]) != before.st_size or _signature(before) != _signature(after):
        raise ClosureError("MAIN_PATH_UNSAFE")
    return reads[0]


def read_files(root: Path, paths: tuple[str, ...], governed: bool = False) -> tuple[ValidatedResource, ...]:
    """Keep ancestor descriptors pinned; reject namespace/content changes before return."""
    names = normalized_paths(paths)
    try:
        with authority_directory(root) as root_fd, ExitStack() as stack:
            directories = {"": root_fd}
            observations: list[tuple[int, str, tuple[int, ...]]] = []

            def directory(relative: str) -> int:
                if relative in directories:
                    return directories[relative]
                parent, _, component = relative.rpartition("/")
                parent_fd = directory(parent)
                descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                stack.callback(os.close, descriptor)
                info = os.fstat(descriptor)
                if not stat.S_ISDIR(info.st_mode):
                    raise ClosureError("MAIN_PATH_UNSAFE")
                observations.append((parent_fd, component, _signature(info)))
                directories[relative] = descriptor
                return descriptor

            def python_paths(relative: str) -> tuple[str, ...]:
                descriptor = directory(relative)
                entries = normalized_paths(tuple(os.listdir(descriptor)))
                discovered: list[str] = []
                for entry in entries:
                    name = f"{relative}/{entry}"
                    info = os.stat(entry, dir_fd=descriptor, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        if name.endswith(".py"):
                            raise ClosureError("MAIN_PATH_UNSAFE")
                        discovered.extend(python_paths(name))
                    elif stat.S_ISREG(info.st_mode):
                        if name.endswith(".py"):
                            discovered.append(name)
                    else:
                        raise ClosureError("MAIN_PATH_UNSAFE")
                return tuple(discovered)

            if governed:
                names = normalized_paths((*names, *python_paths("src/memcontam")))
            resources: list[ValidatedResource] = []
            for name in names:
                parent, _, filename = name.rpartition("/")
                parent_fd = directory(parent)
                before = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
                raw = _read_stable(parent_fd, filename, stack)
                observations.append((parent_fd, filename, _signature(before)))
                resources.append(ValidatedResource(
                    FileBinding(path=name, size=len(raw), sha256=hashlib.sha256(raw).hexdigest()), raw,
                ))
            for parent_fd, name, expected in observations:
                if _signature(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != expected:
                    raise ClosureError("MAIN_PATH_UNSAFE")
        return tuple(resources)
    except (OSError, AuthorityFileError) as error:
        raise ClosureError("MAIN_PATH_UNSAFE") from error
