from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Final

DIRECTORY_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
ALLOWED_NAMES: Final = {
    ".phase12-filter-v5-rootless-local-bootstrap-tmp",
    "phase12-filter-v5-rootless-local",
}


class CleanupError(Exception):
    pass


def _remove(directory: int, name: str, device: int) -> None:
    try:
        child = os.open(name, DIRECTORY_FLAGS, dir_fd=directory)
    except NotADirectoryError:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if info.st_dev != device or info.st_uid != os.getuid():
            raise CleanupError
        os.unlink(name, dir_fd=directory)
        return
    try:
        info = os.fstat(child)
        if info.st_dev != device or info.st_uid != os.getuid():
            raise CleanupError
        for entry in os.listdir(child):
            _remove(child, entry, device)
    finally:
        os.close(child)
    os.rmdir(name, dir_fd=directory)


def cleanup(root: Path, names: tuple[str, ...]) -> None:
    raw = os.fspath(root)
    if not raw.startswith("/") or raw != os.path.normpath(raw):
        raise CleanupError
    descriptor = os.open(raw, DIRECTORY_FLAGS)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise CleanupError
        for name in names:
            if name not in ALLOWED_NAMES:
                raise CleanupError
            try:
                _remove(descriptor, name, info.st_dev)
            except FileNotFoundError:
                continue
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--venvs-root", type=Path, required=True)
    parser.add_argument("--remove-transient", action="store_true")
    parser.add_argument("--remove-incomplete-venv", action="store_true")
    try:
        arguments = parser.parse_args()
        if not arguments.remove_transient:
            raise CleanupError
        names = [".phase12-filter-v5-rootless-local-bootstrap-tmp"]
        if arguments.remove_incomplete_venv:
            names.append("phase12-filter-v5-rootless-local")
        cleanup(arguments.venvs_root, tuple(names))
    except (CleanupError, OSError, SystemExit):
        return 64
    return 0


if __name__ == "__main__":
    sys.exit(main())
