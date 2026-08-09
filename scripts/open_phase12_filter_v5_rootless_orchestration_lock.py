from __future__ import annotations

import argparse
import os
import stat


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_LOCK_FLAGS = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_directory_chain(raw: str) -> int:
    if not raw.startswith("/") or raw.startswith("//") or raw != os.path.normpath(raw):
        raise OSError
    components = raw[1:].split("/")
    if len(components) < 3 or components[-2:] != ["runs", "phase12-filter-v5-rootless-qa"]:
        raise OSError
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in components:
            if component in {"", ".", ".."}:
                raise OSError
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError:
        os.close(descriptor)
        raise


def open_lock(qa_root: str) -> int:
    previous = os.umask(0o077)
    try:
        directory = _open_directory_chain(qa_root)
        try:
            created = False
            try:
                descriptor = os.open(
                    "orchestration.lock", _LOCK_FLAGS | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory
                )
                created = True
            except FileExistsError:
                descriptor = os.open("orchestration.lock", _LOCK_FLAGS, dir_fd=directory)
            try:
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    return 64
                if created:
                    os.fsync(descriptor)
                    os.fsync(directory)
            finally:
                os.close(descriptor)
        finally:
            os.close(directory)
    except OSError:
        return 64
    finally:
        os.umask(previous)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qa-root", required=True)
    arguments = parser.parse_args()
    return open_lock(arguments.qa_root)


if __name__ == "__main__":
    raise SystemExit(main())
