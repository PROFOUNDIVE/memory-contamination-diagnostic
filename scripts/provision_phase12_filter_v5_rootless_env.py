from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat
from typing import Final


_ASSIGNMENT: Final = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_./:-]+\n")
_KEY: Final = re.compile(rb"OPENAI_API_KEY=([A-Za-z0-9_-]{20,512})\n")
_READ_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def _read_source(path: Path) -> bytes:
    descriptor = os.open(path, _READ_FLAGS)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 65_536
        ):
            raise OSError
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 65_537):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def provision(source: Path, checkout: Path) -> int:
    previous = os.umask(0o077)
    try:
        try:
            raw = _read_source(source)
            raw.decode("utf-8")
            if not raw or b"\0" in raw or b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
                return 64
            lines = raw.splitlines(keepends=True)
            if b"".join(lines) != raw or any(_ASSIGNMENT.fullmatch(line) is None for line in lines):
                return 64
            keys = [match.group(1) for line in lines if (match := _KEY.fullmatch(line)) is not None]
            if len(keys) != 1:
                return 64
            directory = os.open(checkout, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                info = os.fstat(directory)
                if info.st_uid != os.getuid() or info.st_mode & 0o022:
                    return 64
                descriptor = os.open(
                    ".env",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=directory,
                )
                try:
                    output = b"OPENAI_API_KEY=" + keys[0] + b"\n"
                    offset = 0
                    while offset < len(output):
                        offset += os.write(descriptor, output[offset:])
                    os.fchmod(descriptor, 0o600)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.fsync(directory)
            finally:
                os.close(directory)
        except (FileExistsError, OSError, UnicodeDecodeError):
            return 64
        return 0
    finally:
        os.umask(previous)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    arguments = parser.parse_args()
    return provision(arguments.source, arguments.repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
