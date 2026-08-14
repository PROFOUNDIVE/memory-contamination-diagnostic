from __future__ import annotations

import os
from pathlib import Path
import stat


class AuthorityFileError(ValueError):
    def __init__(self, code: str = "AUTHORITY_FILE_NOT_REGULAR") -> None:
        super().__init__(code)
        self.code = code


def read_regular_nofollow(path: Path) -> bytes:
    target = path if path.is_absolute() else Path.cwd() / path
    if any(part in {".", ".."} for part in target.parts):
        raise AuthorityFileError()
    parts = tuple(part for part in target.parts if part != "/")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in parts[:-1]:
            next_directory = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = next_directory
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise AuthorityFileError()
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1_048_576):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except (OSError, AuthorityFileError) as error:
        raise AuthorityFileError() from error
    finally:
        os.close(directory)
