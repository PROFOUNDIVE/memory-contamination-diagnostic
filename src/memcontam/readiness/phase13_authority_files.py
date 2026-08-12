from __future__ import annotations

import os
import stat
from pathlib import Path


class AuthorityFileError(ValueError):
    pass


def read_regular_nofollow(path: Path) -> bytes:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise AuthorityFileError("AUTHORITY_FILE_NOT_REGULAR")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except (OSError, AuthorityFileError) as error:
        raise AuthorityFileError("AUTHORITY_FILE_NOT_REGULAR") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise AuthorityFileError("AUTHORITY_FILE_NOT_REGULAR")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)
