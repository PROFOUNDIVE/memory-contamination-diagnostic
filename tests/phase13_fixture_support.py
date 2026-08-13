from __future__ import annotations

import fcntl
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


READINESS_SHA256 = "ca82e5a37035d2f0538d7a9288dc30e1768e2816cd321d475c348470b1d96ae8"


@contextmanager
def provision_readiness(source_path: Path, target_path: Path) -> Iterator[None]:
    with source_path.open("rb") as source:
        fcntl.flock(source, fcntl.LOCK_EX)
        raw = source.read()
        assert hashlib.sha256(raw).hexdigest() == READINESS_SHA256
        parent_existed = target_path.parent.exists()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        previous = target_path.read_bytes() if target_path.exists() else None
        target_path.write_bytes(raw)
        try:
            yield
        finally:
            if previous is None:
                target_path.unlink()
                if not parent_existed:
                    target_path.parent.rmdir()
            else:
                target_path.write_bytes(previous)
