from __future__ import annotations

import hashlib
import fcntl
from collections.abc import Iterator
from pathlib import Path

import pytest


READINESS_SOURCE = Path("data/phase13/authority/pilot_a_readiness_manifest_v1.json")
READINESS_TARGET = Path(".omo/evidence/pilot-a-closeout/pilot_a_readiness_manifest.json")
READINESS_LOCK = Path(".omo/evidence/pilot-a-closeout/.fixture.lock")
READINESS_SHA256 = "ca82e5a37035d2f0538d7a9288dc30e1768e2816cd321d475c348470b1d96ae8"


@pytest.fixture(scope="session", autouse=True)
def provision_legacy_readiness_fixture() -> Iterator[None]:
    raw = READINESS_SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == READINESS_SHA256
    READINESS_TARGET.parent.mkdir(parents=True, exist_ok=True)
    with READINESS_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = READINESS_TARGET.read_bytes() if READINESS_TARGET.exists() else None
        READINESS_TARGET.write_bytes(raw)
        try:
            yield
        finally:
            if previous is None:
                READINESS_TARGET.unlink()
            else:
                READINESS_TARGET.write_bytes(previous)
