from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE = Path("data/phase13/authority/pilot_a_readiness_manifest_v1.json")
TARGET = Path(".omo/evidence/pilot-a-closeout/pilot_a_readiness_manifest.json")
SHA256 = "ca82e5a37035d2f0538d7a9288dc30e1768e2816cd321d475c348470b1d96ae8"


def test_legacy_readiness_fixture_is_authentic_and_provisioned() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA256
    assert TARGET.read_bytes() == SOURCE.read_bytes()
