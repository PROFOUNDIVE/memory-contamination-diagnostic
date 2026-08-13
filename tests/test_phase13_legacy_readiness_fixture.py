from __future__ import annotations

import hashlib
from pathlib import Path

from .phase13_fixture_support import provision_readiness


SOURCE = Path("data/phase13/authority/pilot_a_readiness_manifest_v1.json")
TARGET = Path(".omo/evidence/pilot-a-closeout/pilot_a_readiness_manifest.json")
LOCK_LIKE = TARGET.parent / ".fixture.lock"
SHA256 = "ca82e5a37035d2f0538d7a9288dc30e1768e2816cd321d475c348470b1d96ae8"


def test_legacy_readiness_fixture_is_authentic_and_provisioned() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA256
    assert TARGET.read_bytes() == SOURCE.read_bytes()


def test_provisioning_does_not_create_or_change_lock_like_file() -> None:
    assert not LOCK_LIKE.exists()


def test_provisioning_restores_target_and_preserves_unrelated_file(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(SOURCE.read_bytes())
    target = tmp_path / "evidence" / TARGET.name
    target.parent.mkdir()
    target.write_bytes(b"user target")
    lock_like = target.parent / ".fixture.lock"
    lock_like.write_bytes(b"user lock")

    with provision_readiness(source, target):
        assert target.read_bytes() == SOURCE.read_bytes()

    assert target.read_bytes() == b"user target"
    assert lock_like.read_bytes() == b"user lock"


def test_owned_provisioning_removes_target_and_created_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(SOURCE.read_bytes())
    target = tmp_path / "created" / TARGET.name

    with provision_readiness(source, target):
        assert target.read_bytes() == SOURCE.read_bytes()

    assert not target.exists()
    assert not target.parent.exists()
