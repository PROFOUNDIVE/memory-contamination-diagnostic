from __future__ import annotations

import hashlib
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from memcontam.readiness import phase13_v3_runtime_identity as runtime


def test_actual_approved_runtime_has_complete_locked_inventory() -> None:
    identity = runtime.freeze_runtime_identity()

    assert identity.executable == "/home/hyunwoo/phase13_artifacts/PeerJ-fast-track/11주차/v11_5_4_6_phase_c_recovery_001/scientific-runtime-locks-001/bin/python"
    assert identity.prefix != identity.base_prefix
    assert identity.base_prefix == "/home/hyunwoo/miniconda3/envs/memcontam"
    assert len(identity.versions) == 104
    assert dict(identity.versions)["cuda-toolkit"] == "13.0.3.0"
    assert dict(identity.versions)["fsspec"] == "2025.3.0"
    assert hashlib.sha256(Path(identity.executable).read_bytes()).hexdigest() == "c823d8b1e491d4273ea6d682f4078907a786984524263444004b3b685f6765db"
    runtime.validate_runtime_identity(identity)


@pytest.mark.parametrize("missing", ["cuda-toolkit", "fsspec"])
def test_extras_bearing_distribution_is_required(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    current = runtime.freeze_runtime_identity()
    distributions = tuple(metadata.distributions(path=[runtime.PREFIX + "/lib/python3.11/site-packages"]))
    remaining = tuple(dist for dist in distributions if dist.metadata["Name"].lower() != missing)
    monkeypatch.setattr(runtime.metadata, "distributions", lambda **_kwargs: iter(remaining))

    with pytest.raises(runtime.RuntimeIdentityError):
        runtime.validate_runtime_identity(current)


@pytest.mark.parametrize("duplicate", [False, True])
def test_unapproved_or_duplicate_distribution_is_rejected(monkeypatch: pytest.MonkeyPatch, duplicate: bool) -> None:
    current = runtime.freeze_runtime_identity()
    distributions = tuple(metadata.distributions(path=[runtime.PREFIX + "/lib/python3.11/site-packages"]))
    additional = distributions[0] if duplicate else SimpleNamespace(metadata={"Name": "unapproved-runtime-extra"}, version="0.0.0")
    monkeypatch.setattr(runtime.metadata, "distributions", lambda **_kwargs: iter((*distributions, additional)))

    with pytest.raises(runtime.RuntimeIdentityError):
        runtime.validate_runtime_identity(current)


def test_historical_interpreter_is_not_a_successor_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    current = runtime.freeze_runtime_identity()
    monkeypatch.setattr(runtime.sys, "executable", "/home/hyunwoo/miniconda3/envs/memcontam/bin/python")

    with pytest.raises(runtime.RuntimeIdentityError):
        runtime.validate_runtime_identity(current)


def test_foreign_base_site_packages_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    current = runtime.freeze_runtime_identity()
    monkeypatch.setattr(runtime.sys, "path", [*sys.path, "/home/hyunwoo/miniconda3/envs/memcontam/lib/python3.11/site-packages"])

    with pytest.raises(runtime.RuntimeIdentityError):
        runtime.validate_runtime_identity(current)


def test_changed_resolved_executable_hash_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    current = runtime.freeze_runtime_identity()
    monkeypatch.setattr(runtime, "EXECUTABLE_SHA256", "0" * 64, raising=False)

    with pytest.raises(runtime.RuntimeIdentityError):
        runtime.validate_runtime_identity(current)
