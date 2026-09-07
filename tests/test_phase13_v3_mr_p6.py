from __future__ import annotations

import importlib
from pathlib import Path
import sqlite3
from types import ModuleType

import pytest
from memcontam.readiness import phase13_v3_builder as builders
from memcontam.readiness.phase13_v3_publication import P5_PATHS, P6_PATHS
from .test_phase13_v3_artifact_builder import builder_source as builder_source, staged as staged


def test_mr_p6_requires_validated_mr_p5(tmp_path):
    module = importlib.import_module("memcontam.readiness.phase13_v3_builder")
    with pytest.raises(ValueError, match="MAIN_PREDECESSOR_MISSING"):
        module.build_mr_p6(tmp_path, tmp_path, tmp_path)


def test_mr_p6_authorizes_exact_mr_p5_and_stops(staged):
    module, root, output, authority = staged
    module.build_mr_p5(root, authority, output)
    authorization = module.build_mr_p6(root, authority, output)
    assert module.validate_stage(root, authority, output, stage="mr-p6") == authorization
    assert authorization.status == "AUTHORIZED_EXECUTION"
    assert not tuple(output.rglob("*.sqlite3"))


@pytest.mark.parametrize("field", ("execution_package_sha256", "execution_package_hash"))
def test_stale_or_self_rehashed_mr_p5_never_authorizes(staged, field):
    from memcontam.readiness.phase13_v3_cost_models import canonical_bytes, digest
    module, root, output, authority = staged
    module.build_mr_p5(root, authority, output)
    authorization = module.build_mr_p6(root, authority, output).model_copy(update={field: "e" * 64})
    authorization = authorization.model_copy(update={"authorization_hash": digest(authorization, "authorization_hash")})
    (output / "mr_p6/authorized_execution_v3.json").write_bytes(canonical_bytes(authorization))
    (output / "mr_p6/authorized_execution_v3.sha256").write_text(digest(authorization) + "\n")
    with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        module.validate_stage(root, authority, output, stage="mr-p6")


def test_audit_compares_complete_fresh_rebuild(staged, builder_source, tmp_path):
    module, root, output, authority = staged
    module.build_mr_p5(root, authority, output)
    module.build_mr_p6(root, authority, output)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    module.build_mr_p4(root, authority, fresh, governed_source_commit=builder_source[1])
    module.build_mr_p5(root, authority, fresh)
    module.build_mr_p6(root, authority, fresh)
    module.audit(root, authority, output, compare_output_root=fresh)
    target = fresh / "main_live_contract_v3.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="MAIN_ARTIFACT_BINDING_MISMATCH"):
        module.audit(root, authority, output, compare_output_root=fresh)


def test_validate_creates_no_production_ledger(
    staged: tuple[ModuleType, Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, root, output, authority = staged
    builders.build_mr_p5(root, authority, output)
    expected = builders.build_mr_p6(root, authority, output)
    before = {str(path.relative_to(output)): path.read_bytes()
              for path in output.rglob("*") if path.is_file()}

    def deny_database(*args: str, **kwargs: str) -> None:
        pytest.fail("validate must not open a ledger")

    monkeypatch.setattr(sqlite3, "connect", deny_database)
    actual = builders.validate_mr_p6(root, authority, output)

    assert actual == expected
    assert {str(path.relative_to(output)): path.read_bytes()
            for path in output.rglob("*") if path.is_file()} == before
    assert not tuple(output.rglob("*.sqlite3"))


@pytest.mark.parametrize("sidecar_kind", ("no_lf", "uppercase", "leading_space", "two_lfs", "wrong_digest"))
def test_mr_p6_rejects_noncanonical_or_unbound_sidecar(
    staged: tuple[ModuleType, Path, Path, Path], sidecar_kind: str,
) -> None:
    _, root, output, authority = staged
    builders.build_mr_p5(root, authority, output)
    builders.build_mr_p6(root, authority, output)
    sidecar = output / P6_PATHS[1]
    original = sidecar.read_bytes()
    malformed = {
        "no_lf": original.rstrip(b"\n"),
        "uppercase": original.upper(),
        "leading_space": b" " + original,
        "two_lfs": original + b"\n",
        "wrong_digest": b"0" * 64 + b"\n",
    }
    sidecar.write_bytes(malformed[sidecar_kind])

    with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        builders.validate_mr_p6(root, authority, output)


@pytest.mark.parametrize("predecessor", (P5_PATHS[-1], P5_PATHS[1]))
def test_mr_p6_requires_complete_package_and_proof_before_publication(
    staged: tuple[ModuleType, Path, Path, Path], predecessor: str,
) -> None:
    _, root, output, authority = staged
    builders.build_mr_p5(root, authority, output)
    (output / predecessor).unlink()

    with pytest.raises(ValueError, match="MAIN_PREDECESSOR_MISSING"):
        builders.build_mr_p6(root, authority, output)

    assert all(not (output / path).exists() for path in P6_PATHS)


def test_mr_p6_collision_preserves_authorization_and_sidecar(
    staged: tuple[ModuleType, Path, Path, Path],
) -> None:
    _, root, output, authority = staged
    builders.build_mr_p5(root, authority, output)
    builders.build_mr_p6(root, authority, output)
    original = {path: (output / path).read_bytes() for path in P6_PATHS}

    with pytest.raises(ValueError, match="MAIN_ARTIFACT_EXISTS"):
        builders.build_mr_p6(root, authority, output)

    assert {path: (output / path).read_bytes() for path in P6_PATHS} == original
