from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
from types import ModuleType
from typing import Literal, assert_never

import pytest
from memcontam.readiness import phase13_v3_builder as builders
from memcontam.readiness.phase13_v3_publication import P4_PATHS, P5_PATHS

pytest_plugins = ("tests.test_phase13_v3_artifact_builder",)


def test_mr_p5_requires_validated_mr_p4(tmp_path):
    module = importlib.import_module("memcontam.readiness.phase13_v3_builder")
    with pytest.raises(ValueError, match="MAIN_PREDECESSOR_MISSING"):
        module.build_mr_p5(tmp_path, tmp_path, tmp_path)


def test_mr_p5_has_acyclic_proof_and_separate_inventories(staged):
    module, root, output, authority = staged
    package = module.build_mr_p5(root, authority, output)
    assert package.governed_source is not None
    assert package.generated_closure is not None
    assert all("mr_p5/" not in row.path and "mr_p6/" not in row.path for row in package.generated_closure.rows)
    assert module.validate_stage(root, authority, output, stage="mr-p5") == package
    assert not (output / "mr_p6").exists()


@pytest.mark.parametrize("field", ("package_core_hash", "cost_proof_hash", "complete_inputs_hash", "generated_closure_hash", "base_inputs_hash", "witness_hash", "live_contract_hash"))
def test_self_rehashed_package_bindings_never_validate(staged, field):
    from memcontam.readiness.phase13_v3_cost_models import canonical_bytes, digest
    module, root, output, authority = staged
    package = module.build_mr_p5(root, authority, output).model_copy(update={field: "f" * 64})
    package = package.model_copy(update={"package_hash": digest(package, "package_hash")})
    (output / "mr_p5/execution_package_v3.json").write_bytes(canonical_bytes(package))
    with pytest.raises(ValueError, match="MAIN_(COST_PROOF|ARTIFACT_BINDING)_MISMATCH"):
        module.validate_stage(root, authority, output, stage="mr-p5")


@pytest.mark.parametrize("change_kind", ("artifact", "governed"))
def test_governed_source_drift_invalidates_package_but_artifact_commit_does_not(
    staged: tuple[ModuleType, Path, Path, Path],
    change_kind: Literal["artifact", "governed"],
) -> None:
    _, root, output, authority = staged
    package = builders.build_mr_p5(root, authority, output)

    match change_kind:
        case "artifact":
            artifact = root / "synthetic-artifact-descendant.json"
            artifact.write_text("{}\n", encoding="utf-8")
            subprocess.run(
                ("git", "-C", str(root), "add", "--", artifact.name),
                check=True, env={**os.environ, "GIT_MASTER": "1"},
            )
            subprocess.run(
                ("git", "-C", str(root), "-c", "user.name=Fixture", "-c",
                 "user.email=fixture@example.invalid", "commit", "--quiet",
                 "-m", "Synthetic artifact-only descendant"),
                check=True, env={**os.environ, "GIT_MASTER": "1"},
            )
            assert builders.validate_mr_p5(root, authority, output) == package
        case "governed":
            source = root / "src/memcontam/__init__.py"
            original = source.read_bytes()
            try:
                source.write_bytes(original + b"\n")
                with pytest.raises(ValueError, match="MAIN_GOVERNED_SOURCE_DRIFT"):
                    builders.validate_mr_p5(root, authority, output)
            finally:
                source.write_bytes(original)
        case unreachable:
            assert_never(unreachable)


@pytest.mark.parametrize("predecessor", P4_PATHS)
def test_mr_p5_rejects_each_missing_predecessor_without_publication(
    staged: tuple[ModuleType, Path, Path, Path], predecessor: str,
) -> None:
    _, root, output, authority = staged
    (output / predecessor).unlink()

    with pytest.raises(ValueError, match="MAIN_PREDECESSOR_MISSING"):
        builders.build_mr_p5(root, authority, output)

    assert all(not (output / path).exists() for path in P5_PATHS)


@pytest.mark.parametrize("artifact", P5_PATHS[:-1])
def test_mr_p5_rejects_noncanonical_bound_artifact_bytes(
    staged: tuple[ModuleType, Path, Path, Path], artifact: str,
) -> None:
    _, root, output, authority = staged
    builders.build_mr_p5(root, authority, output)
    target = output / artifact
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(ValueError, match="MAIN_COST_PROOF_MISMATCH"):
        builders.validate_mr_p5(root, authority, output)


def test_mr_p5_collision_preserves_complete_existing_stage(
    staged: tuple[ModuleType, Path, Path, Path],
) -> None:
    _, root, output, authority = staged
    builders.build_mr_p5(root, authority, output)
    original = {path: (output / path).read_bytes() for path in P5_PATHS}

    with pytest.raises(ValueError, match="MAIN_ARTIFACT_EXISTS"):
        builders.build_mr_p5(root, authority, output)

    assert {path: (output / path).read_bytes() for path in P5_PATHS} == original


def test_mr_p5_rebuild_is_byte_deterministic(staged, tmp_path):
    module, root, output, authority = staged
    first = module.build_mr_p5(root, authority, output)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    module.build_mr_p4(root, authority, fresh, governed_source_commit=first.governed_source.governed_source_commit, identity=first.identity)
    second = module.build_mr_p5(root, authority, fresh)
    assert first == second


def test_mr_p5_validator_accepts_untouched_package(staged):
    module, root, output, authority = staged
    package = module.build_mr_p5(root, authority, output)
    assert module.validate_mr_p5(root, authority, output) == package


def test_mr_p5_rejects_missing_package_sidecar(staged):
    module, root, output, authority = staged
    module.build_mr_p5(root, authority, output)
    (output / P5_PATHS[-1]).unlink()
    with pytest.raises(ValueError, match="MAIN_PREDECESSOR_MISSING"):
        module.validate_mr_p5(root, authority, output)


def test_mr_p5_rejects_package_directory_symlink(staged, tmp_path):
    module, root, output, authority = staged
    module.build_mr_p5(root, authority, output)
    target = output / "mr_p5/execution_package_v3.json"
    target.unlink()
    target.symlink_to(tmp_path / "escape.json")
    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
        module.validate_mr_p5(root, authority, output)


def test_mr_p5_rejects_artifact_hash_drift(staged):
    module, root, output, authority = staged
    module.build_mr_p5(root, authority, output)
    target = output / P5_PATHS[0]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="MAIN_COST_PROOF_MISMATCH"):
        module.validate_mr_p5(root, authority, output)


def test_mr_p5_does_not_publish_mr_p6(staged):
    module, root, output, authority = staged
    module.build_mr_p5(root, authority, output)
    assert not (output / "mr_p6").exists()


def test_mr_p5_rejects_preexisting_output(staged):
    module, root, output, authority = staged
    output.mkdir(exist_ok=True)
    (output / "mr_p5").mkdir()
    (output / P5_PATHS[0]).write_bytes(b"old")
    with pytest.raises(ValueError, match="MAIN_ARTIFACT_EXISTS"):
        module.build_mr_p5(root, authority, output)
