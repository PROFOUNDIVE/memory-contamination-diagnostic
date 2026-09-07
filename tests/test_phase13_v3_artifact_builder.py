from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts")


@pytest.fixture(scope="session")
def builder_source(tmp_path_factory):
    from memcontam.readiness.phase13_v3_builder_inputs import STATIC_PATHS
    from memcontam.readiness.phase13_v3_resource_files import read_files
    from memcontam.readiness.phase13_v3_authority_models import ROUTED_DOCUMENTS, PROVENANCE_FILENAME
    root = tmp_path_factory.mktemp("builder-source")
    subprocess.run(("git", "clone", "--local", "--quiet", str(ROOT), str(root)), check=True,
                   env={**os.environ, "GIT_MASTER": "1"})
    shutil.copytree(ROOT / "src", root / "src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("build_phase13_corrected_main_closure.py", "diagnose_phase13_mr_p5_closure.py", "build_phase13_main_registries.py"):
        shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
    for name in ("artifact_builder", "mr_p4", "mr_p5", "mr_p6"):
        shutil.copyfile(ROOT / f"tests/test_phase13_v3_{name}.py", root / f"tests/test_phase13_v3_{name}.py")
    for resource in read_files(ROOT, STATIC_PATHS):
        target = root / resource.binding.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource.raw)
    authority = tmp_path_factory.mktemp("builder-authority")
    for resource in read_files(AUTHORITY, (*tuple(name for _, name in ROUTED_DOCUMENTS), PROVENANCE_FILENAME)):
        (authority / resource.binding.path).write_bytes(resource.raw)
    subprocess.run(("git", "-C", str(root), "add", "src", "scripts"), check=True, env={**os.environ, "GIT_MASTER": "1"})
    subprocess.run(("git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "--quiet", "-m", "Synthetic governed fixture"), check=True, env={**os.environ, "GIT_MASTER": "1"})
    commit = subprocess.check_output(("git", "-C", str(root), "rev-parse", "HEAD"), env={**os.environ, "GIT_MASTER": "1"}).decode().strip()
    return root, commit, authority


@pytest.fixture
def staged(tmp_path, builder_source):
    module = importlib.import_module("memcontam.readiness.phase13_v3_builder")
    root, commit, authority = builder_source
    module.build_mr_p4(root, authority, tmp_path, governed_source_commit=commit)
    return module, root, tmp_path, authority


def publication():
    module = importlib.import_module("memcontam.readiness.phase13_v3_publication")
    return module.publish_artifacts


def test_builder_publishes_complete_bytes_without_replacing_rename(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("replacing rename forbidden")
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(os, "replace", forbidden)
    publication()(tmp_path, (("authority_v3/current_authority_v3.json", b"{}\n"),))
    assert (tmp_path / "authority_v3/current_authority_v3.json").read_bytes() == b"{}\n"


def test_builder_rejects_symlinked_output_parent(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "authority_v3").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
        publication()(tmp_path, (("authority_v3/current_authority_v3.json", b"{}\n"),))
    assert tuple(other.iterdir()) == ()


@pytest.mark.parametrize("target", (
    "mr_p4/corrected_v1/manifest_v2.json", "mr_p5/execution_package_v1.json",
    "mr_p5/execution_package_v2.json", "mr_p6/authorized_execution_v2.json",
))
def test_builder_never_overwrites_historical_target(tmp_path, target):
    path = tmp_path / target
    path.parent.mkdir(parents=True)
    path.write_bytes(b"historical bytes")
    with pytest.raises(ValueError, match="MAIN_HISTORICAL_OUTPUT_FORBIDDEN"):
        publication()(tmp_path, ((target, b"new"),))
    assert path.read_bytes() == b"historical bytes"


@pytest.mark.parametrize("target", ("../escape", "/absolute", "a/../b", "a\\b", "a//b"))
def test_builder_rejects_unsafe_target(tmp_path, target):
    with pytest.raises(ValueError, match="MAIN_PATH_UNSAFE"):
        publication()(tmp_path, ((target, b"new"),))


def test_builder_collision_preserves_every_preexisting_byte(tmp_path):
    path = tmp_path / "authority_v3/current_authority_v3.json"
    path.parent.mkdir()
    path.write_bytes(b"existing")
    with pytest.raises(ValueError, match="MAIN_ARTIFACT_EXISTS"):
        publication()(tmp_path, (("authority_v3/current_authority_v3.json", b"new"),))
    assert path.read_bytes() == b"existing"


@pytest.mark.parametrize("failure_at", (1, 2, 3, 4, 5, 6))
def test_publication_fsync_failure_leaves_no_partial_final(tmp_path, monkeypatch, failure_at):
    original = os.fsync
    calls = 0
    def fail(descriptor):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("injected fsync failure")
        return original(descriptor)
    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(ValueError, match="MAIN_ARTIFACT_PUBLICATION_FAILED"):
        publication()(tmp_path, (("authority_v3/current_authority_v3.json", b"{}\n"),
                                ("cost_envelope_v3/base_inputs_v3.json", b"{}\n")))
    assert not tuple(tmp_path.rglob("*.json"))


def test_partial_write_failure_leaves_no_final(tmp_path, monkeypatch):
    original = os.write
    calls = 0
    def fail(descriptor, raw):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(descriptor, raw[:1])
        raise OSError("injected partial write")
    monkeypatch.setattr(os, "write", fail)
    with pytest.raises(ValueError, match="MAIN_ARTIFACT_PUBLICATION_FAILED"):
        publication()(tmp_path, (("authority_v3/current_authority_v3.json", b"{}\n"),))
    assert not tuple(tmp_path.rglob("*.json"))


def test_stages_and_validators_are_available_without_running_builder():
    module = importlib.import_module("memcontam.readiness.phase13_v3_builder")
    for name in ("build_mr_p4", "build_mr_p5", "build_mr_p6", "validate_stage", "audit"):
        assert callable(getattr(module, name, None))


def test_cli_exposes_exact_stages_without_implicit_execution(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "build_phase13_corrected_main_closure.py"
    shutil.copyfile(ROOT / "scripts" / script.name, script)
    result = subprocess.run(("bash", "-c", 'source .omo/evidence/phase13_shell_contract.sh; phase13_python "$@"',
        "--", str(script), "--help"), cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "{mr-p4,mr-p5,mr-p6,validate,audit}" in result.stdout
