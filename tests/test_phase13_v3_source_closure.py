from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
from types import ModuleType
from unittest.mock import Mock

import pytest

from memcontam.clients.openai_responses import OpenAIResponsesClient


MODULE = "memcontam.readiness.phase13_v3_source_closure"
ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "src/memcontam/__init__.py": b"",
    "src/memcontam/readiness/phase13_main_live_cli.py": b"",
    "src/memcontam/registry.py": b"CALLBACKS = {}\n",
    "src/memcontam/callback.py": b"def callback(): return 1\n",
    "pyproject.toml": b"[project]\nname = 'fixture'\n",
    "scripts/build_phase13_corrected_main_closure.py": b"",
    "scripts/diagnose_phase13_mr_p5_closure.py": b"",
    "scripts/build_phase13_main_registries.py": b"",
}


@pytest.fixture(autouse=True)
def provider_denial(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    constructor = Mock(side_effect=AssertionError("provider construction forbidden"))
    request = Mock(side_effect=AssertionError("provider request forbidden"))
    monkeypatch.setattr(OpenAIResponsesClient, "__init__", constructor)
    monkeypatch.setattr(OpenAIResponsesClient, "chat", request)
    yield
    assert constructor.call_count == request.call_count == 0


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *args), check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_MASTER": "1", "GIT_CONFIG_NOSYSTEM": "1"},
    ).stdout.strip()


@dataclass(frozen=True, slots=True)
class Repository:
    root: Path
    commit: str


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    for name, raw in FILES.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
        "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture")
    return Repository(tmp_path, git(tmp_path, "rev-parse", "HEAD"))


@pytest.fixture
def api() -> ModuleType:
    assert importlib.util.find_spec(MODULE) is not None, "Task-5 inventory API missing"
    return importlib.import_module(MODULE)


def test_authoritative_inventory_contract_exists() -> None:
    assert importlib.util.find_spec(MODULE) is not None, "Task-5 inventory API missing"


def test_inventory_matches_independent_commit_set(api: ModuleType, repository: Repository) -> None:
    inventory = api.freeze_governed(repository.root, repository.commit)
    assert {row.path: row.sha256 for row in inventory.rows} == {
        name: hashlib.sha256(raw).hexdigest() for name, raw in FILES.items()
    }
    assert inventory.governed_source_commit == repository.commit
    assert api.validate_governed(repository.root, inventory) == inventory


def test_authoritative_inventory_catches_nonimported_registry_omission(
    api: ModuleType, repository: Repository,
) -> None:
    inventory = api.freeze_governed(repository.root, repository.commit)
    omitted = tuple(row for row in inventory.rows if not row.path.endswith("registry.py"))
    raw = json.dumps([row.model_dump() for row in omitted], ensure_ascii=False,
                     sort_keys=True, separators=(",", ":")).encode() + b"\n"
    forged = inventory.model_copy(update={"rows": omitted,
                                          "governed_tree_sha256": hashlib.sha256(raw).hexdigest()})
    with pytest.raises(api.ClosureError, match="^MAIN_GOVERNED_SOURCE_DRIFT$"):
        api.validate_governed(repository.root, forged)


@pytest.mark.parametrize("mutation", ["changed", "deleted", "extra"])
def test_governed_tree_drift(api: ModuleType, repository: Repository, mutation: str) -> None:
    inventory = api.freeze_governed(repository.root, repository.commit)
    target = repository.root / "src/memcontam/callback.py"
    if mutation == "changed":
        target.write_bytes(b"changed\n")
    elif mutation == "deleted":
        target.unlink()
    else:
        target.with_name("extra.py").write_bytes(b"extra\n")
    with pytest.raises(api.ClosureError, match="^MAIN_GOVERNED_SOURCE_DRIFT$"):
        api.validate_governed(repository.root, inventory)


def test_inventory_rejects_symlinked_parent(api: ModuleType, repository: Repository) -> None:
    source = repository.root / "src"
    source.rename(repository.root / "relocated")
    source.symlink_to("relocated", target_is_directory=True)
    with pytest.raises(api.ClosureError, match="^MAIN_PATH_UNSAFE$"):
        api.freeze_governed(repository.root, repository.commit)


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_inventory_rejects_final_nonregular(api: ModuleType, repository: Repository, kind: str) -> None:
    target = repository.root / "src/memcontam/callback.py"
    target.unlink()
    if kind == "symlink":
        target.symlink_to("registry.py")
    elif kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    with pytest.raises(api.ClosureError, match="^MAIN_PATH_UNSAFE$"):
        api.freeze_governed(repository.root, repository.commit)


@pytest.mark.parametrize("name", ["/abs", "../escape", "a/../b", "", "a//b", "./a",
                                 "a/", "a\\b", "a\x00b", "\udcff", "C:/drive"])
def test_resource_path_rejected_before_access(api: ModuleType, tmp_path: Path, name: str) -> None:
    with pytest.raises(api.ClosureError, match="^MAIN_PATH_UNSAFE$"):
        api.freeze_resources(tmp_path, (name,))


@pytest.mark.parametrize("names", [("a", "a"), ("A", "a"), ("Dir/a", "dir/b")])
def test_resource_names_reject_aliases(api: ModuleType, tmp_path: Path, names: tuple[str, ...]) -> None:
    with pytest.raises(api.ClosureError, match="^MAIN_PATH_UNSAFE$"):
        api.freeze_resources(tmp_path, names)


def test_missing_resource_fails_closed(api: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(api.ClosureError, match="^MAIN_PATH_UNSAFE$"):
        api.freeze_resources(tmp_path, ("missing.json",))


def test_resource_closure_returns_validated_bytes(api: ModuleType, tmp_path: Path) -> None:
    name, raw = "generated/\uac00.json", b'{"value":1}\n'
    (tmp_path / "generated").mkdir()
    (tmp_path / name).write_bytes(raw)
    closure = api.freeze_resources(tmp_path, (name,))
    validated = api.validate_resources(tmp_path, closure, (name,))
    assert validated[0].raw == raw
    assert validated[0].binding.sha256 == hashlib.sha256(raw).hexdigest()
    assert "governed_source_commit" not in closure.model_dump()


@pytest.mark.parametrize("mutation", ["omitted", "extra", "changed"])
def test_resources_require_independent_expected_set(api: ModuleType, tmp_path: Path, mutation: str) -> None:
    for name in ("registry.json", "resource.bin"):
        (tmp_path / name).write_bytes(b"data")
    expected = ("registry.json", "resource.bin")
    closure = api.freeze_resources(tmp_path, expected)
    if mutation == "omitted":
        closure = api.freeze_resources(tmp_path, expected[:1])
    elif mutation == "extra":
        expected = expected[:1]
    else:
        (tmp_path / expected[0]).write_bytes(b"drift")
    with pytest.raises(api.ClosureError, match="^MAIN_GOVERNED_SOURCE_DRIFT$"):
        api.validate_resources(tmp_path, closure, expected)


def test_unstable_descriptor_read_is_rejected(api: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "resource.bin"
    target.write_bytes(b"original")
    original_read = os.read

    def mutate(descriptor: int, count: int) -> bytes:
        raw = original_read(descriptor, count)
        if raw == b"original":
            target.write_bytes(b"mutation")
        return raw

    monkeypatch.setattr(os, "read", mutate)
    with pytest.raises(api.ClosureError, match="^MAIN_PATH_UNSAFE$"):
        api.freeze_resources(tmp_path, (target.name,))


def test_artifact_only_descendant_keeps_source_binding(api: ModuleType, repository: Repository) -> None:
    inventory = api.freeze_governed(repository.root, repository.commit)
    (repository.root / "generated.json").write_bytes(b"{}\n")
    git(repository.root, "add", "generated.json")
    git(repository.root, "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
        "-c", "core.hooksPath=/dev/null", "commit", "-qm", "artifact")
    assert api.validate_governed(repository.root, inventory) == inventory


@pytest.mark.parametrize("commit", ["HEAD", "a" * 40, "../HEAD"])
def test_source_requires_existing_exact_commit(api: ModuleType, repository: Repository, commit: str) -> None:
    with pytest.raises(api.ClosureError, match="^MAIN_GOVERNED_SOURCE_DRIFT$"):
        api.freeze_governed(repository.root, commit)


def test_ast_diagnostic_is_explicitly_nonauthoritative(repository: Repository) -> None:
    diagnostic = runpy.run_path(str(ROOT / "scripts/diagnose_phase13_mr_p5_closure.py"))
    package = repository.root / "package.json"
    package.write_bytes(b'{"artifacts":[]}')
    report = diagnostic["_report"](repository.root, Path("package.json"))
    assert report.authoritative is False
    assert "src/memcontam/registry.py" not in report.omitted_local_imports


@pytest.mark.parametrize("name", ["src/memcontam/code.py",
                                 "data/phase13/main/mr_p5/execution_package_v3.json",
                                 "data/phase13/main/mr_p6/authorized_execution_v3.json"])
def test_resources_exclude_governed_and_successor_artifacts(api: ModuleType, tmp_path: Path, name: str) -> None:
    with pytest.raises(api.ClosureError, match="^MAIN_GOVERNED_SOURCE_DRIFT$"):
        api.freeze_resources(tmp_path, (name,))


@pytest.mark.parametrize("mutation", ["swap", "parent_swap", "grow"])
def test_resource_read_races_fail_closed(api: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    parent = tmp_path / "resources"
    parent.mkdir()
    target = parent / "resource.bin"
    target.write_bytes(b"original")
    read = os.read
    changed = False

    def race(descriptor: int, count: int) -> bytes:
        nonlocal changed
        raw = read(descriptor, count)
        if raw and not changed:
            changed = True
            if mutation == "swap":
                target.unlink()
                target.write_bytes(b"original")
            elif mutation == "parent_swap":
                parent.rename(tmp_path / "old")
                parent.symlink_to("old", target_is_directory=True)
            else:
                with target.open("ab") as stream:
                    stream.write(b"growth")
        return raw

    monkeypatch.setattr(os, "read", race)
    with pytest.raises(api.ClosureError, match="^MAIN_PATH_UNSAFE$"):
        api.freeze_resources(tmp_path, ("resources/resource.bin",))


def test_governed_case_collision_is_rejected(api: ModuleType, repository: Repository) -> None:
    (repository.root / "src/memcontam/Callback.py").write_bytes(b"other")
    with pytest.raises(api.ClosureError, match="^MAIN_PATH_UNSAFE$"):
        api.freeze_governed(repository.root, repository.commit)


def test_missing_commit_member_is_rejected(api: ModuleType, repository: Repository) -> None:
    git(repository.root, "rm", "scripts/build_phase13_main_registries.py")
    git(repository.root, "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
        "-c", "core.hooksPath=/dev/null", "commit", "-qm", "incomplete")
    with pytest.raises(api.ClosureError, match="^MAIN_GOVERNED_SOURCE_DRIFT$"):
        api.freeze_governed(repository.root, git(repository.root, "rev-parse", "HEAD"))


def test_git_environment_cannot_substitute_source_authority(
    api: ModuleType, repository: Repository, monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = repository.root / "other"
    git(repository.root, "clone", "-q", "--no-hardlinks", str(repository.root), str(other))
    (other / "artifact.json").write_bytes(b"{}")
    git(other, "add", "artifact.json")
    git(other, "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
        "-c", "core.hooksPath=/dev/null", "commit", "-qm", "foreign artifact")
    foreign_commit = git(other, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    with pytest.raises(api.ClosureError, match="^MAIN_GOVERNED_SOURCE_DRIFT$"):
        api.freeze_governed(repository.root, foreign_commit)
