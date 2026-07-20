from __future__ import annotations

import os
import re
import subprocess
import tempfile
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCKS = {
    "runtime": ROOT / "requirements.lock",
    "dev": ROOT / "requirements-dev.lock",
}
EXPECTED_RUNTIME_DEPENDENCIES = (
    "openai>=1.0,<3",
    "pydantic>=2.0,<3",
    "pyyaml>=6.0,<7",
    "sentence-transformers>=3.0,<6",
)
EXPECTED_DEV_DEPENDENCIES = (
    "pytest>=8.0,<10",
    "ruff>=0.5,<1",
    "mypy>=1.10,<2",
    "pip-tools==7.6.0",
)
PACKAGE_LINE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^\s\\]+)")
HASH_LINE = re.compile(r"^\s+--hash=sha256:(?P<digest>[0-9a-f]{64})$")


def _metadata() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_entries(path: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    entries: dict[str, tuple[str, tuple[str, ...]]] = {}
    current_name: str | None = None
    current_version: str | None = None
    current_hashes: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        package_match = PACKAGE_LINE.match(line)
        if package_match:
            if current_name is not None and current_version is not None:
                entries[current_name] = (current_version, tuple(current_hashes))
            current_name = _canonicalize(package_match.group("name"))
            current_version = package_match.group("version")
            current_hashes = []
            continue

        hash_match = HASH_LINE.match(line)
        if hash_match and current_name is not None:
            current_hashes.append(hash_match.group("digest"))

    if current_name is not None and current_version is not None:
        entries[current_name] = (current_version, tuple(current_hashes))
    return entries


def _compile_environment(custom_command: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_INDEX_URL": "https://pypi.org/simple",
            "CUSTOM_COMPILE_COMMAND": custom_command,
        }
    )
    return environment


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_dependency_metadata_keeps_bounded_runtime_and_dev_policy() -> None:
    project = _metadata()["project"]

    assert project["requires-python"] == ">=3.11,<3.14"
    assert tuple(project["dependencies"]) == EXPECTED_RUNTIME_DEPENDENCIES
    assert tuple(project["optional-dependencies"]["dev"]) == EXPECTED_DEV_DEPENDENCIES


def test_committed_locks_are_present_exactly_pinned_and_hash_locked() -> None:
    project = _metadata()["project"]
    runtime_names = {
        _canonicalize(requirement.split("<", 1)[0].split(">=", 1)[0])
        for requirement in project["dependencies"]
    }
    dev_names = {
        _canonicalize(requirement.split("<", 1)[0].split(">=", 1)[0].split("==", 1)[0])
        for requirement in project["optional-dependencies"]["dev"]
    }

    for lock_path in LOCKS.values():
        assert lock_path.is_file(), lock_path
        entries = _lock_entries(lock_path)
        assert entries, lock_path
        assert all(version and hashes for version, hashes in entries.values())
        assert all(len(digest) == 64 for _, hashes in entries.values() for digest in hashes)

    runtime_entries = _lock_entries(LOCKS["runtime"])
    dev_entries = _lock_entries(LOCKS["dev"])
    assert runtime_names <= runtime_entries.keys()
    assert runtime_names <= dev_entries.keys()
    assert dev_names <= dev_entries.keys()
    assert runtime_entries["openai"][0]
    assert dev_entries["pip-tools"][0]
    assert dev_entries["pip-tools"][0] == "7.6.0"

    runtime_header = LOCKS["runtime"].read_text(encoding="utf-8")
    dev_header = LOCKS["dev"].read_text(encoding="utf-8")
    assert "python -m piptools compile --generate-hashes --resolver=backtracking" in runtime_header
    assert "python -m piptools compile --extra=dev --generate-hashes --resolver=backtracking" in dev_header


def test_committed_locks_reproduce_with_mandated_pip_tools_commands() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        runtime_output = temporary_root / "requirements.lock"
        dev_output = temporary_root / "requirements-dev.lock"
        _run(
            [
                "python",
                "-m",
                "piptools",
                "compile",
                "--generate-hashes",
                "--resolver=backtracking",
                f"--output-file={runtime_output}",
                "pyproject.toml",
            ],
            environment=_compile_environment(
                "python -m piptools compile --generate-hashes --resolver=backtracking "
                "--output-file=requirements.lock pyproject.toml"
            ),
        )
        _run(
            [
                "python",
                "-m",
                "piptools",
                "compile",
                "--extra=dev",
                "--generate-hashes",
                "--resolver=backtracking",
                f"--output-file={dev_output}",
                "pyproject.toml",
            ],
            environment=_compile_environment(
                "python -m piptools compile --extra=dev --generate-hashes "
                "--resolver=backtracking --output-file=requirements-dev.lock pyproject.toml"
            ),
        )

        assert runtime_output.read_bytes() == LOCKS["runtime"].read_bytes()
        assert dev_output.read_bytes() == LOCKS["dev"].read_bytes()


def test_committed_locks_support_hash_checked_dry_run_entries() -> None:
    environment = os.environ.copy()
    environment.update({"PIP_CONFIG_FILE": "/dev/null", "PIP_INDEX_URL": "https://pypi.org/simple"})
    for lock_path in LOCKS.values():
        _run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--require-hashes",
                "--no-deps",
                "-r",
                str(lock_path),
            ],
            environment=environment,
        )


def test_installed_package_version_matches_project_version() -> None:
    project = _metadata()["project"]
    assert version(project["name"]) == project["version"]
