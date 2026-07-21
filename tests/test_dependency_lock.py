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
PACKAGE_LINE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[^]]+\])?)==[^\s\\]+")
HASH_LINE = re.compile(r"^\s+--hash=sha256:[0-9a-f]{64}$")


def _metadata() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _lock_entries(path: Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    current_name: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        package_match = PACKAGE_LINE.match(line)
        if package_match:
            package_name = package_match.group("name")
            assert package_name is not None
            normalized_name = package_name.split("[", 1)[0].lower().replace("_", "-")
            entries[normalized_name] = []
            current_name = normalized_name
        elif HASH_LINE.match(line) and current_name is not None:
            entries[current_name].append(line)
    return entries


def _run(command: list[str], environment: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _environment(custom_compile_command: str | None = None) -> dict[str, str]:
    environment = os.environ | {
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_INDEX_URL": "https://pypi.org/simple",
    }
    if custom_compile_command is not None:
        environment["CUSTOM_COMPILE_COMMAND"] = custom_compile_command
    return environment


def test_dependency_policy_keeps_bounded_dependencies_and_includes_unsafe_closure() -> None:
    project = _metadata()["project"]
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.11,<3.14"
    assert project["dependencies"] == [
        "openai>=1.0,<3",
        "pydantic>=2.0,<3",
        "pyyaml>=6.0,<7",
        "sentence-transformers>=3.0,<6",
        "tiktoken>=0.7,<1",
    ]
    assert project["optional-dependencies"]["dev"] == [
        "pytest>=8.0,<10",
        "ruff>=0.5,<1",
        "mypy>=1.10,<2",
        "pip-tools==7.6.0",
    ]
    assert _metadata()["tool"]["pip-tools"]["compile"]["allow-unsafe"] is True


def test_committed_locks_are_exact_hash_pinned_dependency_closures() -> None:
    entries = {name: _lock_entries(path) for name, path in LOCKS.items()}
    assert all(
        lock_entries and all(hashes for hashes in lock_entries.values())
        for lock_entries in entries.values()
    )
    assert {
        "openai",
        "pydantic",
        "pyyaml",
        "sentence-transformers",
        "setuptools",
        "tiktoken",
    } <= entries["runtime"].keys()
    assert {"pip", "pip-tools", "setuptools"} <= entries["dev"].keys()
    assert "pip-tools==7.6.0" in LOCKS["dev"].read_text(encoding="utf-8")


def test_committed_locks_reproduce_with_mandated_pip_tools_commands() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        outputs = {
            "runtime": temporary_root / "requirements.lock",
            "dev": temporary_root / "requirements-dev.lock",
        }
        _run(
            [
                "python",
                "-m",
                "piptools",
                "compile",
                "--generate-hashes",
                "--resolver=backtracking",
                f"--output-file={outputs['runtime']}",
                "pyproject.toml",
            ],
            _environment(
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
                f"--output-file={outputs['dev']}",
                "pyproject.toml",
            ],
            _environment(
                "python -m piptools compile --extra=dev --generate-hashes "
                "--resolver=backtracking --output-file=requirements-dev.lock pyproject.toml"
            ),
        )
        assert all(outputs[name].read_bytes() == path.read_bytes() for name, path in LOCKS.items())


def test_committed_locks_support_hash_checked_dependency_resolution() -> None:
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
                "-r",
                str(lock_path),
            ],
            _environment(),
        )


def test_installed_package_version_matches_project_version() -> None:
    project = _metadata()["project"]
    assert version(project["name"]) == project["version"]
