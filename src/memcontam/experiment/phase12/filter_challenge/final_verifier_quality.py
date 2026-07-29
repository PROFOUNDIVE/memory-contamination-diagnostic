from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


def verify_code_quality(repository_root: Path, base_commit: str, implementation_commit: str) -> dict[str, JsonValue]:
    changed = _git(repository_root, "diff", "--name-only", base_commit, implementation_commit)
    paths = tuple(path for path in changed.splitlines() if path.endswith(".py"))
    if not paths:
        raise FinalVerifierError("CODE_QUALITY_PYTHON_FILES_REQUIRED")
    findings = [finding for path in paths for finding in _structural_findings(repository_root / path)]
    if findings:
        raise FinalVerifierError("CODE_QUALITY_REJECTED")
    commands = (
        _command(repository_root, "ruff", (sys.executable, "-m", "ruff", "check", *paths)),
        _command(repository_root, "mypy", (sys.executable, "-m", "mypy", *paths)),
        _command(repository_root, "diff-check", ("git", "diff", "--check", base_commit, implementation_commit)),
    )
    if any(command["exit_code"] != 0 for command in commands):
        raise FinalVerifierError("CODE_QUALITY_REJECTED")
    return {"commands": list(commands), "findings": []}


def _structural_findings(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return [f"syntax:{path.as_posix()}"]
    if "filter_challenge" not in path.as_posix():
        return []
    forbidden = {"OpenAICompatibleClient", "build_llm_client"}
    return [f"provider:{path.as_posix()}" for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in forbidden]


def _command(root: Path, command_id: str, command: tuple[str, ...]) -> dict[str, JsonValue]:
    result = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    return {
        "command_id": command_id,
        "exit_code": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(("git", "-C", str(root), *arguments), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise FinalVerifierError("CODE_QUALITY_BASE_INVALID")
    return result.stdout.strip()
