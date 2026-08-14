from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_phase13_generic_authority import _freeze


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-m", "memcontam.cli", "phase13", *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


def test_phase13_help_exposes_only_prospective_generic_validators_for_new_surface() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "validate-authority-freeze" in result.stdout
    assert "validate-execution-registry" in result.stdout
    assert "validate-provenance" in result.stdout
    assert "calibration-v2" not in result.stdout


def test_authority_validator_runs_without_importing_live_provider_module(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    requirements = tmp_path / "requirements.json"
    requirements.write_text(
        json.dumps(
            {
                "schema_version": "phase13_authority_requirements_v1",
                "authority_hashes": {
                    "protocol": "1" * 64,
                    "experiment_design": "2" * 64,
                },
                "registry_kinds": ["execution"],
                "parameter_names": ["H_run", "backbone", "baselines", "rag_corpus", "tasks"],
            }
        ),
        encoding="utf-8",
    )
    script = """
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "memcontam.clients.openai_responses":
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from memcontam.cli import main
sys.argv = ["memcontam", "phase13", "validate-authority-freeze", *sys.argv[1:]]
main()
"""

    result = subprocess.run(
        (
            sys.executable,
            "-c",
            script,
            "--freeze",
            str(freeze),
            "--requirements",
            str(requirements),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "valid"' in result.stdout


def test_authority_validator_reports_missing_file_as_contract_error(tmp_path: Path) -> None:
    result = _run(
        "validate-authority-freeze",
        "--freeze",
        str(tmp_path / "missing-freeze.json"),
        "--requirements",
        str(tmp_path / "missing-requirements.json"),
    )

    assert result.returncode != 0
    assert "AUTHORITY_FILE_NOT_REGULAR" in result.stderr
