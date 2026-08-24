from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

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
    assert "materialize-core-datasets" in result.stdout
    assert "validate-core-datasets" in result.stdout
    assert "calibration-v2" not in result.stdout


def test_authority_validator_runs_without_importing_live_provider_module(tmp_path: Path) -> None:
    freeze_payload = {
        "schema_version": "phase13_authority_freeze_v1",
        "freeze_id": "prospective-freeze",
        "authorities": [
            {
                "role": "protocol",
                "artifact": {"path": "authority/protocol.json", "sha256": "1" * 64},
            },
            {
                "role": "experiment_design",
                "artifact": {"path": "authority/design.json", "sha256": "2" * 64},
            },
            {
                "role": "post_cutoff_addendum",
                "artifact": {"path": "authority/addendum.md", "sha256": "4" * 64},
            },
        ],
        "registries": [
            {
                "kind": "execution",
                "registry_id": "execution-future",
                "artifact": {"path": "registry/execution.json", "sha256": "3" * 64},
            }
        ],
        "parameters": {
            "backbone": "prospective-backbone",
            "H_run": 7,
            "tasks": ["future-task"],
            "baselines": ["future-baseline"],
            "rag_corpus": "future-corpus",
        },
    }
    canonical = json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode()
    freeze_payload["closure_hash"] = hashlib.sha256(canonical).hexdigest()
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(freeze_payload), encoding="utf-8")
    requirements = tmp_path / "requirements.json"
    requirements.write_text(
        json.dumps(
            {
                "schema_version": "phase13_authority_requirements_v1",
                "authority_hashes": {
                    "protocol": "1" * 64,
                    "experiment_design": "2" * 64,
                    "post_cutoff_addendum": "4" * 64,
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


def test_core_dataset_validator_reports_missing_bundle_without_traceback(tmp_path: Path) -> None:
    result = _run(
        "validate-core-datasets",
        "--root",
        str(tmp_path / "missing"),
        "--trajectory-seed",
        "1729",
    )

    assert result.returncode != 0
    assert "CORE_DATASET_FILE_NOT_REGULAR" in result.stderr
    assert "Traceback" not in result.stderr
