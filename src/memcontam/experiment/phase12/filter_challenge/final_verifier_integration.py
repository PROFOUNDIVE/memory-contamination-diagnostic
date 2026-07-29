from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.evidence_contract import json_value_from_bytes
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft import MFT_IDS
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


COMMAND_IDS = (
    "validate-search-config",
    "validate-selected-policy",
    "mft",
    "build-archive",
    "validate-archive",
    "cost-preview",
    "bct-readiness",
)


def verify_integration(
    repository_root: Path,
    evidence_root: Path,
    implementation_commit: str,
    search_config: Path,
    fixture_root: Path,
    prerequisites: Path,
    scratch_root: Path,
) -> dict[str, JsonValue]:
    if scratch_root.exists():
        raise FinalVerifierError("SCRATCH_ROOT_EXISTS")
    scratch_root.mkdir(parents=True)
    outputs, commands = _run_commands(
        repository_root, implementation_commit, search_config, fixture_root, prerequisites, scratch_root
    )
    _compare_evidence(evidence_root, outputs)
    mutations = _run_mutations(repository_root, outputs, search_config, prerequisites, scratch_root)
    readiness = outputs["bct-readiness"]
    mft = outputs["mft"]
    families = readiness.get("family_statuses")
    if (
        mft.get("ordered_test_ids") != list(MFT_IDS)
        or mft.get("all_passed") is not True
        or mft.get("provider_calls_issued") != 0
        or readiness.get("provider_calls_issued") != 0
        or not isinstance(families, list)
        or any(not isinstance(item, dict) or item.get("status") != "not_executed" for item in families)
    ):
        raise FinalVerifierError("INTEGRATION_RESULT_INVALID")
    return {
        "bct_family_statuses": {str(item["test_id"]): item["status"] for item in families if isinstance(item, dict)},
        "command_ids": list(COMMAND_IDS),
        "commands": commands,
        "mft_pass_ids": mft["ordered_test_ids"],
        "mutations": mutations,
        "provider_calls_issued": 0,
    }


def _run_commands(
    repository_root: Path,
    implementation_commit: str,
    search_config: Path,
    fixture_root: Path,
    prerequisites: Path,
    scratch_root: Path,
) -> tuple[dict[str, dict[str, JsonValue]], list[JsonValue]]:
    output = scratch_root / "outputs"
    archive_root = scratch_root / "archives"
    output.mkdir()
    invocations = (
        ("validate-search-config", ("--config", str(search_config), "--output", str(output / "validate-search.json"))),
        ("validate-selected-policy", ("--search-config", str(search_config), "--selected-policy", str(fixture_root / "FilterChallengeSelectedPolicy.yaml"), "--stage", "main", "--output", str(output / "validate-selected-policy.json"))),
        ("mft", ("--search-config", str(search_config), "--fixture-root", str(fixture_root), "--output", str(output / "mft.json"))),
        ("build-archive", ("--search-config", str(search_config), "--fixture-root", str(fixture_root), "--implementation-commit", implementation_commit, "--freeze-id", "phase12-filter-v5-build-freeze-v1", "--run-id", "filter-v5-build-synthetic", "--output-root", str(archive_root), "--output", str(output / "build-archive.json"))),
        ("validate-archive", ("--archive", str(archive_root / "filter-v5-build-synthetic"), "--expected-implementation-commit", implementation_commit, "--expected-search-config-hash", _search_hash(search_config), "--output", str(output / "archive-validation.json"))),
        ("cost-preview", ("--search-config", str(search_config), "--output", str(output / "cost.json"))),
        ("bct-readiness", ("--search-config", str(search_config), "--mft-report", str(output / "mft.json"), "--archive-report", str(output / "archive-validation.json"), "--execution-prerequisites", str(prerequisites), "--output", str(output / "bct-readiness.json"))),
    )
    records: list[JsonValue] = []
    values: dict[str, dict[str, JsonValue]] = {}
    for command_id, arguments in invocations:
        result = _cli(repository_root, command_id, arguments)
        records.append(_record(command_id, result))
        if result.returncode != 0:
            raise FinalVerifierError("INTEGRATION_COMMAND_FAILED")
        path = output / {
            "validate-search-config": "validate-search.json", "validate-selected-policy": "validate-selected-policy.json",
            "mft": "mft.json", "build-archive": "build-archive.json", "validate-archive": "archive-validation.json",
            "cost-preview": "cost.json", "bct-readiness": "bct-readiness.json",
        }[command_id]
        values[command_id] = _json(path)
    return values, records


def _compare_evidence(evidence_root: Path, outputs: dict[str, dict[str, JsonValue]]) -> None:
    expected_mft = _json(evidence_root / "mft_fv5_report.json").get("report")
    expected_archive = _json(evidence_root / "archive_validation_report.json").get("report")
    expected_bct = _json(evidence_root / "bct_readiness_report.json").get("report")
    if (outputs["mft"], outputs["validate-archive"], outputs["bct-readiness"]) != (
        expected_mft, expected_archive, expected_bct
    ):
        raise FinalVerifierError("INTEGRATION_EVIDENCE_MISMATCH")


def _run_mutations(
    repository_root: Path,
    outputs: dict[str, dict[str, JsonValue]],
    search_config: Path,
    prerequisites: Path,
    scratch_root: Path,
) -> list[JsonValue]:
    archive = scratch_root / "archives" / "filter-v5-build-synthetic"
    archive_failure = _cli(
        repository_root,
        "validate-archive",
        ("--archive", str(archive), "--expected-implementation-commit", "b" * 40, "--expected-search-config-hash", str(outputs["validate-search-config"]["search_config_hash"]), "--output", str(scratch_root / "archive-mutation.json")),
    )
    mutated_prerequisites = scratch_root / "authorized-prerequisites.json"
    payload = json.loads(prerequisites.read_text(encoding="utf-8"))
    payload["runtime_authorization_present"] = True
    mutated_prerequisites.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    bct_failure = _cli(
        repository_root,
        "bct-readiness",
        ("--search-config", str(search_config), "--mft-report", str(scratch_root / "outputs" / "mft.json"), "--archive-report", str(scratch_root / "outputs" / "archive-validation.json"), "--execution-prerequisites", str(mutated_prerequisites), "--output", str(scratch_root / "bct-mutation.json")),
    )
    archive_observed = _code(archive_failure, "IMPLEMENTATION_COMMIT_MISMATCH")
    bct_observed = _code(bct_failure, "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN")
    if archive_observed != "IMPLEMENTATION_COMMIT_MISMATCH" or bct_observed != "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN":
        raise FinalVerifierError("INTEGRATION_MUTATION_FAILED")
    return [
        {"mutation_id": "archive_commit", "expected": "IMPLEMENTATION_COMMIT_MISMATCH", "observed": archive_observed},
        {"mutation_id": "bct_authorization", "expected": "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN", "observed": bct_observed},
    ]


def _cli(root: Path, command_id: str, arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    environment = os.environ | {"PYTHONPATH": str(Path(__file__).parents[4])}
    return subprocess.run((sys.executable, "-m", "memcontam.cli", "phase12", "filter-v5", command_id, *arguments), cwd=root, env=environment, check=False, capture_output=True, text=True)


def _record(command_id: str, result: subprocess.CompletedProcess[str]) -> dict[str, JsonValue]:
    return {"command_id": command_id, "exit_code": result.returncode, "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(), "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest()}


def _code(result: subprocess.CompletedProcess[str], expected: str) -> str:
    return expected if result.returncode != 0 and expected in result.stderr else "UNEXPECTED_RESULT"


def _json(path: Path) -> dict[str, JsonValue]:
    value = json_value_from_bytes(path.read_bytes(), "INTEGRATION_REPORT_INVALID")
    if not isinstance(value, dict):
        raise FinalVerifierError("INTEGRATION_REPORT_INVALID")
    return value


def _search_hash(path: Path) -> str:
    return str(_json_value(path).get("search_config_hash"))


def _json_value(path: Path) -> dict[str, JsonValue]:
    import importlib

    value = getattr(importlib.import_module("yaml"), "safe_load")(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalVerifierError("INTEGRATION_SEARCH_CONFIG_INVALID")
    return value
