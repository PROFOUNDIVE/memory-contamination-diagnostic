from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.final_verifier_command_records import (
    command_record,
    reconcile_summary_records,
    record_json,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import json_value_from_bytes
from memcontam.experiment.phase12.filter_challenge.final_verifier_integration_support import (
    GUARD_SENTINEL_CONTENT,
    GuardedCommand,
    install_execution_guards,
    load_yaml_object,
    run_guarded_command,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft import MFT_IDS
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.validation_summary import Task17CommandRecord


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
    validation_summary: Path,
) -> dict[str, JsonValue]:
    if scratch_root.exists():
        raise FinalVerifierError("SCRATCH_ROOT_EXISTS")
    scratch_root.mkdir(parents=True)
    guard_root = install_execution_guards(scratch_root)
    outputs, records, command_guards = _run_commands(
        repository_root, implementation_commit, search_config, fixture_root, prerequisites, scratch_root, guard_root
    )
    _reconcile_outputs(evidence_root, outputs, validation_summary, records)
    mutations, mutation_guards = _run_mutations(
        repository_root, outputs, search_config, prerequisites, scratch_root, guard_root
    )
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
    reconciled_outputs: dict[str, JsonValue] = {name: value for name, value in outputs.items()}
    return {
        "bct_family_statuses": {str(item["test_id"]): item["status"] for item in families if isinstance(item, dict)},
        "command_ids": list(COMMAND_IDS),
        "commands": [record_json(record) for record in records],
        "mft_pass_ids": mft["ordered_test_ids"],
        "mutations": mutations,
        "provider_calls_issued": 0,
        "reconciled_outputs": reconciled_outputs,
        "execution_guards": _execution_guards((*command_guards, *mutation_guards)),
    }


def _run_commands(
    repository_root: Path,
    implementation_commit: str,
    search_config: Path,
    fixture_root: Path,
    prerequisites: Path,
    scratch_root: Path,
    guard_root: Path,
) -> tuple[dict[str, dict[str, JsonValue]], tuple[Task17CommandRecord, ...], tuple[GuardedCommand, ...]]:
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
    records: list[Task17CommandRecord] = []
    guards: list[GuardedCommand] = []
    values: dict[str, dict[str, JsonValue]] = {}
    for command_id, arguments in invocations:
        guarded = run_guarded_command(repository_root, command_id, arguments, guard_root)
        result = guarded.result
        guards.append(guarded)
        if result.returncode != 0:
            raise FinalVerifierError("INTEGRATION_COMMAND_FAILED")
        records.append(command_record(repository_root, scratch_root, command_id, arguments, result.stdout, result.stderr))
        path = output / {
            "validate-search-config": "validate-search.json", "validate-selected-policy": "validate-selected-policy.json",
            "mft": "mft.json", "build-archive": "build-archive.json", "validate-archive": "archive-validation.json",
            "cost-preview": "cost.json", "bct-readiness": "bct-readiness.json",
        }[command_id]
        values[command_id] = _json(path)
    return values, tuple(records), tuple(guards)


def _reconcile_outputs(
    evidence_root: Path,
    outputs: dict[str, dict[str, JsonValue]],
    validation_summary: Path,
    records: tuple[Task17CommandRecord, ...],
) -> None:
    expected_mft = _json(evidence_root / "mft_fv5_report.json").get("report")
    expected_archive = _json(evidence_root / "archive_validation_report.json").get("report")
    expected_bct = _json(evidence_root / "bct_readiness_report.json").get("report")
    search = outputs["validate-search-config"]
    policy = outputs["validate-selected-policy"]
    cost = outputs["cost-preview"]
    if (
        search.get("valid") is not True
        or search.get("provider_calls_issued") != 0
        or policy.get("stage") != "main"
        or policy.get("selected_policy_required") is not True
        or policy.get("selected_policy_reference_valid") is not True
        or policy.get("validation_scope") != "schema_reference_only"
        or policy.get("execution_authorized") is not False
        or policy.get("provider_calls_issued") != 0
        or outputs["mft"] != expected_mft
        or outputs["build-archive"] != expected_archive
        or outputs["validate-archive"] != expected_archive
        or cost != {"candidate_estimates": [], "price_registry_id": None, "status": "not_estimated"}
        or outputs["bct-readiness"] != expected_bct
    ):
        raise FinalVerifierError("INTEGRATION_EVIDENCE_MISMATCH")
    reconcile_summary_records(validation_summary, records)


def _run_mutations(
    repository_root: Path,
    outputs: dict[str, dict[str, JsonValue]],
    search_config: Path,
    prerequisites: Path,
    scratch_root: Path,
    guard_root: Path,
) -> tuple[list[JsonValue], tuple[GuardedCommand, ...]]:
    archive = scratch_root / "archives" / "filter-v5-build-synthetic"
    mutated_archive = scratch_root / "mutated-archive"
    shutil.copytree(archive, mutated_archive)
    run = json.loads((mutated_archive / "run.json").read_text(encoding="utf-8"))
    run["implementation_commit"] = "b" * 40
    (mutated_archive / "run.json").write_text(
        json.dumps(run, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    archive_failure = run_guarded_command(
        repository_root,
        "validate-archive",
        ("--archive", str(mutated_archive), "--expected-implementation-commit", str(outputs["build-archive"]["implementation_commit"]), "--expected-search-config-hash", str(outputs["validate-search-config"]["search_config_hash"]), "--output", str(scratch_root / "archive-mutation.json")),
        guard_root,
    )
    mutated_prerequisites = scratch_root / "authorized-prerequisites.json"
    payload = json.loads(prerequisites.read_text(encoding="utf-8"))
    payload["runtime_authorization_present"] = True
    mutated_prerequisites.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    bct_failure = run_guarded_command(
        repository_root,
        "bct-readiness",
        ("--search-config", str(search_config), "--mft-report", str(scratch_root / "outputs" / "mft.json"), "--archive-report", str(scratch_root / "outputs" / "archive-validation.json"), "--execution-prerequisites", str(mutated_prerequisites), "--output", str(scratch_root / "bct-mutation.json")),
        guard_root,
    )
    provenance_report = json.loads((scratch_root / "outputs" / "mft.json").read_text(encoding="utf-8"))
    provenance_report["safety_report"]["cases"][4]["status"] = "implementation_failure"
    (scratch_root / "provenance-mutation.json").write_text(
        json.dumps(provenance_report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    provenance_failure = run_guarded_command(
        repository_root,
        "bct-readiness",
        ("--search-config", str(search_config), "--mft-report", str(scratch_root / "provenance-mutation.json"), "--archive-report", str(scratch_root / "outputs" / "archive-validation.json"), "--execution-prerequisites", str(prerequisites), "--output", str(scratch_root / "provenance-mutation-output.json")),
        guard_root,
    )
    archive_observed = _code(archive_failure.result, "IMPLEMENTATION_COMMIT_MISMATCH")
    bct_observed = _code(bct_failure.result, "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN")
    provenance_observed = _code(provenance_failure.result, "MFT_STATUS_MISMATCH")
    if (
        archive_observed != "IMPLEMENTATION_COMMIT_MISMATCH"
        or bct_observed != "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN"
        or provenance_observed != "MFT_STATUS_MISMATCH"
    ):
        raise FinalVerifierError("INTEGRATION_MUTATION_FAILED")
    return [
        {"mutation_id": "archive_bytes", "expected": "IMPLEMENTATION_COMMIT_MISMATCH", "observed": archive_observed},
        {"mutation_id": "bct_authorization", "expected": "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN", "observed": bct_observed},
        {"mutation_id": "provenance_evidence", "expected": "MFT_STATUS_MISMATCH", "observed": provenance_observed},
    ], (archive_failure, bct_failure, provenance_failure)


def _execution_guards(commands: tuple[GuardedCommand, ...]) -> dict[str, JsonValue]:
    if not commands or any(command.sentinel_content != GUARD_SENTINEL_CONTENT for command in commands):
        raise FinalVerifierError("FINAL_VERIFIER_EXECUTION_GUARD_REACHED")
    return {"bct_behavior": "not_reached", "provider_constructor": "not_reached"}


def _code(result: subprocess.CompletedProcess[str], expected: str) -> str:
    return expected if result.returncode != 0 and expected in result.stderr else "UNEXPECTED_RESULT"


def _json(path: Path) -> dict[str, JsonValue]:
    value = json_value_from_bytes(path.read_bytes(), "INTEGRATION_REPORT_INVALID")
    if not isinstance(value, dict):
        raise FinalVerifierError("INTEGRATION_REPORT_INVALID")
    return value


def _search_hash(path: Path) -> str:
    return str(load_yaml_object(path).get("search_config_hash"))
