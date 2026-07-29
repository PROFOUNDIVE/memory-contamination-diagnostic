from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.bct import BCT_TEST_IDS
from memcontam.experiment.phase12.filter_challenge.evidence_contract import json_value_from_bytes
from memcontam.experiment.phase12.filter_challenge.final_verifier_command_records import record_json
from memcontam.experiment.phase12.filter_challenge.final_verifier_plan_checks import LEDGER_CHECKS
from memcontam.experiment.phase12.filter_challenge.final_verifier_quality import quality_commands
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import (
    FinalVerifierError,
    FinalVerifierRequest,
)
from memcontam.experiment.phase12.filter_challenge.mft import MFT_IDS
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.validation_summary import Task17ValidationSummary


_COMMAND_IDS = (
    "validate-search-config", "validate-selected-policy", "mft", "build-archive",
    "validate-archive", "cost-preview", "bct-readiness",
)
_MUTATIONS = [
    {"mutation_id": "archive_bytes", "expected": "IMPLEMENTATION_COMMIT_MISMATCH", "observed": "IMPLEMENTATION_COMMIT_MISMATCH"},
    {"mutation_id": "bct_authorization", "expected": "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN", "observed": "BCT_EXECUTION_AUTHORIZATION_FORBIDDEN"},
    {"mutation_id": "provenance_evidence", "expected": "MFT_STATUS_MISMATCH", "observed": "MFT_STATUS_MISMATCH"},
]


@dataclass(frozen=True, slots=True)
class TerminalSemanticPayload:
    base_commit: str
    bct: dict[str, JsonValue]
    checklist: list[JsonValue]
    outputs: dict[str, JsonValue]
    summary: Task17ValidationSummary


def validate_terminal_semantics(
    request: FinalVerifierRequest, bindings: dict[str, JsonValue], approvals: dict[str, dict[str, JsonValue]]
) -> TerminalSemanticPayload:
    try:
        summary = Task17ValidationSummary.model_validate_json(request.validation_summary.read_bytes())
    except ValidationError as error:
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH") from error
    plan = approvals["plan-compliance"]
    quality = approvals["code-quality"]
    integration = approvals["integration"]
    scope = approvals["scope"]
    checklist = plan.get("checklist")
    outputs = integration.get("reconciled_outputs")
    if not isinstance(checklist, list) or not isinstance(outputs, dict):
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")
    _validate_plan(checklist)
    _validate_quality(request, bindings, summary, quality, scope)
    _validate_integration(request, summary, integration, outputs)
    return TerminalSemanticPayload(summary.initial_head, _as_dict(outputs["bct-readiness"]), checklist, outputs, summary)


def _validate_plan(checklist: list[JsonValue]) -> None:
    expected = [{"clause_id": identifier, "description": description, "status": "pass"} for identifier, description in LEDGER_CHECKS.descriptions]
    if checklist != expected:
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")


def _validate_quality(
    request: FinalVerifierRequest,
    bindings: dict[str, JsonValue],
    summary: Task17ValidationSummary,
    quality: dict[str, JsonValue],
    scope: dict[str, JsonValue],
) -> None:
    implementation = bindings.get("implementation_commit")
    changed_paths = _changed_paths(request.repository_root, summary.initial_head, summary.implementation_commit)
    paths = tuple(path for path in changed_paths if path.endswith(".py"))
    expected_commands = list(quality_commands(request.repository_root, paths, summary.initial_head, summary.implementation_commit))
    gates = [{"gate_id": gate.gate_id, "status": gate.status} for gate in summary.validation_gates]
    if (
        summary.implementation_commit != implementation
        or quality.get("base_commit") != summary.initial_head
        or scope.get("base_commit") != summary.initial_head
        or quality.get("implementation_commit") != summary.implementation_commit
        or scope.get("implementation_commit") != summary.implementation_commit
        or quality.get("changed_paths") != changed_paths
        or scope.get("changed_paths") != changed_paths
        or quality.get("commands") != expected_commands
        or quality.get("findings") != []
        or [command.get("command_id") for command in expected_commands] != [gate["gate_id"] for gate in gates]
        or any(command.get("exit_code") != 0 for command in expected_commands)
        or scope.get("authority_status") != "matched"
        or scope.get("forbidden_diff_count") != 0
        or scope.get("source_dirty_allowlist") != ["?? Pilot-A 관련 기록.md"]
        or scope.get("task_worktree_clean") is not True
    ):
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")


def _validate_integration(
    request: FinalVerifierRequest,
    summary: Task17ValidationSummary,
    integration: dict[str, JsonValue],
    outputs: dict[str, JsonValue],
) -> None:
    expected_records = [record_json(record) for record in summary.command_records]
    expected_mft = _evidence_report(request.evidence_root, "mft_fv5_report.json")
    expected_archive = _evidence_report(request.evidence_root, "archive_validation_report.json")
    expected_bct = _evidence_report(request.evidence_root, "bct_readiness_report.json")
    expected_families = {test_id: "not_executed" for test_id in BCT_TEST_IDS}
    policy = _as_dict(outputs.get("validate-selected-policy"))
    search = _as_dict(outputs.get("validate-search-config"))
    mft = _as_dict(outputs.get("mft"))
    build = _as_dict(outputs.get("build-archive"))
    archive = _as_dict(outputs.get("validate-archive"))
    cost = _as_dict(outputs.get("cost-preview"))
    bct = _as_dict(outputs.get("bct-readiness"))
    if (
        integration.get("command_ids") != list(_COMMAND_IDS)
        or integration.get("commands") != expected_records
        or integration.get("mft_pass_ids") != list(MFT_IDS)
        or integration.get("provider_calls_issued") != 0
        or integration.get("bct_family_statuses") != expected_families
        or integration.get("execution_guards") != {"bct_behavior": "not_reached", "provider_constructor": "not_reached"}
        or integration.get("mutations") != _MUTATIONS
        or set(outputs) != set(_COMMAND_IDS)
        or search.get("valid") is not True
        or search.get("provider_calls_issued") != 0
        or policy.get("stage") != "main"
        or policy.get("validation_scope") != "schema_reference_only"
        or policy.get("selected_policy_required") is not True
        or policy.get("selected_policy_reference_valid") is not True
        or policy.get("execution_authorized") is not False
        or policy.get("provider_calls_issued") != 0
        or mft != expected_mft
        or build != expected_archive
        or archive != expected_archive
        or cost != {"candidate_estimates": [], "price_registry_id": None, "status": "not_estimated"}
        or bct != expected_bct
        or build.get("implementation_commit") != summary.archive_implementation_commit
        or build.get("freeze_id") != summary.archive_freeze_id
        or build.get("search_config_hash") != summary.archive_search_config_hash
        or bct.get("software_interface_status") != summary.bct_software_interface_status
        or bct.get("execution_status") != summary.bct_execution_status
        or bct.get("canonical_patch_status") != "pending_before_provider_backed_pilot_b"
        or bct.get("provider_authorization_status") != "absent"
        or bct.get("provider_calls_issued") != 0
    ):
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")


def _evidence_report(root: Path, name: str) -> dict[str, JsonValue]:
    report = json_value_from_bytes((root / name).read_bytes(), "FINAL_APPROVAL_MISMATCH")
    if not isinstance(report, dict) or not isinstance(report.get("report"), dict):
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")
    return _as_dict(report["report"])


def _changed_paths(root: Path, base_commit: str, implementation_commit: str) -> list[str]:
    result = subprocess.run(
        ("git", "-c", "core.quotepath=false", "-C", str(root), "diff", "--name-only", base_commit, implementation_commit),
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")
    return result.stdout.splitlines()


def _as_dict(value: JsonValue | None) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")
    return value
