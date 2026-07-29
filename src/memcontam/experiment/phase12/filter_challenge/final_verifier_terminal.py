from __future__ import annotations

import subprocess
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.bct import BCT_TEST_IDS
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EVIDENCE_FILENAMES,
    json_value_from_bytes,
    sha256_path,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_plan_checks import (
    LEDGER_CHECKS,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_terminal_semantics import (
    validate_terminal_semantics,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import (
    FinalVerifierError,
    FinalVerifierRequest,
)
from memcontam.experiment.phase12.filter_challenge.mft import MFT_IDS
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


_APPROVAL_MODES = ("plan-compliance", "code-quality", "integration", "scope")
_COMMAND_IDS = (
    "validate-search-config", "validate-selected-policy", "mft", "build-archive",
    "validate-archive", "cost-preview", "bct-readiness",
)


def build_terminal_report(
    request: FinalVerifierRequest, bindings: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    approvals = _load_approvals(request, bindings)
    semantic = validate_terminal_semantics(request, bindings, approvals)
    integration = approvals["integration"]
    outputs = semantic.outputs
    checklist = semantic.checklist
    bct = semantic.bct
    base_commit = semantic.base_commit
    head = _git(request.repository_root, "rev-parse", "HEAD")
    ordered_commit_series: list[JsonValue] = [
        value
        for value in _git(
            request.repository_root, "rev-list", "--reverse", f"{base_commit}..{head}"
        ).splitlines()
    ]
    worktree_status: list[JsonValue] = [
        value
        for value in _git(request.repository_root, "status", "--porcelain=v1").splitlines()
    ]
    implemented_components: list[JsonValue] = [
        item.get("clause_id") if isinstance(item, dict) else None for item in checklist
    ]
    return {
        "bindings": bindings,
        "build_status": "FILTER_V5_BUILD_AND_MFT_COMPLETE",
        "initial_head": base_commit,
        "implementation_commit": bindings["implementation_commit"],
        "evidence_commit": bindings["evidence_commit"],
        "final_head": head,
        "ordered_commit_series": ordered_commit_series,
        "worktree_status": worktree_status,
        "implemented_components": implemented_components,
        "mft_result": integration.get("mft_pass_ids"),
        "provenance_result": checklist[3],
        "route_result": checklist[4],
        "archive_result": outputs.get("validate-archive"),
        "bct_result": bct,
        "canonical_patch_status": bct["canonical_patch_status"],
        "remaining_scientific_choices": _remaining_choices(bct),
        "next_gate_status": "READY_FOR_AUTHORIZED_FILTER_V5_BEHAVIORAL_CAPABILITY_RUN",
        "provider_calls_issued": integration.get("provider_calls_issued"),
        "evidence_paths_and_hashes": [
            {"path": name, "sha256": sha256_path(request.evidence_root / name)}
            for name in EVIDENCE_FILENAMES
        ],
    }


def _remaining_choices(bct: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "inventory": bct["scientific_inventory_status"],
        "operational_suite": "unresolved", "decision_rule": "unresolved", "kappa": "unresolved",
        "coverage_contract": "unresolved", "probe_count": "unresolved", "replicate_count": "unresolved",
        "retry_count": "unresolved", "canonicalizer": "unresolved", "tolerance": "unresolved",
        "evaluability_rate": "unresolved", "inclusion_rate": "unresolved", "ordinary_route_rate": "unresolved",
        "price_registry": "unresolved", "monetary_cost_cap": "unresolved", "latency_cap": "unresolved",
        "ci_procedure": "unresolved", "constraint_order": "unresolved", "tie_break": "unresolved",
        "provider_authorization": bct["provider_authorization_status"],
    }


def _load_approvals(
    request: FinalVerifierRequest, bindings: dict[str, JsonValue]
) -> dict[str, dict[str, JsonValue]]:
    if len(request.approval_paths) != len(_APPROVAL_MODES):
        raise FinalVerifierError("FINAL_APPROVALS_REQUIRED")
    approvals: dict[str, dict[str, JsonValue]] = {}
    for mode, path in zip(_APPROVAL_MODES, request.approval_paths, strict=True):
        value = json_value_from_bytes(path.read_bytes(), "FINAL_APPROVAL_INVALID")
        if (
            not isinstance(value, dict)
            or value.get("mode") != mode
            or value.get("verdict") != "APPROVE"
            or value.get("bindings") != bindings
        ):
            raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")
        approvals[mode] = value
    if not _complete_payloads(approvals):
        raise FinalVerifierError("FINAL_APPROVAL_MISMATCH")
    return approvals


def _complete_payloads(approvals: dict[str, dict[str, JsonValue]]) -> bool:
    plan = approvals["plan-compliance"]
    quality = approvals["code-quality"]
    integration = approvals["integration"]
    scope = approvals["scope"]
    checklist = plan.get("checklist")
    commands = quality.get("commands")
    integration_commands = integration.get("commands")
    mutations = integration.get("mutations")
    families = integration.get("bct_family_statuses")
    return (
        checklist == [
            {"clause_id": clause_id, "description": description, "status": "pass"}
            for clause_id, description in LEDGER_CHECKS.descriptions
        ]
        and isinstance(commands, list)
        and _commands_complete(commands, ("ruff", "mypy", "diff-check"))
        and quality.get("findings") == []
        and integration.get("command_ids") == list(_COMMAND_IDS)
        and isinstance(integration_commands, list)
        and _commands_complete(integration_commands, _COMMAND_IDS)
        and integration.get("mft_pass_ids") == list(MFT_IDS)
        and isinstance(mutations, list)
        and {item.get("mutation_id") for item in mutations if isinstance(item, dict)}
        == {"archive_bytes", "bct_authorization", "provenance_evidence"}
        and all(
            isinstance(item, dict) and item.get("observed") == item.get("expected")
            for item in mutations
        )
        and families == {test_id: "not_executed" for test_id in BCT_TEST_IDS}
        and integration.get("provider_calls_issued") == 0
        and integration.get("execution_guards")
        == {"bct_behavior": "not_reached", "provider_constructor": "not_reached"}
        and _reconciled_outputs_complete(integration.get("reconciled_outputs"))
        and scope.get("authority_status") == "matched"
        and scope.get("forbidden_diff_count") == 0
        and scope.get("task_worktree_clean") is True
        and scope.get("source_dirty_allowlist") == ["?? Pilot-A 관련 기록.md"]
        and isinstance(scope.get("base_commit"), str)
    )


def _commands_complete(values: list[JsonValue], expected: tuple[str, ...]) -> bool:
    return len(values) == len(expected) and all(
        isinstance(value, dict)
        and value.get("command_id") == command_id
        and value.get("exit_code") == 0
        and isinstance(value.get("stdout_sha256"), str)
        and isinstance(value.get("stderr_sha256"), str)
        for command_id, value in zip(expected, values, strict=True)
    )


def _reconciled_outputs_complete(value: JsonValue | None) -> bool:
    return isinstance(value, dict) and set(value) == set(_COMMAND_IDS) and all(
        isinstance(item, dict) for item in value.values()
    )


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments), check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise FinalVerifierError("FINAL_GIT_INVALID")
    return result.stdout.strip()
