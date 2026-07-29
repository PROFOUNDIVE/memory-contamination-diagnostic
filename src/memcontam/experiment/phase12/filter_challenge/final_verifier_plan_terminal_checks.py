from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EVIDENCE_FILENAMES,
    canonical_json_bytes,
    sha256_path,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.validation_summary import Task17ValidationSummary

ReportLookup = Callable[[Path, str], dict[str, JsonValue] | None]


def clause_10(report: ReportLookup, root: Path, summary: JsonValue) -> bool:
    validation = report(root, "test_lint_typecheck_report.json")
    if not isinstance(summary, dict):
        return False
    try:
        contract = Task17ValidationSummary.model_validate(summary)
    except ValidationError:
        return False
    return (
        validation is not None
        and validation.get("validation_status") == "pass"
        and validation.get("provider_calls_issued") == 0
        and validation.get("command_records")
        == [record.model_dump(mode="json") for record in contract.command_records]
        and validation.get("validation_gates")
        == [gate.model_dump(mode="json") for gate in contract.validation_gates]
    )


def clause_11(report: ReportLookup, root: Path, summary: JsonValue) -> bool:
    del summary
    reports = {name: report(root, name) for name in EVIDENCE_FILENAMES}
    manifest = reports["implementation_manifest.json"]
    if manifest is None or any(value is None for value in reports.values()):
        return False
    header = manifest.get("header")
    hashes = manifest.get("reports")
    return (
        set(path.name for path in root.iterdir()) == set(EVIDENCE_FILENAMES)
        and isinstance(header, dict)
        and set(header)
        == {
            "amendment", "authority_hashes", "config_schema_hashes", "implementation_commit",
            "plan_sha256", "policy", "validation_summary_sha256",
        }
        and "evidence_commit" not in header
        and "implementation_manifest_sha256" not in header
        and isinstance(hashes, dict)
        and set(hashes) == set(EVIDENCE_FILENAMES[1:])
        and all(value is not None and value.get("header") == header for value in reports.values())
        and all(hashes.get(name) == sha256_path(root / name) for name in EVIDENCE_FILENAMES[1:])
        and all(
            value is not None and canonical_json_bytes(value) == (root / name).read_bytes()
            for name, value in reports.items()
        )
    )


def clause_12(report: ReportLookup, root: Path, summary: JsonValue) -> bool:
    policy_report = report(root, "policy_schema_hashes.json")
    policy_header = policy_report.get("header") if policy_report is not None else None
    policy = policy_header.get("policy") if isinstance(policy_header, dict) else None
    manifest = report(root, "implementation_manifest.json")
    header = manifest.get("header") if manifest is not None else None
    readiness_report = report(root, "bct_readiness_report.json")
    readiness = readiness_report.get("report") if readiness_report is not None else None
    return (
        isinstance(summary, dict)
        and isinstance(header, dict)
        and isinstance(readiness, dict)
        and summary.get("implementation_commit") == header.get("implementation_commit")
        and isinstance(policy, dict)
        and policy.get("canonical_patch_status") == "pending_before_provider_backed_pilot_b"
        and readiness.get("canonical_patch_status") == "pending_before_provider_backed_pilot_b"
        and readiness.get("scientific_inventory_status") == "pending_freeze"
        and readiness.get("provider_authorization_status") == "absent"
    )
