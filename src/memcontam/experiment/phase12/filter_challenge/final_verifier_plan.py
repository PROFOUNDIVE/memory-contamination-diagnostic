from __future__ import annotations

from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.evidence_contract import json_value_from_bytes
from memcontam.experiment.phase12.filter_challenge.final_verifier_types import FinalVerifierError
from memcontam.experiment.phase12.filter_challenge.mft import MFT_IDS
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


def verify_plan_compliance(evidence_root: Path, summary: JsonValue) -> dict[str, JsonValue]:
    reports = {name: _report(evidence_root, name) for name in (
        "policy_schema_hashes.json", "mft_fv5_report.json", "information_boundary_report.json",
        "route_invariance_report.json", "answer_call_provenance_report.json",
        "archive_validation_report.json", "test_lint_typecheck_report.json", "bct_readiness_report.json",
    )}
    header_value = reports["policy_schema_hashes.json"].get("header")
    mft_value = reports["mft_fv5_report.json"].get("report")
    archive_value = reports["archive_validation_report.json"].get("report")
    readiness_value = reports["bct_readiness_report.json"].get("report")
    if not isinstance(header_value, dict):
        raise FinalVerifierError("PLAN_COMPLIANCE_REJECTED")
    if not isinstance(mft_value, dict):
        raise FinalVerifierError("PLAN_COMPLIANCE_REJECTED")
    if not isinstance(archive_value, dict):
        raise FinalVerifierError("PLAN_COMPLIANCE_REJECTED")
    if not isinstance(readiness_value, dict):
        raise FinalVerifierError("PLAN_COMPLIANCE_REJECTED")
    header: dict[str, JsonValue] = header_value
    mft: dict[str, JsonValue] = mft_value
    archive: dict[str, JsonValue] = archive_value
    readiness: dict[str, JsonValue] = readiness_value
    policy = header.get("policy")
    execution_counts = mft.get("execution_counts")
    statuses = (
        isinstance(policy, dict) and policy.get("identity") == "Filter-Challenge-v1",
        isinstance(header.get("config_schema_hashes"), dict),
        mft.get("ordered_test_ids") == list(MFT_IDS) and mft.get("all_passed") is True,
        isinstance(execution_counts, list)
        and all(isinstance(item, dict) and item.get("count") == 1 for item in execution_counts),
        reports["information_boundary_report.json"].get("mft_status") == "pass",
        reports["route_invariance_report.json"].get("mft_status") == "pass",
        reports["answer_call_provenance_report.json"].get("mft_status") == "pass",
        archive.get("archive_valid") is True and archive.get("provider_calls_issued") == 0,
        readiness.get("software_interface_status") == "ready" and readiness.get("execution_status") == "blocked",
        _families_unexecuted(readiness),
        readiness.get("behavioral_calls_executed") is False and readiness.get("provider_calls_issued") == 0,
        isinstance(summary, dict) and summary.get("provider_calls_issued") == 0,
    )
    if not all(statuses):
        raise FinalVerifierError("PLAN_COMPLIANCE_REJECTED")
    return {"checklist": [{"clause": f"ledger-{index}", "status": "pass"} for index in range(1, 13)]}


def _families_unexecuted(readiness: dict[str, JsonValue]) -> bool:
    families = readiness.get("family_statuses")
    return (
        isinstance(families, list)
        and len(families) == 4
        and all(isinstance(item, dict) and item.get("status") == "not_executed" for item in families)
    )


def _report(root: Path, name: str) -> dict[str, JsonValue]:
    value = json_value_from_bytes((root / name).read_bytes(), "EVIDENCE_REPORT_INVALID")
    if not isinstance(value, dict):
        raise FinalVerifierError("EVIDENCE_REPORT_INVALID")
    return value
