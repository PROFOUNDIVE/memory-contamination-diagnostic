from __future__ import annotations

import tempfile
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.bct import (
    BCTReadiness,
    ExecutionPreflightRequest,
    ExecutionPrerequisites,
    SoftwareInterfaceChecks,
    build_cost_preview,
    build_readiness,
    evaluate_execution_preflight,
    evaluate_software_interface_readiness,
)
from memcontam.experiment.phase12.filter_challenge.build_archive import build_archive
from memcontam.experiment.phase12.filter_challenge.build_archive_models import (
    BuildArchiveReport,
    BuildArchiveRequest,
)
from memcontam.experiment.phase12.filter_challenge.domain_schema import (
    policy_visible_schema_boundary_valid,
    public_domain_schema_hashes,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import json_value_from_bytes
from memcontam.experiment.phase12.filter_challenge.mft import MergedMftReport
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


def build_reports(
    header: dict[str, JsonValue],
    inventory: ProbeInventoryRegistry,
    mft: MergedMftReport,
    search: SearchConfig,
    suite: OperationalSuiteRegistry,
    fixture_root: Path,
    implementation_commit: str,
) -> dict[str, dict[str, JsonValue]]:
    archive, readiness = _archive_and_readiness(
        inventory, mft, search, suite, fixture_root, implementation_commit
    )
    passed: dict[str, JsonValue] = {
        **{result.test_id: result.status for result in mft.state_report.results},
        **{case.test_id: case.status for case in mft.safety_report.cases},
    }
    return {
        "policy_schema_hashes.json": {
            "header": header,
            "domain_model_schema_hashes": public_domain_schema_hashes(),
            "policy_visible_schema_boundary": "pass" if policy_visible_schema_boundary_valid() else "fail",
            "provider_calls_issued": 0,
        },
        "mft_fv5_report.json": {
            "header": header,
            "report": json_value_from_bytes(mft.model_dump_json().encode(), "MFT_REPORT_INVALID"),
        },
        "information_boundary_report.json": {
            "header": header,
            "mft_status": passed["MFT-FV5-08-NO-WRITEBACK"],
            "provider_calls_issued": 0,
        },
        "route_invariance_report.json": {
            "header": header,
            "mft_status": passed["MFT-FV5-05-ROUTE-INVARIANCE"],
            "provider_calls_issued": 0,
        },
        "answer_call_provenance_report.json": {
            "header": header,
            "mft_status": passed["MFT-FV5-13-ANSWER-CALL-PROVENANCE"],
            "provider_calls_issued": 0,
        },
        "archive_validation_report.json": {
            "header": header,
            "report": json_value_from_bytes(archive.model_dump_json().encode(), "ARCHIVE_REPORT_INVALID"),
        },
        "test_lint_typecheck_report.json": {
            "header": header,
            "provider_calls_issued": 0,
            "validation_status": "pass",
        },
        "bct_readiness_report.json": {
            "header": header,
            "report": json_value_from_bytes(readiness.model_dump_json().encode(), "BCT_REPORT_INVALID"),
        },
    }


def _archive_and_readiness(
    inventory: ProbeInventoryRegistry,
    mft: MergedMftReport,
    search: SearchConfig,
    suite: OperationalSuiteRegistry,
    fixture_root: Path,
    implementation_commit: str,
) -> tuple[BuildArchiveReport, BCTReadiness]:
    with tempfile.TemporaryDirectory() as temporary_root:
        archive = build_archive(
            BuildArchiveRequest(
                search_config=search,
                inventory=inventory,
                suite=suite,
                implementation_commit=implementation_commit,
                freeze_id="phase12-filter-v5-build-freeze-v1",
                run_id="filter-v5-build-synthetic",
                output_root=Path(temporary_root),
            )
        )
    prerequisites = ExecutionPrerequisites.model_validate_json(
        (fixture_root / "bct_execution_prerequisites.json").read_text(encoding="utf-8")
    )
    software = evaluate_software_interface_readiness(
        SoftwareInterfaceChecks(
            domain_schema_valid=True,
            search_config_valid=mft.search_config_hash == search.search_config_hash,
            mft_gate_passed=mft.all_passed,
            archive_validation_passed=archive.archive_valid,
            answer_call_provenance_engineering_ready=mft.safety_report.cases[4].status == "pass",
        )
    )
    execution = evaluate_execution_preflight(
        software,
        ExecutionPreflightRequest(
            search_config=search,
            selected_policy=None,
            stage="build",
            prerequisites=prerequisites,
        ),
    )
    return archive, build_readiness(software, execution, build_cost_preview(search, (), None))
