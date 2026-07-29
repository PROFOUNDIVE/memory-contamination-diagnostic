from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
from memcontam.experiment.phase12.filter_challenge.cli import _load_build_inputs
from memcontam.experiment.phase12.filter_challenge.contracts import FilterPolicyIdentity
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    AMENDMENT,
    AUTHORITY_HASHES,
    EVIDENCE_FILENAMES,
    NON_MANIFEST_FILENAMES,
    POLICY,
    EvidenceBuildError,
    EvidenceBundle,
    canonical_json_bytes,
    descriptor_sha256,
    json_value_from_bytes,
    require_clean_repository,
    sha256_bytes,
    sha256_path,
)
from memcontam.experiment.phase12.filter_challenge.mft import MergedMftReport, build_mft_report
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


_HEADER_FIELDS: Final = {
    "amendment",
    "authority_hashes",
    "config_schema_hashes",
    "implementation_commit",
    "plan_sha256",
    "policy",
    "validation_summary_sha256",
}


@dataclass(frozen=True, slots=True)
class EvidenceBuildRequest:
    repository_root: Path
    plan: Path
    expected_plan_sha256: str
    implementation_commit: str
    search_config: Path
    fixture_root: Path
    validation_summary: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class EvidenceBindings:
    implementation_commit: str
    plan_sha256: str
    validation_summary_sha256: str


@dataclass(frozen=True, slots=True)
class EvidenceInputs:
    inventory: ProbeInventoryRegistry
    mft: MergedMftReport
    search: SearchConfig
    suite: OperationalSuiteRegistry


def build_evidence_bundle(request: EvidenceBuildRequest) -> EvidenceBundle:
    plan_hash = descriptor_sha256(request.plan).sha256
    if plan_hash != request.expected_plan_sha256:
        raise EvidenceBuildError("PLAN_SHA256_MISMATCH")
    require_clean_repository(request.repository_root, request.implementation_commit)
    summary_hash = sha256_path(request.validation_summary)
    summary = json_value_from_bytes(
        request.validation_summary.read_bytes(), "VALIDATION_SUMMARY_INVALID"
    )
    _validate_summary(summary, request.expected_plan_sha256, request.implementation_commit)
    search, inventory, suite = _load_build_inputs(request.search_config, request.fixture_root)
    mft = build_mft_report(search, inventory, suite)
    bindings = EvidenceBindings(request.implementation_commit, plan_hash, summary_hash)
    inputs = EvidenceInputs(inventory=inventory, mft=mft, search=search, suite=suite)
    header = _header(bindings, inputs)
    reports = _reports(header, inputs, request)
    if request.output_root.exists():
        validate_evidence_bundle(request.output_root)
        raise EvidenceBuildError("EVIDENCE_OUTPUT_EXISTS")
    request.output_root.mkdir(parents=True)
    for name, report in reports.items():
        (request.output_root / name).write_bytes(canonical_json_bytes(report))
    manifest_reports: dict[str, JsonValue] = {
        name: sha256_path(request.output_root / name) for name in NON_MANIFEST_FILENAMES
    }
    manifest: dict[str, JsonValue] = {
        "header": header,
        "reports": manifest_reports,
        "schema_version": "filter_challenge_implementation_manifest_v1",
    }
    manifest_path = request.output_root / EVIDENCE_FILENAMES[0]
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return validate_evidence_bundle(request.output_root)


def validate_evidence_bundle(root: Path) -> EvidenceBundle:
    if not root.is_dir() or {path.name for path in root.iterdir()} != set(EVIDENCE_FILENAMES):
        raise EvidenceBuildError("EVIDENCE_FILE_SET_INVALID")
    reports = {
        name: json_value_from_bytes((root / name).read_bytes(), "EVIDENCE_JSON_INVALID")
        for name in EVIDENCE_FILENAMES
    }
    if any(canonical_json_bytes(report) != (root / name).read_bytes() for name, report in reports.items()):
        raise EvidenceBuildError("EVIDENCE_CANONICAL_JSON_REQUIRED")
    first = reports[EVIDENCE_FILENAMES[0]]
    if not isinstance(first, dict):
        raise EvidenceBuildError("EVIDENCE_GRAPH_MISMATCH")
    header = first.get("header")
    report_hashes = first.get("reports")
    if not isinstance(header, dict) or not isinstance(report_hashes, dict):
        raise EvidenceBuildError("EVIDENCE_GRAPH_MISMATCH")
    if set(header) != _HEADER_FIELDS or set(report_hashes) != set(NON_MANIFEST_FILENAMES):
        raise EvidenceBuildError("EVIDENCE_GRAPH_MISMATCH")
    if any(
        not isinstance(report, dict) or report.get("header") != header
        for report in reports.values()
    ):
        raise EvidenceBuildError("EVIDENCE_GRAPH_MISMATCH")
    if any(
        report_hashes.get(name) != sha256_path(root / name) for name in NON_MANIFEST_FILENAMES
    ):
        raise EvidenceBuildError("EVIDENCE_GRAPH_MISMATCH")
    return EvidenceBundle(
        header=header,
        implementation_manifest_sha256=sha256_path(root / EVIDENCE_FILENAMES[0]),
        root=root,
    )


def _validate_summary(summary: JsonValue, plan_hash: str, implementation_commit: str) -> None:
    if not isinstance(summary, dict):
        raise EvidenceBuildError("VALIDATION_SUMMARY_INVALID")
    if summary.get("reviewed_plan_sha256") != plan_hash:
        raise EvidenceBuildError("VALIDATION_SUMMARY_PLAN_MISMATCH")
    if summary.get("implementation_commit") != implementation_commit:
        raise EvidenceBuildError("VALIDATION_SUMMARY_COMMIT_MISMATCH")
    if summary.get("provider_calls_issued") != 0:
        raise EvidenceBuildError("VALIDATION_SUMMARY_PROVIDER_CALLS_INVALID")


def _header(bindings: EvidenceBindings, inputs: EvidenceInputs) -> dict[str, JsonValue]:
    amendment: dict[str, JsonValue] = {name: value for name, value in AMENDMENT.items()}
    authority_hashes: dict[str, JsonValue] = {name: value for name, value in AUTHORITY_HASHES.items()}
    policy: dict[str, JsonValue] = {name: value for name, value in POLICY.items()}
    schema_hashes: dict[str, JsonValue] = {
        "domain_contract": sha256_bytes(canonical_json_bytes(FilterPolicyIdentity.model_json_schema())),
        "mft": sha256_bytes(canonical_json_bytes(MergedMftReport.model_json_schema())),
        "search_config": inputs.search.search_config_hash,
        "search_config_schema": sha256_bytes(canonical_json_bytes(SearchConfig.model_json_schema())),
    }
    return {
        "amendment": amendment,
        "authority_hashes": authority_hashes,
        "config_schema_hashes": schema_hashes,
        "implementation_commit": bindings.implementation_commit,
        "plan_sha256": bindings.plan_sha256,
        "policy": policy,
        "validation_summary_sha256": bindings.validation_summary_sha256,
    }


def _reports(
    header: dict[str, JsonValue],
    inputs: EvidenceInputs,
    request: EvidenceBuildRequest,
) -> dict[str, dict[str, JsonValue]]:
    archive, readiness = _archive_and_readiness(inputs, request)
    mft_value = json_value_from_bytes(inputs.mft.model_dump_json().encode(), "MFT_REPORT_INVALID")
    readiness_value = json_value_from_bytes(readiness.model_dump_json().encode(), "BCT_REPORT_INVALID")
    archive_value = json_value_from_bytes(archive.model_dump_json().encode(), "ARCHIVE_REPORT_INVALID")
    passed: dict[str, JsonValue] = {
        **{result.test_id: result.status for result in inputs.mft.state_report.results},
        **{case.test_id: case.status for case in inputs.mft.safety_report.cases},
    }
    return {
        "policy_schema_hashes.json": {"header": header, "provider_calls_issued": 0},
        "mft_fv5_report.json": {"header": header, "report": mft_value},
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
        "archive_validation_report.json": {"header": header, "report": archive_value},
        "test_lint_typecheck_report.json": {"header": header, "provider_calls_issued": 0},
        "bct_readiness_report.json": {"header": header, "report": readiness_value},
    }


def _archive_and_readiness(
    inputs: EvidenceInputs, request: EvidenceBuildRequest
) -> tuple[BuildArchiveReport, BCTReadiness]:
    with tempfile.TemporaryDirectory() as temporary_root:
        archive = build_archive(
            BuildArchiveRequest(
                search_config=inputs.search,
                inventory=inputs.inventory,
                suite=inputs.suite,
                implementation_commit=request.implementation_commit,
                freeze_id="phase12-filter-v5-build-freeze-v1",
                run_id="filter-v5-build-synthetic",
                output_root=Path(temporary_root),
            )
        )
    prerequisites = ExecutionPrerequisites.model_validate_json(
        (request.fixture_root / "bct_execution_prerequisites.json").read_text(encoding="utf-8")
    )
    software = evaluate_software_interface_readiness(
        SoftwareInterfaceChecks(
            domain_schema_valid=True,
            search_config_valid=inputs.mft.search_config_hash == inputs.search.search_config_hash,
            mft_gate_passed=inputs.mft.all_passed,
            archive_validation_passed=archive.archive_valid,
            answer_call_provenance_engineering_ready=(
                inputs.mft.safety_report.cases[4].status == "pass"
            ),
        )
    )
    execution = evaluate_execution_preflight(
        software,
        ExecutionPreflightRequest(
            search_config=inputs.search,
            selected_policy=None,
            stage="build",
            prerequisites=prerequisites,
        ),
    )
    return archive, build_readiness(software, execution, build_cost_preview(inputs.search, (), None))
