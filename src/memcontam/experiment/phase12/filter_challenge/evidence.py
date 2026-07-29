from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.cli import _load_build_inputs
from memcontam.experiment.phase12.filter_challenge.contracts import FilterPolicyIdentity
from memcontam.experiment.phase12.filter_challenge.evidence_reports import build_reports
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
from memcontam.experiment.phase12.filter_challenge.validation_summary import Task17ValidationSummary


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
    summary = _validate_summary(
        request.validation_summary.read_bytes(), request.expected_plan_sha256, request.implementation_commit
    )
    search, inventory, suite = _load_build_inputs(request.search_config, request.fixture_root)
    mft = build_mft_report(search, inventory, suite)
    bindings = EvidenceBindings(request.implementation_commit, plan_hash, summary_hash)
    inputs = EvidenceInputs(inventory=inventory, mft=mft, search=search, suite=suite)
    header = _header(bindings, inputs)
    reports = build_reports(
        header,
        inputs.inventory,
        inputs.mft,
        inputs.search,
        inputs.suite,
        request.fixture_root,
        request.implementation_commit,
        summary,
    )
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
        raise EvidenceBuildError("EVIDENCE_HASH_MISMATCH")
    return EvidenceBundle(
        header=header,
        implementation_manifest_sha256=sha256_path(root / EVIDENCE_FILENAMES[0]),
        root=root,
    )


def _validate_summary(summary_bytes: bytes, plan_hash: str, implementation_commit: str) -> Task17ValidationSummary:
    try:
        summary = Task17ValidationSummary.model_validate_json(summary_bytes)
    except ValueError as error:
        raise EvidenceBuildError("VALIDATION_SUMMARY_INVALID") from error
    if summary.reviewed_plan_sha256 != plan_hash:
        raise EvidenceBuildError("VALIDATION_SUMMARY_PLAN_MISMATCH")
    if summary.implementation_commit != implementation_commit:
        raise EvidenceBuildError("VALIDATION_SUMMARY_COMMIT_MISMATCH")
    if summary.provider_calls_issued != 0:
        raise EvidenceBuildError("VALIDATION_SUMMARY_PROVIDER_CALLS_INVALID")
    return summary


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
