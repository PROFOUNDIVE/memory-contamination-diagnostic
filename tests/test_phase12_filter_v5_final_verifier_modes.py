from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal

import pytest
from .phase12_filter_v5_summary_cases import complete_validation_summary

from memcontam.experiment.phase12.filter_challenge.evidence import (
    EvidenceBundle,
    EvidenceBuildRequest,
    build_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    canonical_json_bytes,
    json_value_from_bytes,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier import (
    FinalVerifierError,
    FinalVerifierRequest,
    verify_final_report,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_plan import (
    verify_plan_compliance,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_quality import (
    _structural_findings,
    verify_code_quality,
)
from memcontam.experiment.phase12.filter_challenge import final_verifier_integration
from memcontam.experiment.phase12.filter_challenge.final_verifier_command_records import (
    reconcile_summary_records,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_integration_support import (
    install_execution_guards,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.validation_summary import Task17CommandRecord


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "phase12" / "filter_v5"
Mutation = Literal["forbidden_diff", "invalid_python", "mft_failure", "source_dirty"] | None
_COMMAND_RECORDS: dict[tuple[str, str], tuple[bytes, ...]] = {}


@dataclass(frozen=True, slots=True)
class VerifierFixture:
    base_commit: str
    evidence: EvidenceBuildRequest
    source_repository: Path


@dataclass(frozen=True, slots=True)
class FixtureTemplate:
    base_commit: str
    implementation_commit: str
    plan_bytes: bytes
    repository_bundle: bytes
    repository_overrides: tuple[FixtureFile, ...]
    source_bundle: bytes
    source_overrides: tuple[FixtureFile, ...]
    summary_bytes: bytes


@dataclass(frozen=True, slots=True)
class FixtureFile:
    relative_path: str
    contents: bytes
    mode: int


_FIXTURE_TEMPLATES: dict[str, FixtureTemplate] = {}


def _reset_fixture_templates() -> None:
    _FIXTURE_TEMPLATES.clear()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"},
    ).stdout.strip()


def _fixture(
    tmp_path: Path,
    mutation: Mutation = None,
    forbidden_path: str | None = None,
    provider_source: str | None = None,
) -> VerifierFixture:
    key = _fixture_template_key(mutation, forbidden_path, provider_source)
    template = _FIXTURE_TEMPLATES.get(key)
    if template is None:
        template_root = tmp_path / ".fixture-template"
        fixture = _build_fixture(template_root, mutation, forbidden_path, provider_source)
        template = _fixture_template(fixture)
        shutil.rmtree(template_root)
        _FIXTURE_TEMPLATES[key] = template
    return _materialize_fixture(tmp_path, template)


def _build_fixture(
    tmp_path: Path,
    mutation: Mutation = None,
    forbidden_path: str | None = None,
    provider_source: str | None = None,
) -> VerifierFixture:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    fixture_root = repository / "fixtures"
    shutil.copytree(FIXTURES, fixture_root)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "filter-v5@example.test")
    _git(repository, "config", "user.name", "Filter V5")
    _git(repository, "add", "fixtures")
    _git(repository, "commit", "-qm", "base")
    base_commit = _git(repository, "rev-parse", "HEAD")
    implementation_path = repository / "src" / "filter_v5_marker.py"
    implementation_path.parent.mkdir()
    implementation_path.write_text("MARKER: int = 1\n", encoding="utf-8")
    if mutation == "invalid_python":
        implementation_path.write_text("def broken(:\n", encoding="utf-8")
    if mutation == "forbidden_diff":
        forbidden = repository / "docs" / "forbidden.md"
        forbidden.parent.mkdir()
        forbidden.write_text("forbidden\n", encoding="utf-8")
    if forbidden_path is not None:
        forbidden = repository / forbidden_path
        forbidden.parent.mkdir(parents=True, exist_ok=True)
        forbidden.write_text("forbidden\n", encoding="utf-8")
    if provider_source is not None:
        provider_path = repository / "scripts" / "provider_call.py"
        provider_path.parent.mkdir(parents=True, exist_ok=True)
        provider_path.write_text(provider_source, encoding="utf-8")
    _git(repository, "add", "--all")
    _git(repository, "commit", "-qm", "implementation")
    implementation_commit = _git(repository, "rev-parse", "HEAD")
    plan = tmp_path / "approved-plan.md"
    plan.write_text("# Approved synthetic plan\n", encoding="utf-8")
    plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
    summary = tmp_path / "validation-summary.json"
    summary.write_text(
        complete_validation_summary(
            plan_sha256,
            implementation_commit,
            _actual_command_records(
                repository, implementation_commit, fixture_root, tmp_path / "summary-scratch"
            ),
            base_commit,
            final_verifier_integration._search_hash(
                fixture_root / "FilterChallengeSearchConfig.yaml"
            ),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    evidence = EvidenceBuildRequest(
        repository_root=repository,
        plan=plan,
        expected_plan_sha256=plan_sha256,
        implementation_commit=implementation_commit,
        search_config=fixture_root / "FilterChallengeSearchConfig.yaml",
        fixture_root=fixture_root,
        validation_summary=summary,
        output_root=repository / "evidence",
    )
    build_evidence_bundle(evidence)
    if mutation == "mft_failure":
        _rewrite_mft(evidence.output_root)
    _git(repository, "add", "evidence")
    _git(repository, "commit", "-qm", "evidence")
    source = _source_repository(tmp_path, mutation == "source_dirty")
    return VerifierFixture(base_commit, evidence, source)


def _fixture_template_key(
    mutation: Mutation = None,
    forbidden_path: str | None = None,
    provider_source: str | None = None,
) -> str:
    digest = hashlib.sha256(b"fixture-template-v2\0")
    for value in (
        b"MARKER: int = 1\n",
        b"# Approved synthetic plan\n",
        (mutation or "").encode(),
        (forbidden_path or "").encode(),
        (provider_source or "").encode(),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    for path in sorted(FIXTURES.rglob("*")):
        if path.is_file():
            for value in (
                path.relative_to(FIXTURES).as_posix().encode(),
                stat.S_IMODE(path.stat().st_mode).to_bytes(4, "big"),
                path.read_bytes(),
            ):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
    return digest.hexdigest()


def _fixture_template(fixture: VerifierFixture) -> FixtureTemplate:
    return FixtureTemplate(
        base_commit=fixture.base_commit,
        implementation_commit=fixture.evidence.implementation_commit,
        plan_bytes=fixture.evidence.plan.read_bytes(),
        repository_bundle=_repository_bundle(fixture.evidence.repository_root),
        repository_overrides=_working_tree_files(fixture.evidence.repository_root),
        source_bundle=_repository_bundle(fixture.source_repository),
        source_overrides=_working_tree_files(fixture.source_repository),
        summary_bytes=fixture.evidence.validation_summary.read_bytes(),
    )


def _materialize_fixture(tmp_path: Path, template: FixtureTemplate) -> VerifierFixture:
    repository = tmp_path / "repository"
    _clone_repository(template.repository_bundle, repository)
    _write_fixture_files(repository, template.repository_overrides)
    source = tmp_path / "source"
    _clone_repository(template.source_bundle, source)
    _write_fixture_files(source, template.source_overrides)
    plan = tmp_path / "approved-plan.md"
    plan.write_bytes(template.plan_bytes)
    summary = tmp_path / "validation-summary.json"
    summary.write_bytes(template.summary_bytes)
    fixture_root = repository / "fixtures"
    evidence = EvidenceBuildRequest(
        repository_root=repository,
        plan=plan,
        expected_plan_sha256=hashlib.sha256(template.plan_bytes).hexdigest(),
        implementation_commit=template.implementation_commit,
        search_config=fixture_root / "FilterChallengeSearchConfig.yaml",
        fixture_root=fixture_root,
        validation_summary=summary,
        output_root=repository / "evidence",
    )
    return VerifierFixture(template.base_commit, evidence, source)


def _repository_bundle(repository: Path) -> bytes:
    bundle_path = repository.parent / f".{repository.name}.bundle"
    subprocess.run(
        ("git", "bundle", "create", str(bundle_path), "--all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    bundle = bundle_path.read_bytes()
    bundle_path.unlink()
    return bundle


def _clone_repository(bundle: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle_path = destination.parent / f".{destination.name}.bundle"
    bundle_path.write_bytes(bundle)
    subprocess.run(
        ("git", "clone", "--no-local", str(bundle_path), str(destination)),
        check=True,
        capture_output=True,
        text=True,
    )
    bundle_path.unlink()
    _git(destination, "config", "core.quotePath", "false")


def _working_tree_files(repository: Path) -> tuple[FixtureFile, ...]:
    files: list[FixtureFile] = []
    names = set(_git(repository, "diff", "--name-only", "-z").split("\0"))
    names.update(_git(repository, "ls-files", "--others", "--exclude-standard", "-z").split("\0"))
    for name in sorted(name for name in names if name):
        relative_path = Path(name)
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        path = repository / relative_path
        if path.is_file():
            files.append(
                FixtureFile(
                    relative_path=relative_path.as_posix(),
                    contents=path.read_bytes(),
                    mode=stat.S_IMODE(path.stat().st_mode),
                )
            )
    return tuple(files)


def _write_fixture_files(root: Path, files: tuple[FixtureFile, ...]) -> None:
    for file in files:
        relative_path = Path(file.relative_path)
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.contents)
        target.chmod(file.mode)


def _actual_command_records(
    repository: Path, implementation_commit: str, fixture_root: Path, scratch_root: Path
) -> tuple[Task17CommandRecord, ...]:
    digest = hashlib.sha256()
    for path in sorted(fixture_root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(fixture_root).as_posix().encode())
            digest.update(path.read_bytes())
    key = (implementation_commit, digest.hexdigest())
    cached = _COMMAND_RECORDS.get(key)
    if cached is not None:
        return tuple(Task17CommandRecord.model_validate_json(record) for record in cached)
    scratch_root.mkdir()
    guard_root = install_execution_guards(scratch_root)
    _, records, _ = final_verifier_integration._run_commands(
        repository,
        implementation_commit,
        fixture_root / "FilterChallengeSearchConfig.yaml",
        fixture_root,
        fixture_root / "bct_execution_prerequisites.json",
        scratch_root,
        guard_root,
    )
    shutil.rmtree(scratch_root)
    _COMMAND_RECORDS[key] = tuple(record.model_dump_json().encode() for record in records)
    return records


def _source_repository(tmp_path: Path, dirty: bool) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "filter-v5@example.test")
    _git(source, "config", "user.name", "Filter V5")
    (source / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-qm", "source")
    name = "unexpected.txt" if dirty else "Pilot-A 관련 기록.md"
    (source / name).write_text("untracked\n", encoding="utf-8")
    return source


def _rewrite_mft(evidence_root: Path) -> None:
    report_path = evidence_root / "mft_fv5_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["report"]["all_passed"] = False
    report_path.write_bytes(canonical_json_bytes(report))
    manifest_path = evidence_root / "implementation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reports"][report_path.name] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def _request(
    fixture: VerifierFixture,
    mode: Literal["plan-compliance", "code-quality", "integration", "scope"],
    output: Path,
) -> FinalVerifierRequest:
    request = FinalVerifierRequest(
        mode=mode,
        repository_root=fixture.evidence.repository_root,
        plan=fixture.evidence.plan,
        expected_plan_sha256=fixture.evidence.expected_plan_sha256,
        evidence_root=fixture.evidence.output_root,
        validation_summary=fixture.evidence.validation_summary,
        output=output,
        approval_paths=(),
    )
    match mode:
        case "code-quality":
            return replace(request, base_commit=fixture.base_commit)
        case "integration":
            return replace(
                request,
                search_config=fixture.evidence.search_config,
                fixture_root=fixture.evidence.fixture_root,
                execution_prerequisites=(
                    fixture.evidence.fixture_root / "bct_execution_prerequisites.json"
                ),
                scratch_root=output.parent / "scratch",
            )
        case "scope":
            return replace(
                request,
                base_commit=fixture.base_commit,
                source_repository_root=fixture.source_repository,
            )
        case "plan-compliance":
            return request


def test_plan_compliance_rejects_failed_evidence_clause(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, "mft_failure")

    with pytest.raises(FinalVerifierError, match="LEDGER_CLAUSE_08_REJECTED"):
        verify_final_report(_request(fixture, "plan-compliance", tmp_path / "f1.json"))


def test_code_quality_runs_nonempty_commands_and_rejects_invalid_python(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "passing")
    report = verify_final_report(_request(fixture, "code-quality", tmp_path / "passing" / "f2.json"))
    commands = report["commands"]
    assert isinstance(commands, list) and commands
    assert all(isinstance(command, dict) and command["exit_code"] == 0 for command in commands)

    invalid = _fixture(tmp_path / "invalid", "invalid_python")
    with pytest.raises(FinalVerifierError, match="CODE_QUALITY_REJECTED"):
        verify_final_report(_request(invalid, "code-quality", tmp_path / "invalid" / "f2.json"))


def test_integration_reruns_commands_and_rejects_evidence_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "passing")
    report = verify_final_report(_request(fixture, "integration", tmp_path / "passing" / "f3.json"))
    assert report["command_ids"] == [
        "validate-search-config",
        "validate-selected-policy",
        "mft",
        "build-archive",
        "validate-archive",
        "cost-preview",
        "bct-readiness",
    ]
    mutations = report["mutations"]
    assert isinstance(mutations, list) and mutations
    assert all(isinstance(item, dict) and item["observed"] == item["expected"] for item in mutations)

    mismatch = _fixture(tmp_path / "mismatch", "mft_failure")
    with pytest.raises(FinalVerifierError, match="INTEGRATION_EVIDENCE_MISMATCH"):
        verify_final_report(_request(mismatch, "integration", tmp_path / "mismatch" / "f3.json"))


def test_scope_reads_real_diff_authorities_and_source_status(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "passing")
    report = verify_final_report(_request(fixture, "scope", tmp_path / "passing" / "f4.json"))
    assert report["forbidden_diff_count"] == 0
    assert report["authority_status"] == "matched"
    assert report["source_dirty_allowlist"] == ["?? Pilot-A 관련 기록.md"]

    forbidden = _fixture(tmp_path / "forbidden", "forbidden_diff")
    with pytest.raises(FinalVerifierError, match="SCOPE_FORBIDDEN_DIFF"):
        verify_final_report(_request(forbidden, "scope", tmp_path / "forbidden" / "f4.json"))
    dirty_source = _fixture(tmp_path / "source", "source_dirty")
    with pytest.raises(FinalVerifierError, match="SOURCE_DIRTY_ALLOWLIST_MISMATCH"):
        verify_final_report(_request(dirty_source, "scope", tmp_path / "source" / "f4.json"))


def test_descriptor_rejects_traversal_component(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
        EvidenceBuildError,
        descriptor_sha256,
    )

    with pytest.raises(EvidenceBuildError, match="DESCRIPTOR_PATH_COMPONENT_INVALID"):
        descriptor_sha256(Path("/tmp/../must-not-open"))


def test_modes_reject_irrelevant_or_missing_mode_inputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan_request = _request(fixture, "plan-compliance", tmp_path / "f1.json")
    with pytest.raises(FinalVerifierError, match="IRRELEVANT_MODE_ARGUMENTS"):
        verify_final_report(replace(plan_request, base_commit=fixture.base_commit))


JsonObject = dict[str, JsonValue]


def _json_object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _json_array(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _json_field(value: JsonObject, name: str) -> JsonValue:
    assert name in value
    return value[name]


def _string_field(value: JsonObject, name: str) -> str:
    field = _json_field(value, name)
    assert isinstance(field, str)
    return field


def _write_json(path: Path, value: JsonObject) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _report(root: Path, name: str) -> JsonObject:
    return _json_object(json_value_from_bytes((root / name).read_bytes(), "TEST_REPORT_INVALID"))


def _mutate_ledger_clause(root: Path, summary: Path, clause: int) -> None:
    policy = _report(root, "policy_schema_hashes.json")
    mft = _report(root, "mft_fv5_report.json")
    archive = _report(root, "archive_validation_report.json")
    readiness = _report(root, "bct_readiness_report.json")
    validation = _report(root, "test_lint_typecheck_report.json")
    policy_header = _json_object(_json_field(policy, "header"))
    policy_config_schema_hashes = _json_object(_json_field(policy_header, "config_schema_hashes"))
    mft_report = _json_object(_json_field(mft, "report"))
    mft_state = _json_object(_json_field(mft_report, "state_report"))
    mft_results = _json_array(_json_field(mft_state, "results"))
    mft_safety = _json_object(_json_field(mft_report, "safety_report"))
    mft_cases = _json_array(_json_field(mft_safety, "cases"))
    archive_report = _json_object(_json_field(archive, "report"))
    readiness_report = _json_object(_json_field(readiness, "report"))
    match clause:
        case 1:
            policy["domain_model_schema_hashes"] = {}
            _write_json(root / "policy_schema_hashes.json", policy)
        case 2:
            result = _json_object(mft_results[0])
            actual = _json_object(_json_field(result, "actual"))
            actual["source_state_after_hash"] = "drift"
            _write_json(root / "mft_fv5_report.json", mft)
        case 3:
            assertions = _json_array(_json_field(_json_object(mft_cases[3]), "assertions"))
            _json_object(assertions[0])["actual"] = ["unknown"]
            _write_json(root / "mft_fv5_report.json", mft)
        case 4:
            assertions = _json_array(_json_field(_json_object(mft_cases[4]), "assertions"))
            _json_object(assertions[0])["actual"] = ["missing"]
            _write_json(root / "mft_fv5_report.json", mft)
        case 5:
            result = _json_object(mft_results[2])
            actual = _json_object(_json_field(result, "actual"))
            actual["route_targets"] = ["active"]
            _write_json(root / "mft_fv5_report.json", mft)
        case 6:
            policy_config_schema_hashes["search_config"] = "drift"
            _write_json(root / "policy_schema_hashes.json", policy)
        case 7:
            archive_report["run_id"] = "drift"
            _write_json(root / "archive_validation_report.json", archive)
        case 8:
            execution_counts = _json_array(_json_field(mft_report, "execution_counts"))
            _json_object(execution_counts[0])["count"] = 0
            _write_json(root / "mft_fv5_report.json", mft)
        case 9:
            family_statuses = _json_array(_json_field(readiness_report, "family_statuses"))
            _json_object(family_statuses[0])["test_id"] = "BCT-FV5-INVALID"
            _write_json(root / "bct_readiness_report.json", readiness)
        case 10:
            validation["validation_status"] = "fail"
            _write_json(root / "test_lint_typecheck_report.json", validation)
        case 11:
            (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
        case 12:
            for name in (
                "policy_schema_hashes.json", "mft_fv5_report.json",
                "information_boundary_report.json", "route_invariance_report.json",
                "answer_call_provenance_report.json", "archive_validation_report.json",
                "test_lint_typecheck_report.json", "bct_readiness_report.json",
            ):
                report = _report(root, name)
                header = _json_object(_json_field(report, "header"))
                _json_object(_json_field(header, "policy"))["canonical_patch_status"] = "applied"
                _write_json(root / name, report)
            manifest = _report(root, "implementation_manifest.json")
            manifest_header = _json_object(_json_field(manifest, "header"))
            _json_object(_json_field(manifest_header, "policy"))["canonical_patch_status"] = "applied"
            manifest_reports = _json_object(_json_field(manifest, "reports"))
            manifest["reports"] = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in manifest_reports
            }
            _write_json(root / "implementation_manifest.json", manifest)
        case unreachable:
            raise AssertionError(unreachable)
    if clause == 10:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        payload["provider_calls_issued"] = 0
        summary.write_bytes(canonical_json_bytes(payload))


def test_plan_compliance_emits_stable_semantic_clause_descriptions(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary = json.loads(fixture.evidence.validation_summary.read_text(encoding="utf-8"))

    report = verify_plan_compliance(fixture.evidence.output_root, summary)

    assert report["checklist"] == [
        {"clause_id": "ledger-01-versioned-domain", "description": "versioned domain schema boundary", "status": "pass"},
        {"clause_id": "ledger-02-read-only-pair", "description": "isolated matched read-only pair", "status": "pass"},
        {"clause_id": "ledger-03-native-adapters", "description": "native adapter exposure semantics", "status": "pass"},
        {"clause_id": "ledger-04-answer-provenance", "description": "answer-call provenance relations", "status": "pass"},
        {"clause_id": "ledger-05-routing", "description": "eligibility witness and routing", "status": "pass"},
        {"clause_id": "ledger-06-configuration", "description": "strict configuration registry", "status": "pass"},
        {"clause_id": "ledger-07-archive", "description": "archive logging reconciliation", "status": "pass"},
        {"clause_id": "ledger-08-mft", "description": "exact MFT execution", "status": "pass"},
        {"clause_id": "ledger-09-bct", "description": "behavioral readiness interfaces", "status": "pass"},
        {"clause_id": "ledger-10-validation", "description": "validation gate evidence", "status": "pass"},
        {"clause_id": "ledger-11-evidence", "description": "tracked evidence graph", "status": "pass"},
        {"clause_id": "ledger-12-terminal", "description": "terminal metadata availability", "status": "pass"},
    ]


@pytest.mark.parametrize("clause", range(1, 13))
def test_plan_compliance_rejects_each_mutated_ledger_clause(tmp_path: Path, clause: int) -> None:
    fixture = _fixture(tmp_path)
    _mutate_ledger_clause(
        fixture.evidence.output_root, fixture.evidence.validation_summary, clause
    )
    summary = json.loads(fixture.evidence.validation_summary.read_text(encoding="utf-8"))

    with pytest.raises(FinalVerifierError, match=f"LEDGER_CLAUSE_{clause:02d}_REJECTED"):
        verify_plan_compliance(fixture.evidence.output_root, summary)


def test_plan_compliance_rejects_public_model_order_and_validation_report_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary = json.loads(fixture.evidence.validation_summary.read_text(encoding="utf-8"))
    policy = _report(fixture.evidence.output_root, "policy_schema_hashes.json")
    policy["public_domain_model_names"] = list(
        reversed(_json_array(_json_field(policy, "public_domain_model_names")))
    )
    _write_json(fixture.evidence.output_root / "policy_schema_hashes.json", policy)
    with pytest.raises(FinalVerifierError, match="LEDGER_CLAUSE_01_REJECTED"):
        verify_plan_compliance(fixture.evidence.output_root, summary)

    fixture = _fixture(tmp_path / "validation")
    summary = json.loads(fixture.evidence.validation_summary.read_text(encoding="utf-8"))
    validation = _report(fixture.evidence.output_root, "test_lint_typecheck_report.json")
    validation["command_records"] = []
    _write_json(fixture.evidence.output_root / "test_lint_typecheck_report.json", validation)
    with pytest.raises(FinalVerifierError, match="LEDGER_CLAUSE_10_REJECTED"):
        verify_plan_compliance(fixture.evidence.output_root, summary)


@pytest.mark.parametrize(
    "source",
    (
        "from memcontam.clients.factory import build_llm_client as make_client\nmake_client(None, None)\n",
        "import memcontam.clients.factory as clients\nclients.build_llm_client(None, None)\n",
        "from memcontam.clients.openai_compatible import OpenAICompatibleClient as Client\nClient(None, True)\n",
        "import memcontam.clients.factory as clients\nmake_client = clients.build_llm_client\nmake_client(None, None)\n",
    ),
)
def test_code_quality_rejects_provider_constructor_aliases(tmp_path: Path, source: str) -> None:
    path = tmp_path / "src" / "memcontam" / "experiment" / "phase12" / "filter_challenge" / "bad.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")

    assert _structural_findings(path) == [f"provider:{path.as_posix()}"]


def test_code_quality_rejects_evidence_serialization_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report_path = fixture.evidence.output_root / "policy_schema_hashes.json"
    report_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FinalVerifierError, match="CODE_QUALITY_REJECTED"):
        verify_code_quality(
            fixture.evidence.repository_root,
            fixture.base_commit,
            fixture.evidence.implementation_commit,
            fixture.evidence.output_root,
            fixture.evidence.validation_summary,
        )


def test_integration_reconciles_all_outputs_and_copied_mutations(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = verify_final_report(_request(fixture, "integration", tmp_path / "f3.json"))
    reconciled_outputs = _json_object(_json_field(report, "reconciled_outputs"))
    selected_policy = _json_object(_json_field(reconciled_outputs, "validate-selected-policy"))
    mutations = _json_array(_json_field(report, "mutations"))
    commands = _json_array(_json_field(report, "commands"))

    assert set(reconciled_outputs) == {
        "validate-search-config", "validate-selected-policy", "mft", "build-archive",
        "validate-archive", "cost-preview", "bct-readiness",
    }
    assert selected_policy["execution_authorized"] is False
    assert all(_json_object(command)["cwd"] == "<repository>" for command in commands)
    assert all(
        str(fixture.evidence.repository_root)
        not in _json_array(_json_field(_json_object(command), "normalized_argv"))
        for command in commands
    )
    assert {_string_field(_json_object(item), "mutation_id") for item in mutations} >= {
        "archive_bytes", "bct_authorization", "provenance_evidence",
    }
    assert report["execution_guards"] == {
        "bct_behavior": "not_reached", "provider_constructor": "not_reached"
    }


@pytest.mark.parametrize(
    "forbidden_path",
    (
        "src/memcontam/memory/admission.py",
        "src/memcontam/memory/filtered_state.py",
        "src/memcontam/experiment/phase12/filter_mft.py",
        "tests/test_phase12_filter_v4.py",
        "data/phase12/filter_v4/evidence.json",
        "docs/scientific-golden.json",
    ),
)
def test_scope_rejects_exact_protected_paths(tmp_path: Path, forbidden_path: str) -> None:
    fixture = _fixture(tmp_path, forbidden_path=forbidden_path)

    with pytest.raises(FinalVerifierError, match="SCOPE_FORBIDDEN_DIFF"):
        verify_final_report(_request(fixture, "scope", tmp_path / "f4.json"))


def _terminal_request(
    fixture: VerifierFixture, output: Path, approvals: tuple[Path, ...]
) -> FinalVerifierRequest:
    return FinalVerifierRequest(
        mode="terminal",
        repository_root=fixture.evidence.repository_root,
        plan=fixture.evidence.plan,
        expected_plan_sha256=fixture.evidence.expected_plan_sha256,
        evidence_root=fixture.evidence.output_root,
        validation_summary=fixture.evidence.validation_summary,
        output=output,
        approval_paths=approvals,
    )


def test_terminal_requires_complete_approval_payloads_and_derives_ledger_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    f1 = tmp_path / "f1.json"
    f2 = tmp_path / "f2.json"
    f3 = tmp_path / "f3.json"
    f4 = tmp_path / "f4.json"
    verify_final_report(_request(fixture, "plan-compliance", f1))
    verify_final_report(_request(fixture, "code-quality", f2))
    verify_final_report(_request(fixture, "integration", f3))
    verify_final_report(_request(fixture, "scope", f4))

    report = verify_final_report(_terminal_request(fixture, tmp_path / "terminal.json", (f1, f2, f3, f4)))

    assert set(report) >= {
        "initial_head", "implementation_commit", "evidence_commit", "final_head",
        "ordered_commit_series", "worktree_status", "implemented_components", "mft_result",
        "provenance_result", "route_result", "archive_result", "bct_result",
        "canonical_patch_status", "remaining_scientific_choices", "provider_calls_issued",
        "evidence_paths_and_hashes",
    }

    minimal = tmp_path / "minimal.json"
    minimal.write_text(json.dumps({"mode": "plan-compliance", "verdict": "APPROVE", "bindings": report["bindings"]}), encoding="utf-8")
    with pytest.raises(FinalVerifierError, match="FINAL_APPROVAL_MISMATCH"):
        verify_final_report(_terminal_request(fixture, tmp_path / "minimal-terminal.json", (minimal, f2, f3, f4)))

    tampered = json.loads(f3.read_text(encoding="utf-8"))
    tampered["mutations"] = []
    f3.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(FinalVerifierError, match="FINAL_APPROVAL_MISMATCH"):
        verify_final_report(_terminal_request(fixture, tmp_path / "tampered-terminal.json", (f1, f2, f3, f4)))


def test_nonterminal_rejects_approval_paths_and_cli_emits_approval_marker(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text("{}\n", encoding="utf-8")
    request = _request(fixture, "plan-compliance", tmp_path / "f1.json")
    with pytest.raises(FinalVerifierError, match="IRRELEVANT_MODE_ARGUMENTS"):
        verify_final_report(replace(request, approval_paths=(approval,)))

    command = (
        sys.executable,
        str(ROOT / "scripts" / "verify_phase12_filter_v5_build.py"),
        "plan-compliance",
        "--repository-root", str(fixture.evidence.repository_root),
        "--plan", str(fixture.evidence.plan),
        "--expected-plan-sha256", fixture.evidence.expected_plan_sha256,
        "--evidence-root", str(fixture.evidence.output_root),
        "--validation-summary", str(fixture.evidence.validation_summary),
        "--output", str(tmp_path / "cli-f1.json"),
    )
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "APPROVE"

    rejected = subprocess.run(
        (*command[:-2], "--f1", str(approval), *command[-2:]),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert rejected.stdout.strip() == "IRRELEVANT_MODE_ARGUMENTS"


def _summary_command_records() -> list[dict[str, JsonValue]]:
    return [
        {
            "command_id": command_id,
            "cwd": "<repository>",
            "exit_code": 0,
            "normalized_argv": ["phase12", "filter-v5", command_id],
            "stderr_sha256": "a" * 64,
            "stdout_sha256": "b" * 64,
        }
        for command_id in final_verifier_integration.COMMAND_IDS
    ]


@pytest.mark.parametrize(
    ("mutation"),
    (
        lambda records: None,
        lambda records: list(reversed(records)),
        lambda records: [{**records[0], "normalized_argv": ["wrong"]}, *records[1:]],
        lambda records: [{**records[0], "cwd": "<scratch>"}, *records[1:]],
        lambda records: [{**records[0], "exit_code": 1}, *records[1:]],
        lambda records: [{**records[0], "stdout_sha256": "c" * 64}, *records[1:]],
        lambda records: [{**records[0], "stderr_sha256": "d" * 64}, *records[1:]],
    ),
)
def test_integration_rejects_every_summary_record_mismatch(
    tmp_path: Path, mutation: object
) -> None:
    summary = tmp_path / "summary.json"
    records = _summary_command_records()
    assert callable(mutation)
    mutated = mutation(records)
    summary_payload = complete_validation_summary("0" * 64, "a" * 40).model_dump(mode="json")
    if mutated is None:
        del summary_payload["command_records"]
    else:
        summary_payload["command_records"] = mutated
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    actual_records = tuple(Task17CommandRecord.model_validate(record) for record in records)

    with pytest.raises(FinalVerifierError, match="INTEGRATION_SUMMARY_RECORDS_MISMATCH"):
        reconcile_summary_records(summary, actual_records)


def test_integration_rejects_broken_guard_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)

    def install_broken_guards(scratch_root: Path) -> Path:
        guard_root = scratch_root / "broken-guards"
        guard_root.mkdir()
        return guard_root

    monkeypatch.setattr(final_verifier_integration, "install_execution_guards", install_broken_guards)

    with pytest.raises(FinalVerifierError, match="FINAL_VERIFIER_EXECUTION_GUARD_REACHED"):
        verify_final_report(_request(fixture, "integration", tmp_path / "f3.json"))


def test_fixture_template_materializes_independent_equivalent_repositories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_fixture_templates()
    calls = 0
    original = build_evidence_bundle

    def counted(request: EvidenceBuildRequest) -> EvidenceBundle:
        nonlocal calls
        calls += 1
        return original(request)

    monkeypatch.setattr("tests.test_phase12_filter_v5_final_verifier_modes.build_evidence_bundle", counted)
    first = _fixture(tmp_path / "first")
    second = _fixture(tmp_path / "second")

    assert calls == 1
    assert first.evidence.repository_root != second.evidence.repository_root
    assert first.evidence.implementation_commit == second.evidence.implementation_commit
    assert first.evidence.validation_summary.read_bytes() == second.evidence.validation_summary.read_bytes()
    assert (second.source_repository / "Pilot-A 관련 기록.md").read_text(encoding="utf-8") == "untracked\n"


def test_fixture_template_cache_uses_immutable_bytes_and_isolates_materialization_mutations(
    tmp_path: Path,
) -> None:
    _reset_fixture_templates()
    first = _fixture(tmp_path / "first")
    key = _fixture_template_key()
    template = _FIXTURE_TEMPLATES[key]
    cached_template = template
    evidence_path = first.evidence.output_root / "mft_fv5_report.json"
    evidence_bytes = evidence_path.read_bytes()
    source_path = first.source_repository / "Pilot-A 관련 기록.md"

    assert all(not isinstance(getattr(template, field.name), Path) for field in fields(template))
    assert all(
        isinstance(record, bytes) for records in _COMMAND_RECORDS.values() for record in records
    )

    evidence_path.write_text("tampered\n", encoding="utf-8")
    source_path.write_text("tampered\n", encoding="utf-8")
    first.evidence.plan.write_text("tampered\n", encoding="utf-8")
    first.evidence.validation_summary.write_text("tampered\n", encoding="utf-8")

    second = _fixture(tmp_path / "second")

    assert _FIXTURE_TEMPLATES[key] == cached_template
    assert (second.evidence.output_root / "mft_fv5_report.json").read_bytes() == evidence_bytes
    assert (second.source_repository / "Pilot-A 관련 기록.md").read_text(encoding="utf-8") == "untracked\n"
    assert second.evidence.plan.read_text(encoding="utf-8") == "# Approved synthetic plan\n"
    assert second.evidence.validation_summary.read_text(encoding="utf-8") != "tampered\n"
    assert _fixture_template_key("mft_failure") != key


def test_fixture_template_cache_separates_mutation_inputs(tmp_path: Path) -> None:
    _reset_fixture_templates()
    clean = _fixture(tmp_path / "clean")
    mutated = _fixture(tmp_path / "mutated", "mft_failure")

    assert set(_FIXTURE_TEMPLATES) == {
        _fixture_template_key(),
        _fixture_template_key("mft_failure"),
    }
    assert clean.evidence.implementation_commit == mutated.evidence.implementation_commit
    assert (mutated.evidence.output_root / "mft_fv5_report.json").read_bytes() != (
        clean.evidence.output_root / "mft_fv5_report.json"
    ).read_bytes()
