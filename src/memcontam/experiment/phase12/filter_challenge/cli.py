from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Literal, TypeAlias, assert_never

from pydantic import BaseModel, ValidationError

from memcontam.experiment.phase12.filter_challenge.bct import (
    BCTAuthorizationError,
    ExecutionPreflightRequest,
    ExecutionPrerequisites,
    SoftwareInterfaceChecks,
    build_cost_preview,
    build_readiness,
    evaluate_execution_preflight,
    evaluate_software_interface_readiness,
)
from memcontam.experiment.phase12.filter_challenge.build_archive import (
    ArchiveValidationRequest,
    BuildArchiveError,
    BuildArchiveReport,
    BuildArchiveRequest,
    build_archive,
    validate_archive,
)
from memcontam.experiment.phase12.filter_challenge.mft import (
    MFT_IDS,
    MergedMftReport,
    MftReportError,
    build_mft_report,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.registry import validate_stage
from memcontam.experiment.phase12.filter_challenge.registry_common import (
    RegistryValidationError,
    StrictRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import (
    SearchConfig,
    SelectedPolicy,
)


Stage: TypeAlias = Literal["build", "pilot_b", "main"]
Command: TypeAlias = Literal[
    "validate-search-config", "validate-selected-policy", "mft", "build-archive",
    "validate-archive", "cost-preview", "bct-readiness",
]


class SearchValidationReport(StrictRegistry):
    schema_version: Literal["filter_challenge_search_config_validation_v1"] = (
        "filter_challenge_search_config_validation_v1"
    )
    valid: Literal[True] = True
    search_config_id: str
    search_config_hash: str
    evidence_layer: Literal["build"] = "build"
    scientific_result: Literal[False] = False
    fixture_only: Literal[True] = True
    provider_calls_issued: Literal[0] = 0


class SelectedPolicyValidationReport(StrictRegistry):
    schema_version: Literal["filter_challenge_selected_policy_validation_v1"] = (
        "filter_challenge_selected_policy_validation_v1"
    )
    valid: Literal[True] = True
    stage: Stage
    stage_reason_code: Literal["SELECTED_POLICY_REQUIRED"] | None = None
    search_config_id: str
    search_config_hash: str
    selected_policy_id: str
    selected_policy_hash: str
    selected_policy_required: bool
    selected_policy_reference_valid: Literal[True] = True
    validation_scope: Literal["schema_reference_only"] = "schema_reference_only"
    execution_authorized: Literal[False] = False
    provider_calls_issued: Literal[0] = 0


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    filter_v5 = commands.add_parser("filter-v5")
    subcommands = filter_v5.add_subparsers(dest="filter_v5_command", required=True)

    validate_search = subcommands.add_parser("validate-search-config")
    validate_search.add_argument("--config", type=Path, required=True)
    validate_search.add_argument("--output", type=Path, required=True)

    validate_policy = subcommands.add_parser("validate-selected-policy")
    validate_policy.add_argument("--search-config", type=Path, required=True)
    validate_policy.add_argument("--selected-policy", type=Path, required=True)
    validate_policy.add_argument("--stage", choices=("build", "pilot_b", "main"), required=True)
    validate_policy.add_argument("--output", type=Path, required=True)

    mft = subcommands.add_parser("mft")
    _add_search_and_fixture(mft)
    mft.add_argument("--output", type=Path, required=True)

    archive = subcommands.add_parser("build-archive")
    _add_search_and_fixture(archive)
    archive.add_argument("--implementation-commit", required=True)
    archive.add_argument("--freeze-id", required=True)
    archive.add_argument("--run-id", required=True)
    archive.add_argument("--output-root", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)

    validate_archive_parser = subcommands.add_parser("validate-archive")
    validate_archive_parser.add_argument("--archive", type=Path, required=True)
    validate_archive_parser.add_argument("--expected-implementation-commit", required=True)
    validate_archive_parser.add_argument("--expected-search-config-hash", required=True)
    validate_archive_parser.add_argument("--output", type=Path, required=True)

    cost = subcommands.add_parser("cost-preview")
    cost.add_argument("--search-config", type=Path, required=True)
    cost.add_argument("--output", type=Path, required=True)

    readiness = subcommands.add_parser("bct-readiness")
    readiness.add_argument("--search-config", type=Path, required=True)
    readiness.add_argument("--mft-report", type=Path, required=True)
    readiness.add_argument("--archive-report", type=Path, required=True)
    readiness.add_argument("--execution-prerequisites", type=Path, required=True)
    readiness.add_argument("--output", type=Path, required=True)


def run(args: argparse.Namespace) -> None:
    try:
        report = _dispatch(args.filter_v5_command, args)
    except (
        BuildArchiveError,
        BCTAuthorizationError,
        MftReportError,
        OSError,
        RegistryValidationError,
        ValidationError,
    ) as error:
        raise SystemExit(str(error)) from error
    _write_report(args.output, report)


def _dispatch(command: Command, args: argparse.Namespace) -> BaseModel:
    match command:
        case "validate-search-config":
            search = _load_search(args.config)
            return SearchValidationReport(
                search_config_id=search.registry_id,
                search_config_hash=search.search_config_hash,
            )
        case "validate-selected-policy":
            search = _load_search(args.search_config)
            policy = _load_policy(args.selected_policy)
            gate = validate_stage(search, policy, stage=args.stage)
            return SelectedPolicyValidationReport(
                stage=args.stage,
                stage_reason_code=gate.reason_code,
                search_config_id=search.registry_id,
                search_config_hash=search.search_config_hash,
                selected_policy_id=policy.registry_id,
                selected_policy_hash=policy.selected_policy_hash,
                selected_policy_required=args.stage == "main",
            )
        case "mft":
            search, inventory, suite = _load_build_inputs(
                args.search_config, args.fixture_root
            )
            return build_mft_report(search, inventory, suite)
        case "build-archive":
            search, inventory, suite = _load_build_inputs(
                args.search_config, args.fixture_root
            )
            return build_archive(
                BuildArchiveRequest(
                    search_config=search,
                    inventory=inventory,
                    suite=suite,
                    implementation_commit=args.implementation_commit,
                    freeze_id=args.freeze_id,
                    run_id=args.run_id,
                    output_root=args.output_root,
                )
            )
        case "validate-archive":
            return validate_archive(
                ArchiveValidationRequest(
                    archive=args.archive,
                    expected_implementation_commit=args.expected_implementation_commit,
                    expected_search_config_hash=args.expected_search_config_hash,
                )
            )
        case "cost-preview":
            return build_cost_preview(_load_search(args.search_config), (), None)
        case "bct-readiness":
            return _build_readiness(args)
        case unreachable:
            assert_never(unreachable)


def _build_readiness(args: argparse.Namespace) -> BaseModel:
    search = _load_search(args.search_config)
    mft = MergedMftReport.model_validate_json(args.mft_report.read_text(encoding="utf-8"))
    archive = BuildArchiveReport.model_validate_json(
        args.archive_report.read_text(encoding="utf-8")
    )
    prerequisites = ExecutionPrerequisites.model_validate_json(
        args.execution_prerequisites.read_text(encoding="utf-8")
    )
    if (
        prerequisites.search_config_frozen
        and prerequisites.inventory_frozen
        and prerequisites.canonical_patch_status == "applied"
        and prerequisites.provider_config_enabled
        and prerequisites.runtime_authorization_present
    ):
        raise BCTAuthorizationError("BCT_EXECUTION_AUTHORIZATION_FORBIDDEN")
    provenance = next(
        case
        for case in mft.safety_report.cases
        if case.test_id == "MFT-FV5-13-ANSWER-CALL-PROVENANCE"
    )
    software = evaluate_software_interface_readiness(
        SoftwareInterfaceChecks(
            domain_schema_valid=True,
            search_config_valid=mft.search_config_hash == search.search_config_hash,
            mft_gate_passed=mft.ordered_test_ids == MFT_IDS and mft.all_passed,
            archive_validation_passed=(
                archive.archive_valid and archive.search_config_hash == search.search_config_hash
            ),
            answer_call_provenance_engineering_ready=provenance.status == "pass",
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
    return build_readiness(software, execution, build_cost_preview(search, (), None))


def _add_search_and_fixture(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)


def _load_search(path: Path) -> SearchConfig:
    return SearchConfig.model_validate(_load_yaml(path))


def _load_policy(path: Path) -> SelectedPolicy:
    return SelectedPolicy.model_validate(_load_yaml(path))


def _load_yaml(path: Path) -> JsonValue:
    safe_load = getattr(importlib.import_module("yaml"), "safe_load")
    return safe_load(path.read_text(encoding="utf-8"))


def _load_build_inputs(
    search_path: Path, fixture_root: Path
) -> tuple[SearchConfig, ProbeInventoryRegistry, OperationalSuiteRegistry]:
    return (
        _load_search(search_path),
        ProbeInventoryRegistry.model_validate_json(
            (fixture_root / "probe_inventory_manifest.json").read_text(encoding="utf-8")
        ),
        OperationalSuiteRegistry.model_validate_json(
            (fixture_root / "operational_suite_manifest.json").read_text(encoding="utf-8")
        ),
    )


def _write_report(path: Path, report: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    path.write_text(text, encoding="utf-8")
    print(text, end="")


__all__ = ("add_parser", "run")
