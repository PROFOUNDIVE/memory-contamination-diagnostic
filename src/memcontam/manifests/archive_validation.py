from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from memcontam.config.phase12 import Phase12ConfigError, load_phase12_config
from memcontam.experiment.phase12.contracts import (
    CodeMatrixPlan,
    ExploratoryActivationManifest,
    MftManifest,
    PilotBManifest,
    RouteFeasibilityReport,
    RouteSelectionManifest,
    SeedAllocationManifest,
    SelectedPackageResourceManifest,
)
from memcontam.experiment.phase12.planner import (
    PlanningError,
    validate_exploratory_activation,
    validate_route_selection,
)
from memcontam.manifests.aggregate_manifest import (
    AggregateManifestError,
    read_aggregate_manifest,
    validate_aggregate_manifest,
)
from memcontam.manifests.claim_scope import ClaimScopeError, read_claim_scope, validate_claim_scope
from memcontam.manifests.run_manifest import (
    ExploratoryRunManifestRow,
    ManifestError,
    RunManifest,
    SelectedRouteRunManifestRow,
    read_run_manifest,
    validate_run_manifest,
)


_MANIFEST_PATHS = {
    "run": "run_manifest.jsonl",
    "aggregate": "aggregate_manifest.jsonl",
    "claim": "claim_scope_table.md",
}
_GOVERNANCE_PATHS = {
    "routes": "route_selections.json",
    "allocations": "seed_allocations.json",
    "activations": "exploratory_activations.json",
    "reports": "feasibility_reports.json",
    "pilot": "pilot_b.json",
    "mft": "mft.json",
    "plans": "code_matrix_plans.json",
    "resources": "resource_manifests.json",
}
_Model = TypeVar("_Model", bound=BaseModel)


class ArchiveError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ArchiveValidationReport:
    archive_valid: bool
    resolved_edges: int
    errors: tuple[ArchiveError, ...] = ()

    @property
    def reason_code(self) -> str | None:
        return None if not self.errors else self.errors[0].code

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_valid": self.archive_valid,
            "errors": [error.code for error in self.errors],
            "resolved_edges": self.resolved_edges,
        }


def validate_archive(root: Path) -> ArchiveValidationReport:
    """Reconstruct a Phase 12 archive without admitting or running anything."""
    try:
        resolved_edges = _validate_archive(Path(root).resolve())
    except ArchiveError as error:
        return ArchiveValidationReport(False, 0, (error,))
    return ArchiveValidationReport(True, resolved_edges)


def _validate_archive(root: Path) -> int:
    paths = {name: root / relative for name, relative in _MANIFEST_PATHS.items()}
    if any(not path.is_file() for path in paths.values()):
        raise ArchiveError("MANIFEST_DESTINATION_MISSING")

    try:
        runs = read_run_manifest(paths["run"])
    except ManifestError as error:
        raise ArchiveError(error.code) from error
    _validate_public_paths(root, runs)

    governance = _read_governance(root)
    try:
        validate_run_manifest(runs, governance.routes, governance.allocations, governance.activations)
    except ManifestError as error:
        raise ArchiveError(error.code) from error

    try:
        aggregates = read_aggregate_manifest(paths["aggregate"])
        validate_aggregate_manifest(aggregates, runs)
    except AggregateManifestError as error:
        raise ArchiveError(error.code) from error
    try:
        claims = read_claim_scope(paths["claim"])
        validate_claim_scope(claims, aggregates)
    except ClaimScopeError as error:
        raise ArchiveError(error.code) from error

    _validate_config_commits(runs)
    _validate_route_provenance(runs, governance)
    _validate_activation_provenance(runs, governance)
    return 3 + len(runs.rows) * 4 + len(aggregates.rows) + len(claims.rows)


@dataclass(frozen=True)
class _Governance:
    routes: Mapping[str, RouteSelectionManifest]
    allocations: Mapping[str, SeedAllocationManifest]
    activations: Mapping[str, ExploratoryActivationManifest]
    reports: Mapping[str, RouteFeasibilityReport]
    pilots: Mapping[str, PilotBManifest]
    mfts: Mapping[str, MftManifest]
    plans: Mapping[str, CodeMatrixPlan]
    resources: Mapping[str, SelectedPackageResourceManifest]


def _read_governance(root: Path) -> _Governance:
    directory = root / "governance"
    return _Governance(
        routes=_read_models(directory / _GOVERNANCE_PATHS["routes"], RouteSelectionManifest),
        allocations=_read_models(directory / _GOVERNANCE_PATHS["allocations"], SeedAllocationManifest),
        activations=_read_models(
            directory / _GOVERNANCE_PATHS["activations"], ExploratoryActivationManifest
        ),
        reports=_read_models(directory / _GOVERNANCE_PATHS["reports"], RouteFeasibilityReport),
        pilots=_read_models(directory / _GOVERNANCE_PATHS["pilot"], PilotBManifest),
        mfts=_read_models(directory / _GOVERNANCE_PATHS["mft"], MftManifest),
        plans=_read_models(directory / _GOVERNANCE_PATHS["plans"], CodeMatrixPlan),
        resources=_read_models(
            directory / _GOVERNANCE_PATHS["resources"], SelectedPackageResourceManifest
        ),
    )


def _read_models(path: Path, model: type[_Model]) -> dict[str, _Model]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        parsed = [model.model_validate(record) for record in records]
    except (OSError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as error:
        raise ArchiveError("GOVERNANCE_SCHEMA_INVALID") from error
    return {_governance_id(record): record for record in parsed}


def _governance_id(record: BaseModel) -> str:
    for name in ("manifest_id", "plan_id", "report_id"):
        value = getattr(record, name, None)
        if isinstance(value, str) and value:
            return value
    raise ArchiveError("GOVERNANCE_SCHEMA_INVALID")


def _validate_public_paths(root: Path, manifest: RunManifest) -> None:
    for row in manifest.rows:
        for value in (
            row.config_path,
            row.raw_log_path,
            row.output_path,
            row.public_artifact_manifest_path,
        ):
            path = Path(value).resolve()
            try:
                relative = path.relative_to(root)
            except ValueError as error:
                raise ArchiveError("PUBLIC_ARTIFACT_OUTSIDE_ARCHIVE") from error
            if "audit" in relative.parts:
                raise ArchiveError("AUDIT_ARTIFACT_REFERENCE_FORBIDDEN")


def _validate_config_commits(manifest: RunManifest) -> None:
    configs: dict[str, str] = {}
    for row in manifest.rows:
        commit = configs.get(row.config_path)
        if commit is None:
            try:
                commit = load_phase12_config(Path(row.config_path)).repository_commit
            except Phase12ConfigError as error:
                raise ArchiveError("CONFIG_REFERENCE_INVALID") from error
            configs[row.config_path] = commit
        if row.git_commit != commit:
            raise ArchiveError("REPOSITORY_COMMIT_MISMATCH")


def _validate_route_provenance(manifest: RunManifest, governance: _Governance) -> None:
    selections = {
        row.route_selection_manifest_id
        for row in manifest.rows
        if isinstance(row, SelectedRouteRunManifestRow)
        or (isinstance(row, ExploratoryRunManifestRow) and row.metadata_kind == "exploratory_code_scientific")
    }
    for selection_id in selections:
        if selection_id is None:
            continue
        selection = governance.routes.get(selection_id)
        if selection is None:
            continue
        allocation = governance.allocations.get(selection.seed_allocation_manifest_id)
        report = governance.reports.get(selection.selected_feasibility_report_id)
        if allocation is None or report is None:
            raise ArchiveError("ROUTE_SELECTION_MISMATCH")
        pilot = governance.pilots.get(report.pilot_b_manifest_id)
        if pilot is None or pilot.artifact_hash != report.pilot_b_manifest_hash:
            raise ArchiveError("PILOT_B_MANIFEST_HASH_MISMATCH")
        mft = governance.mfts.get(report.mft_manifest_id)
        if mft is None or mft.artifact_hash != report.mft_manifest_hash:
            raise ArchiveError("MFT_MANIFEST_HASH_MISMATCH")
        try:
            validate_route_selection(tuple(governance.reports.values()), pilot, mft, selection, allocation)
        except PlanningError as error:
            code = "ROUTE_SELECTION_MISMATCH" if error.code == "ROUTE_SELECTION_INVALID" else error.code
            raise ArchiveError(code) from error
        for row in manifest.rows:
            if (
                isinstance(row, SelectedRouteRunManifestRow)
                and row.route_selection_manifest_id == selection_id
                and (
                    row.run_template_registry_id != allocation.run_template_registry_id
                    or row.run_template_registry_hash != allocation.run_template_registry_hash
                )
            ):
                raise ArchiveError("RUN_TEMPLATE_REGISTRY_MISMATCH")


def _validate_activation_provenance(manifest: RunManifest, governance: _Governance) -> None:
    for row in manifest.rows:
        if not isinstance(row, ExploratoryRunManifestRow) or row.metadata_kind != "exploratory_code_scientific":
            continue
        activation_id = row.exploratory_activation_manifest_id
        if activation_id is None:
            continue
        activation = governance.activations.get(activation_id)
        if activation is None:
            continue
        plan = governance.plans.get(activation.exploratory_plan_id)
        if plan is None or plan.artifact_hash != activation.exploratory_plan_hash:
            raise ArchiveError("EXPLORATORY_PLAN_HASH_MISMATCH")
        resource = governance.resources.get(activation.resource_manifest_id)
        if resource is None or resource.artifact_hash != activation.resource_manifest_hash:
            raise ArchiveError("EXPLORATORY_RESOURCE_RESERVATION_NOT_PASS")
        if (
            resource.exploratory_plan_id != plan.plan_id
            or resource.exploratory_plan_hash != plan.artifact_hash
            or activation.exploratory_run_template_registry_id != plan.exploratory_run_template_registry_id
            or activation.exploratory_run_template_registry_hash
            != plan.exploratory_run_template_registry_hash
        ):
            raise ArchiveError("EXPLORATORY_PLAN_HASH_MISMATCH")
        if (
            row.run_template_registry_id != plan.exploratory_run_template_registry_id
            or row.run_template_registry_hash != plan.exploratory_run_template_registry_hash
        ):
            raise ArchiveError("RUN_TEMPLATE_REGISTRY_MISMATCH")
        if resource.mandatory_package_status != "fully_resourced":
            raise ArchiveError("EXPLORATORY_RESOURCE_RESERVATION_NOT_PASS")
        if plan.estimated_exploratory_calls > resource.exploratory_call_budget:
            raise ArchiveError("EXPLORATORY_BUDGET_INSUFFICIENT")
        if resource.exploratory_call_budget + resource.reproducibility_reserve > resource.remaining_call_capacity:
            raise ArchiveError("REPRODUCIBILITY_RESERVE_INSUFFICIENT")
        selection = governance.routes.get(activation.route_selection_manifest_id)
        allocation = governance.allocations.get(activation.seed_allocation_manifest_id)
        if selection is None or allocation is None:
            raise ArchiveError("ACTIVATION_MISMATCH")
        report = governance.reports.get(selection.selected_feasibility_report_id)
        pilot = None if report is None else governance.pilots.get(report.pilot_b_manifest_id)
        mft = None if report is None else governance.mfts.get(report.mft_manifest_id)
        if report is None or pilot is None or mft is None:
            raise ArchiveError("ACTIVATION_MISMATCH")
        try:
            validated_route = validate_route_selection(
                tuple(governance.reports.values()), pilot, mft, selection, allocation
            )
            validate_exploratory_activation(plan, resource, activation, validated_route)
        except PlanningError as error:
            raise ArchiveError("ACTIVATION_MISMATCH") from error


__all__ = ["ArchiveError", "ArchiveValidationReport", "validate_archive"]
