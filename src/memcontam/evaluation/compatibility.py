from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from pydantic import BaseModel

from memcontam.logging.schema_v3 import (
    LOGGING_V3,
    NonScientificExploratoryCodeRunMetadata,
    PreRouteRunMetadata,
    RunMetadataV3,
    ScientificExploratoryCodeRunMetadata,
    SelectedRouteRunMetadata,
)


class CompatibilityError(ValueError):
    def __init__(self, code: str, field: str) -> None:
        self.code = code
        self.field = field
        super().__init__(f"{code}: {field}")


@dataclass(frozen=True)
class CompatibilityKey:
    schema_version: str
    contract_level: str
    metadata_kind: str
    protocol_version: str
    evidence_layer: str
    run_family: str
    task_family: str
    baseline_condition_id: str
    run_template_id: str
    execution_key: str
    prefix_template_key_or_none: str | None
    sensitivity_cell_ref: str
    metric_registry_version: str
    embedding_contract_hash: str
    tool_contract_hash: str
    candidate_registry_version: str
    split_manifest_version: str
    behavior_registry_version: str
    run_template_registry_version: str
    rerun_policy_version: str
    scientific_admission: str | None
    route_selection_manifest_id: str | None
    seed_allocation_manifest_id: str | None
    exploratory_activation_manifest_id: str | None


_METADATA_TYPES = (
    PreRouteRunMetadata,
    SelectedRouteRunMetadata,
    NonScientificExploratoryCodeRunMetadata,
    ScientificExploratoryCodeRunMetadata,
)


def _canonical(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical_or_none(value: Any) -> str | None:
    return None if value is None else _canonical(value)


def _require_value(value: object, code: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompatibilityError(code, field)
    return value


def _require_evidence(value: Any) -> str:
    if value is None:
        raise CompatibilityError("SCIENTIFIC_ADMISSION_REQUIRED", "scientific_admission")
    return _canonical(value)


def _validate_applicability(
    run: RunMetadataV3,
) -> tuple[str | None, str | None, str | None, str | None]:
    if isinstance(run, PreRouteRunMetadata):
        if run.route_selection_manifest_id is not None:
            raise CompatibilityError("GOVERNANCE_FORBIDDEN", "route_selection_manifest_id")
        if run.seed_allocation_manifest_id is not None:
            raise CompatibilityError("GOVERNANCE_FORBIDDEN", "seed_allocation_manifest_id")
        return _canonical_or_none(run.scientific_admission_ref_or_none), None, None, None
    if isinstance(run, SelectedRouteRunMetadata):
        return (
            _require_evidence(run.scientific_admission_ref),
            _require_value(
                run.route_selection_manifest_id,
                "ROUTE_SELECTION_REQUIRED",
                "route_selection_manifest_id",
            ),
            _require_value(
                run.seed_allocation_manifest_id,
                "SEED_ALLOCATION_REQUIRED",
                "seed_allocation_manifest_id",
            ),
            None,
        )
    if isinstance(run, NonScientificExploratoryCodeRunMetadata):
        for field in (
            "source_route_selection_manifest_id",
            "source_seed_allocation_manifest_id",
            "exploratory_activation_manifest_id",
        ):
            if getattr(run, field) is not None:
                raise CompatibilityError("EXPLORATORY_ACTIVATION_FORBIDDEN", field)
        return _canonical_or_none(run.scientific_admission_ref_or_none), None, None, None
    if isinstance(run, ScientificExploratoryCodeRunMetadata):
        return (
            _require_evidence(run.scientific_admission_ref),
            _require_value(
                run.source_route_selection_manifest_id,
                "ROUTE_SELECTION_REQUIRED",
                "source_route_selection_manifest_id",
            ),
            _require_value(
                run.source_seed_allocation_manifest_id,
                "SEED_ALLOCATION_REQUIRED",
                "source_seed_allocation_manifest_id",
            ),
            _require_value(
                run.exploratory_activation_manifest_id,
                "EXPLORATORY_ACTIVATION_REQUIRED",
                "exploratory_activation_manifest_id",
            ),
        )
    raise CompatibilityError("UNKNOWN_METADATA_KIND", "metadata_kind")


def build_compatibility_key(run: RunMetadataV3) -> CompatibilityKey:
    if not isinstance(run, _METADATA_TYPES):
        raise CompatibilityError("SCHEMA_CONTRACT_MISMATCH", "metadata_kind")
    if run.schema_version != LOGGING_V3:
        raise CompatibilityError("SCHEMA_CONTRACT_MISMATCH", "schema_version")
    if run.contract_level != "phase12":
        raise CompatibilityError("SCHEMA_CONTRACT_MISMATCH", "contract_level")

    scientific_admission, route_selection, seed_allocation, activation = _validate_applicability(
        run
    )
    return CompatibilityKey(
        schema_version=run.schema_version,
        contract_level=run.contract_level,
        metadata_kind=run.metadata_kind,
        protocol_version=run.protocol_version,
        evidence_layer=run.evidence_layer,
        run_family=run.run_family,
        task_family=run.task_family,
        baseline_condition_id=run.baseline_condition_id,
        run_template_id=run.run_template_id,
        execution_key=_canonical(run.execution_key),
        prefix_template_key_or_none=run.prefix_template_key_or_none,
        sensitivity_cell_ref=_canonical(run.sensitivity_cell_ref),
        metric_registry_version=run.metric_registry_version,
        embedding_contract_hash=run.embedding_contract_hash,
        tool_contract_hash=run.tool_contract_hash,
        candidate_registry_version=run.candidate_registry_version,
        split_manifest_version=run.split_manifest_version,
        behavior_registry_version=run.behavior_registry_version,
        run_template_registry_version=run.run_template_registry_version,
        rerun_policy_version=run.rerun_policy_version,
        scientific_admission=scientific_admission,
        route_selection_manifest_id=route_selection,
        seed_allocation_manifest_id=seed_allocation,
        exploratory_activation_manifest_id=activation,
    )


def _first_difference(expected: CompatibilityKey, found: CompatibilityKey) -> str:
    for field in fields(CompatibilityKey):
        if getattr(expected, field.name) != getattr(found, field.name):
            return field.name
    raise RuntimeError("identical compatibility keys have no differing field")


def validate_compatible_runs(runs: Sequence[RunMetadataV3]) -> CompatibilityKey:
    if not runs:
        raise CompatibilityError("EMPTY_RUNS", "runs")

    key = build_compatibility_key(runs[0])
    seed_assignments: dict[str, int] = {}
    for run in runs:
        candidate = build_compatibility_key(run)
        if candidate != key:
            raise CompatibilityError("COMPATIBILITY_MISMATCH", _first_difference(key, candidate))
        if candidate.metadata_kind not in {"selected_route", "exploratory_code_scientific"}:
            continue
        slot = _require_value(
            run.abstract_seed_slot_or_none,
            "SEED_ALLOCATION_REQUIRED",
            "abstract_seed_slot_or_none",
        )
        assigned_seed = seed_assignments.setdefault(slot, run.trajectory_seed)
        if assigned_seed != run.trajectory_seed:
            raise CompatibilityError("SEED_ASSIGNMENT_MISMATCH", "abstract_seed_slot_or_none")
    return key
