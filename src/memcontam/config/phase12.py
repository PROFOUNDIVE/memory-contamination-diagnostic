from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator

from memcontam.experiment.phase12.contracts import (
    CandidateTemplateSet,
    BaselineConditionSpec,
    PrefixExecutionKey,
    RouteCandidateId,
    SensitivityCellSpec,
    ExploratoryGovernanceArtifact,
    RouteGovernanceArtifact,
    canonical_json_hash,
)


_REPOSITORY_COMMIT = "830b89c8c169ffa9cdea472887fdae134dbae7cf"
_DESIGN_SHA256 = "984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0"
_LEGAL_EVIDENCE_LAYERS = {"build", "calibration", "main", "extension"}
_REQUIRED_TEMPLATE_PACKAGE_FIELDS = {
    "runtime_refs",
    "call_policy",
    "core",
    "sensitivity",
    "replication",
    "extension",
    "model_role_contract",
    "expected_template_counts_by_route",
}
_REQUIRED_RUNTIME_REF_FIELDS = {
    "behavior_test_registry_id",
    "behavior_test_registry_hash",
    "inv03_equivalence_registry_id",
    "inv03_equivalence_registry_hash",
    "primary_capacity_contract_id",
    "primary_corpus_family",
    "prompt_version",
    "rag_branch_index_policy_version",
    "tool_contract_hash",
    "filter_view_policy_version",
}
_REQUIRED_CALL_POLICY_FIELDS = {
    "component_policy_version",
    "call_cost_registry_id",
    "call_cost_registry_hash",
    "conditional_call_rate_registry_id",
    "conditional_call_rate_registry_hash",
    "conditional_call_scope_registry_id",
    "conditional_call_scope_registry_hash",
    "conservative_rate_upper_bound_registry_id",
    "conservative_rate_upper_bound_registry_hash",
    "pilot_call_statistics_manifest_id",
    "pilot_call_statistics_manifest_hash",
    "reflexion_terminal_failure_reflection",
}


class Phase12ConfigError(ValueError):
    pass


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentDesignRef(_ConfigModel):
    path: str
    sha256: str


class LoggingContractRef(_ConfigModel):
    contract_level: str
    schema_version: str


class RouteCandidateConfigSpec(_ConfigModel):
    candidate_route: RouteCandidateId
    capacity_is_net_of_pilot_allowance: bool
    expected_candidate_template_set_hash: str
    expected_template_counts: dict[str, int]
    extension_enabled: bool
    main_backbone_snapshot: str
    max_calls: int
    replication_analysis_status: str
    replication_backbone_snapshot: str
    requested_core_counts: dict[str, int]
    requested_count_vector_source: str
    requested_extension_counts: dict[str, int]


class InvalidVariantSpec(_ConfigModel):
    id: str
    reason: str
    remove: str | None = None
    patch: dict[str, Any] | None = None


class Phase12StudyConfig(_ConfigModel):
    fixture_id: str
    protocol_version: Literal["phase12_primary_v1"]
    repository_commit: str
    authoritative_experiment_design: ExperimentDesignRef
    logging_contract: LoggingContractRef
    arms: tuple[Literal["clean", "correct", "irrelevant", "contam", "filter"], ...]
    tasks: tuple[str, ...]
    conditions: tuple[BaselineConditionSpec, ...]
    sensitivity_cells: tuple[SensitivityCellSpec, ...]
    template_package: dict[str, Any]
    route_candidates: tuple[RouteCandidateConfigSpec, ...]
    registry_ids: dict[str, str]
    registry_hashes: dict[str, str]
    tool_mode: Literal["text_only", "python_sandbox"]
    selection_status: Literal["unselected", "selected"]
    exploratory_activation_status: Literal["inactive", "active"]
    template_package_hash: str
    prefix_execution_key: PrefixExecutionKey
    prefix_identity_fields: tuple[str, ...]
    seed_slots: tuple[str, ...]
    route_selection_manifest_id: str | None
    seed_allocation_manifest_id: str | None
    invalid_variants: tuple[InvalidVariantSpec, ...]
    canonical_candidate_template_sets: dict[RouteCandidateId, dict[str, Any]]

    @model_validator(mode="before")
    @classmethod
    def _reject_unrelated_sensitivity_fields(cls, value: Any) -> Any:
        factor_fields = {
            "timing": "timing_quantile",
            "horizon": "horizon",
            "affinity": "affinity_band",
            "fh_budget": "fh_context_budget_id",
            "embedding": "embedding_contract_id",
            "behavior": "behavior_test_id",
        }
        if isinstance(value, dict):
            logging_contract = value.get("logging_contract")
            if (
                isinstance(logging_contract, dict)
                and logging_contract.get("schema_version") != "logging_v3"
            ):
                raise ValueError("LOGGING_CONTRACT_MISMATCH")
            prefix_key = value.get("prefix_execution_key")
            if isinstance(prefix_key, dict) and prefix_key.get("kind") != "branch_free_prefix":
                raise ValueError("PREFIX_EXECUTION_KEY_REQUIRED")
            for condition in value.get("conditions", []):
                if (
                    isinstance(condition, dict)
                    and condition.get("baseline_family") == "nomem"
                    and condition.get("execution_key", {}).get("kind") != "nomem_singleton"
                ):
                    raise ValueError("NOMEM_ARM_FORBIDDEN")
            if (
                value.get("selection_status") == "selected"
                and value.get("seed_allocation_manifest_id") is None
            ):
                raise ValueError("SEED_ALLOCATION_REQUIRED")
            if value.get("exploratory_activation_status") == "active":
                raise ValueError("EXPLORATORY_ACTIVATION_REQUIRED")
            for cell in value.get("sensitivity_cells", []):
                if isinstance(cell, dict) and cell.get("kind") in factor_fields:
                    allowed = {"kind", "cell_id", "base_cell_id", factor_fields[cell["kind"]]}
                    if set(cell) - allowed:
                        raise ValueError("UNRELATED_SENSITIVITY_FIELD")
        return value

    @model_validator(mode="after")
    def _validate_contract(self) -> Phase12StudyConfig:
        if self.repository_commit != _REPOSITORY_COMMIT:
            raise ValueError("REPOSITORY_COMMIT_MISMATCH")
        if self.authoritative_experiment_design.sha256 != _DESIGN_SHA256:
            raise ValueError("EXPERIMENT_DESIGN_HASH_MISMATCH")
        if (
            self.logging_contract.contract_level != "phase12"
            or self.logging_contract.schema_version != "logging_v3"
        ):
            raise ValueError("LOGGING_CONTRACT_MISMATCH")
        if set(self.arms) != {"clean", "correct", "irrelevant", "contam", "filter"}:
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        missing = _REQUIRED_TEMPLATE_PACKAGE_FIELDS - set(self.template_package)
        if missing or not self.sensitivity_cells:
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if self.tool_mode != "text_only":
            raise ValueError("PRIMARY_TOOL_FORBIDDEN")
        runtime_refs = self.template_package["runtime_refs"]
        call_policy = self.template_package["call_policy"]
        if (
            not isinstance(runtime_refs, dict)
            or not isinstance(call_policy, dict)
            or _REQUIRED_RUNTIME_REF_FIELDS - set(runtime_refs)
            or _REQUIRED_CALL_POLICY_FIELDS - set(call_policy)
        ):
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if set(self.template_package.get("legal_evidence_layers", ())) != _LEGAL_EVIDENCE_LAYERS:
            raise ValueError("INVALID_TEMPLATE_EVIDENCE_LAYER")
        roles = self.template_package["model_role_contract"]
        core = self.template_package["core"]
        replication = self.template_package["replication"]
        extension = self.template_package["extension"]
        if not all(isinstance(value, dict) for value in (roles, core, replication, extension)):
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if (
            core.get("model_snapshot") != roles.get("main_backbone_snapshot")
            or replication.get("model_snapshot") != roles.get("replication_backbone_snapshot")
            or extension.get("model_snapshot") != roles.get("main_backbone_snapshot")
        ):
            raise ValueError("INVALID_MODEL_ROLE_ASSIGNMENT")
        sensitivity = self.template_package["sensitivity"]
        if not isinstance(sensitivity, list) or {
            item.get("family") for item in sensitivity if isinstance(item, dict)
        } != {"timing", "horizon", "affinity", "fh_budget", "behavior", "embedding"}:
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if any(
            not isinstance(item, dict) or item.get("evidence_layer") not in _LEGAL_EVIDENCE_LAYERS
            for item in sensitivity
        ):
            raise ValueError("INVALID_TEMPLATE_EVIDENCE_LAYER")
        if len(self.sensitivity_cells) != 12 or len(self.route_candidates) != 2:
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if {candidate.candidate_route for candidate in self.route_candidates} != {"3w", "5w"}:
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if (
            self.registry_ids.get("behavior") != runtime_refs["behavior_test_registry_id"]
            or self.registry_hashes.get("behavior") != runtime_refs["behavior_test_registry_hash"]
            or self.registry_hashes.get("inv03_equivalence")
            != runtime_refs["inv03_equivalence_registry_hash"]
        ):
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if self.selection_status == "unselected":
            if (
                self.route_selection_manifest_id is not None
                or self.seed_allocation_manifest_id is not None
            ):
                raise ValueError("PREMATURE_ROUTE_SELECTION")
        elif self.seed_allocation_manifest_id is None:
            raise ValueError("SEED_ALLOCATION_REQUIRED")
        if self.exploratory_activation_status == "active":
            raise ValueError("EXPLORATORY_ACTIVATION_REQUIRED")
        return self


class ResolvedPhase12Config(_ConfigModel):
    source: Phase12StudyConfig
    repository_commit: str
    authoritative_experiment_design: ExperimentDesignRef
    logging_schema_version: Literal["logging_v3"]
    contract_level: Literal["phase12"]
    template_package_hash: str


def load_phase12_config(path: Path) -> Phase12StudyConfig:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        return Phase12StudyConfig.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, yaml.YAMLError) as exc:
        message = str(exc)
        if "UNRELATED_SENSITIVITY_FIELD" in message:
            raise Phase12ConfigError("UNRELATED_SENSITIVITY_FIELD") from exc
        raise Phase12ConfigError(
            "REQUIRED_CONFIG_CELL_MISSING" if "Field required" in message else message
        ) from exc


def resolve_phase12_config(config: Phase12StudyConfig) -> ResolvedPhase12Config:
    return ResolvedPhase12Config(
        source=config,
        repository_commit=config.repository_commit,
        authoritative_experiment_design=config.authoritative_experiment_design,
        logging_schema_version="logging_v3",
        contract_level="phase12",
        template_package_hash=config.template_package_hash,
    )


def build_candidate_template_set(
    config: ResolvedPhase12Config, candidate: RouteCandidateId
) -> CandidateTemplateSet:
    payload = config.source.canonical_candidate_template_sets[candidate]
    expected_hash = next(
        route.expected_candidate_template_set_hash
        for route in config.source.route_candidates
        if route.candidate_route == candidate
    )
    if (
        payload.get("candidate_route") != candidate
        or payload.get("repository_commit") != config.repository_commit
        or payload.get("authoritative_experiment_design_sha256")
        != config.authoritative_experiment_design.sha256
        or payload.get("template_package_hash") != config.template_package_hash
        or payload.get("artifact_hash") != expected_hash
        or canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
        != expected_hash
    ):
        raise Phase12ConfigError("CANONICAL_CANDIDATE_TEMPLATE_MISMATCH")
    return CandidateTemplateSet.model_validate(payload)


def _parse_route_governance_shape(record: dict[str, Any]) -> RouteGovernanceArtifact:
    return TypeAdapter(RouteGovernanceArtifact).validate_python(record)


def _parse_exploratory_governance_shape(record: dict[str, Any]) -> ExploratoryGovernanceArtifact:
    return TypeAdapter(ExploratoryGovernanceArtifact).validate_python(record)
