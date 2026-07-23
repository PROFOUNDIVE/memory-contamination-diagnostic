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
from memcontam.phase12_types import CanonicalRunFamily


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
_CANONICAL_CONFIG_NAMES = (
    "readiness.yaml",
    "pilot_a.yaml",
    "pilot_b.yaml",
    "main_3w.yaml",
    "main_5w.yaml",
    "exploratory_code.yaml",
)
_PRIMARY_REGISTRY_IDS = {
    "behavior": "behavior-registry-v2-complete",
    "candidate": "candidate-v1",
    "embedding": "bge-m3-pinned-v1",
    "inv03_equivalence": "inv03-equivalence-registry-v1",
    "metric": "metric-v1",
    "split": "split-v1",
}
_PRIMARY_REGISTRY_HASHES = {
    "behavior": "52627253c96cd0f1592e74a32e26f81bd68749f7b1c76c15396abaf0ae02cd36",
    "inv03_equivalence": "ffdb247dc187d462208dbe9f7a4ead8bfa27def24a3052baedca77c50aa2e620",
}
_EXPLORATORY_REGISTRY_IDS = {
    **_PRIMARY_REGISTRY_IDS,
    "exploratory_run_templates": "exploratory-registry-v1",
}
_ROUTE_TEMPLATE_REGISTRIES = {
    "3w": (
        "registry-3w-complete-v3",
        "e4048b1dc6af2b68d187a99e14ea3a4b8f8388d64a020f428b90de62b4d88fc0",
    ),
    "5w": (
        "registry-5w-complete-v3",
        "292c01743ab78b27724a210ebef5e4320171fd19a8ba9d70c1e85476aca16404",
    ),
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
    patch: dict[str, object] | None = None


class CanonicalRouteInput(_ConfigModel):
    candidate_route: RouteCandidateId
    max_calls: int
    requested_core_counts: dict[str, int]
    requested_extension_counts: dict[str, int]
    run_template_registry_id: str
    run_template_registry_hash: str

    @model_validator(mode="after")
    def _validate_frozen_registry(self) -> CanonicalRouteInput:
        expected_id, expected_hash = _ROUTE_TEMPLATE_REGISTRIES[self.candidate_route]
        if self.run_template_registry_id != expected_id:
            raise ValueError("FROZEN_REGISTRY_ID_UNKNOWN")
        if self.run_template_registry_hash != expected_hash:
            raise ValueError("FROZEN_REGISTRY_HASH_UNKNOWN")
        return self


class CanonicalPrimaryConfig(_ConfigModel):
    config_kind: Literal["phase12_canonical_primary_v1"]
    config_id: str
    protocol_version: Literal["phase12_primary_v1"]
    repository_commit: str
    authoritative_experiment_design: ExperimentDesignRef
    logging_contract: LoggingContractRef
    run_family: CanonicalRunFamily
    evidence_layer: Literal["build", "calibration", "main"]
    arms: tuple[Literal["clean", "correct", "irrelevant", "contam", "filter"], ...]
    tasks: tuple[str, ...]
    tool_mode: Literal["text_only"]
    selection_status: Literal["unselected", "candidate"]
    candidate_route: RouteCandidateId | None
    registry_ids: dict[str, str]
    registry_hashes: dict[str, str]
    candidate_routes: tuple[CanonicalRouteInput, ...]
    route_selection_manifest_id: None = None
    seed_allocation_manifest_id: None = None

    @model_validator(mode="after")
    def _validate_canonical_contract(self) -> CanonicalPrimaryConfig:
        _validate_canonical_common(self)
        if self.run_family == "exploratory_code":
            raise ValueError("CANONICAL_RUN_FAMILY_INVALID")
        _validate_registry_ids(self.registry_ids, _PRIMARY_REGISTRY_IDS)
        _validate_registry_ids(self.registry_hashes, _PRIMARY_REGISTRY_HASHES)
        if self.run_family == "main":
            if self.selection_status != "candidate" or self.candidate_route is None:
                raise ValueError("CANDIDATE_ROUTE_REQUIRED")
            if {route.candidate_route for route in self.candidate_routes} != {self.candidate_route}:
                raise ValueError("CANDIDATE_ROUTE_REQUIRED")
        elif self.selection_status != "unselected" or self.candidate_route is not None:
            raise ValueError("PREMATURE_ROUTE_SELECTION")
        elif {route.candidate_route for route in self.candidate_routes} != {"3w", "5w"}:
            raise ValueError("CANDIDATE_ROUTE_REQUIRED")
        return self


class CanonicalExploratoryConfig(_ConfigModel):
    config_kind: Literal["phase12_canonical_exploratory_v1"]
    config_id: str
    protocol_version: Literal["phase12_code_exploratory_v1"]
    repository_commit: str
    authoritative_experiment_design: ExperimentDesignRef
    logging_contract: LoggingContractRef
    run_family: CanonicalRunFamily
    evidence_layer: Literal["main"]
    selection_status: Literal["unselected"]
    candidate_route: None = None
    activation_status: Literal["inactive"]
    task_family: Literal["game24"]
    baseline_condition_ids: tuple[Literal["nomem", "bot_style", "dc_rs"], ...]
    registry_ids: dict[str, str]
    registry_hashes: dict[str, str]
    exploratory_run_template_registry_id: str
    exploratory_run_template_registry_hash: str
    abstract_slots: tuple[str, ...]
    estimated_exploratory_calls: int
    oci_contract_path: str
    route_selection_manifest_id: None = None
    seed_allocation_manifest_id: None = None
    selected_package_resource_manifest_id: None = None
    exploratory_activation_manifest_id: None = None

    @model_validator(mode="after")
    def _validate_canonical_contract(self) -> CanonicalExploratoryConfig:
        _validate_canonical_common(self)
        if self.run_family != "exploratory_code":
            raise ValueError("CANONICAL_RUN_FAMILY_INVALID")
        _validate_registry_ids(self.registry_ids, _EXPLORATORY_REGISTRY_IDS)
        _validate_registry_ids(self.registry_hashes, _PRIMARY_REGISTRY_HASHES)
        if (
            self.exploratory_run_template_registry_id
            != self.registry_ids["exploratory_run_templates"]
        ):
            raise ValueError("CROSS_LAYER_REGISTRY_ID")
        if self.exploratory_run_template_registry_hash != _exploratory_registry_hash(self):
            raise ValueError("FROZEN_REGISTRY_HASH_UNKNOWN")
        return self


class TemplatePackageSpec(_ConfigModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    runtime_refs: dict[str, object]
    call_policy: dict[str, object]
    core: dict[str, object]
    sensitivity: tuple[dict[str, object], ...]
    replication: dict[str, object]
    extension: dict[str, object]
    model_role_contract: dict[str, object]
    expected_template_counts_by_route: dict[str, object]
    legal_evidence_layers: tuple[str, ...]


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
    template_package: TemplatePackageSpec
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
    canonical_candidate_template_sets: dict[RouteCandidateId, CandidateTemplateSet]

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
        if not self.sensitivity_cells:
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if self.tool_mode != "text_only":
            raise ValueError("PRIMARY_TOOL_FORBIDDEN")
        runtime_refs = self.template_package.runtime_refs
        call_policy = self.template_package.call_policy
        if _REQUIRED_RUNTIME_REF_FIELDS - set(runtime_refs) or _REQUIRED_CALL_POLICY_FIELDS - set(
            call_policy
        ):
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if set(self.template_package.legal_evidence_layers) != _LEGAL_EVIDENCE_LAYERS:
            raise ValueError("INVALID_TEMPLATE_EVIDENCE_LAYER")
        roles = self.template_package.model_role_contract
        core = self.template_package.core
        replication = self.template_package.replication
        extension = self.template_package.extension
        if (
            core.get("model_snapshot") != roles.get("main_backbone_snapshot")
            or replication.get("model_snapshot") != roles.get("replication_backbone_snapshot")
            or extension.get("model_snapshot") != roles.get("main_backbone_snapshot")
        ):
            raise ValueError("INVALID_MODEL_ROLE_ASSIGNMENT")
        sensitivity = self.template_package.sensitivity
        if {item.get("family") for item in sensitivity} != {
            "timing",
            "horizon",
            "affinity",
            "fh_budget",
            "behavior",
            "embedding",
        }:
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if any(item.get("evidence_layer") not in _LEGAL_EVIDENCE_LAYERS for item in sensitivity):
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
    source: Phase12StudyConfig | CanonicalPrimaryConfig | CanonicalExploratoryConfig
    repository_commit: str
    authoritative_experiment_design: ExperimentDesignRef
    logging_schema_version: Literal["logging_v3"]
    contract_level: Literal["phase12"]
    template_package_hash: str


def load_phase12_config(
    path: Path,
) -> Phase12StudyConfig | CanonicalPrimaryConfig | CanonicalExploratoryConfig:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError("REQUIRED_CONFIG_CELL_MISSING")
        if payload.get("config_kind") == "phase12_canonical_primary_v1":
            return CanonicalPrimaryConfig.model_validate(payload)
        if payload.get("config_kind") == "phase12_canonical_exploratory_v1":
            return CanonicalExploratoryConfig.model_validate(payload)
        return Phase12StudyConfig.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, yaml.YAMLError) as exc:
        message = str(exc)
        for code in (
            "UNRELATED_SENSITIVITY_FIELD",
            "FROZEN_REGISTRY_ID_MISSING",
            "CROSS_LAYER_REGISTRY_ID",
            "FROZEN_REGISTRY_ID_UNKNOWN",
            "FROZEN_REGISTRY_HASH_UNKNOWN",
        ):
            if code in message:
                raise Phase12ConfigError(code) from exc
        raise Phase12ConfigError(
            "REQUIRED_CONFIG_CELL_MISSING" if "Field required" in message else message
        ) from exc


def resolve_phase12_config(
    config: Phase12StudyConfig | CanonicalPrimaryConfig | CanonicalExploratoryConfig,
) -> ResolvedPhase12Config:
    return ResolvedPhase12Config(
        source=config,
        repository_commit=config.repository_commit,
        authoritative_experiment_design=config.authoritative_experiment_design,
        logging_schema_version="logging_v3",
        contract_level="phase12",
        template_package_hash=(
            config.template_package_hash
            if isinstance(config, Phase12StudyConfig)
            else canonical_json_hash(config.model_dump(mode="json"))
        ),
    )


def load_all_canonical_configs(root: Path) -> dict[str, ResolvedPhase12Config]:
    configs: dict[str, ResolvedPhase12Config] = {}
    for name in _CANONICAL_CONFIG_NAMES:
        path = root / name
        try:
            configs[name] = resolve_phase12_config(load_phase12_config(path))
        except Phase12ConfigError:
            raise
        except OSError as exc:
            raise Phase12ConfigError("CANONICAL_CONFIG_MISSING") from exc
    if {path.name for path in root.glob("*.yaml")} != set(_CANONICAL_CONFIG_NAMES):
        raise Phase12ConfigError("CANONICAL_CONFIG_SET_MISMATCH")
    return configs


def build_candidate_template_set(
    config: ResolvedPhase12Config, candidate: RouteCandidateId
) -> CandidateTemplateSet:
    if not isinstance(config.source, Phase12StudyConfig):
        raise Phase12ConfigError("CANONICAL_TEMPLATE_PAYLOAD_REQUIRED")
    payload = config.source.canonical_candidate_template_sets[candidate]
    expected_hash = next(
        route.expected_candidate_template_set_hash
        for route in config.source.route_candidates
        if route.candidate_route == candidate
    )
    canonical_payload = payload.model_dump(mode="json")
    if (
        payload.candidate_route != candidate
        or payload.repository_commit != config.repository_commit
        or payload.authoritative_experiment_design_sha256
        != config.authoritative_experiment_design.sha256
        or payload.template_package_hash != config.template_package_hash
        or payload.artifact_hash != expected_hash
        or canonical_json_hash(
            {key: value for key, value in canonical_payload.items() if key != "artifact_hash"}
        )
        != expected_hash
    ):
        raise Phase12ConfigError("CANONICAL_CANDIDATE_TEMPLATE_MISMATCH")
    return payload


def _parse_route_governance_shape(record: dict[str, Any]) -> RouteGovernanceArtifact:
    return TypeAdapter(RouteGovernanceArtifact).validate_python(record)


def _parse_exploratory_governance_shape(record: dict[str, Any]) -> ExploratoryGovernanceArtifact:
    return TypeAdapter(ExploratoryGovernanceArtifact).validate_python(record)


def _validate_canonical_common(
    config: CanonicalPrimaryConfig | CanonicalExploratoryConfig,
) -> None:
    if config.repository_commit != _REPOSITORY_COMMIT:
        raise ValueError("REPOSITORY_COMMIT_MISMATCH")
    if config.authoritative_experiment_design.sha256 != _DESIGN_SHA256:
        raise ValueError("EXPERIMENT_DESIGN_HASH_MISMATCH")
    if (
        config.logging_contract.contract_level != "phase12"
        or config.logging_contract.schema_version != "logging_v3"
    ):
        raise ValueError("LOGGING_CONTRACT_MISMATCH")


def _validate_registry_ids(actual: dict[str, str], expected: dict[str, str]) -> None:
    if set(actual) != set(expected):
        raise ValueError("FROZEN_REGISTRY_ID_MISSING")
    if actual == expected:
        return
    if any(value in _EXPLORATORY_REGISTRY_IDS.values() for value in actual.values()):
        raise ValueError("CROSS_LAYER_REGISTRY_ID")
    raise ValueError("FROZEN_REGISTRY_ID_UNKNOWN")


def _exploratory_registry_hash(config: CanonicalExploratoryConfig) -> str:
    return canonical_json_hash(
        {
            "abstract_slots": config.abstract_slots,
            "estimated_exploratory_calls": config.estimated_exploratory_calls,
            "registry_id": config.exploratory_run_template_registry_id,
        }
    )
