from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

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
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.rootless_local_firewall import (
    ROOTLESS_PROFILE_FORBIDDEN,
    has_forbidden_rootless_profile,
)


class RegistryClosure(StrictRegistry):
    search_config_id: str
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str
    operational_probe_suite_manifest_id: str
    operational_probe_suite_manifest_hash: str


class StageGateResult(StrictRegistry):
    stage: Literal["build", "pilot_b", "main"]
    reason_code: Literal["SELECTED_POLICY_REQUIRED"] | None


def validate_registry_closure(
    search_config: SearchConfig,
    inventory: ProbeInventoryRegistry,
    suite: OperationalSuiteRegistry,
) -> RegistryClosure:
    if (
        search_config.calibration_probe_inventory_id != inventory.registry_id
        or search_config.calibration_probe_inventory_manifest_hash
        != inventory.calibration_probe_inventory_manifest_hash
    ):
        raise RegistryValidationError("INVENTORY_MANIFEST_REFERENCE_MISMATCH")
    if set(search_config.calibration_probe_ids) != set(inventory.probe_ids):
        raise RegistryValidationError("CALIBRATION_PROBE_IDS_MISMATCH")
    if (
        search_config.operational_probe_suite_manifest_id != suite.registry_id
        or search_config.operational_probe_suite_manifest_hash
        != suite.operational_probe_suite_manifest_hash
    ):
        raise RegistryValidationError("SUITE_MANIFEST_REFERENCE_MISMATCH")
    if {candidate.operational_probe_suite_id for candidate in search_config.suite_candidates} != set(
        suite.suite_ids
    ):
        raise RegistryValidationError("SUITE_CANDIDATE_IDS_MISMATCH")
    return RegistryClosure(
        search_config_id=search_config.registry_id,
        calibration_probe_inventory_id=inventory.registry_id,
        calibration_probe_inventory_manifest_hash=inventory.calibration_probe_inventory_manifest_hash,
        operational_probe_suite_manifest_id=suite.registry_id,
        operational_probe_suite_manifest_hash=suite.operational_probe_suite_manifest_hash,
    )


def validate_stage(
    search_config: SearchConfig | Mapping[str, JsonValue],
    selected_policy: SelectedPolicy | Mapping[str, JsonValue] | None = None,
    *,
    stage: Literal["build", "pilot_b", "main"],
) -> StageGateResult:
    if isinstance(search_config, Mapping):
        if has_forbidden_rootless_profile(search_config):
            raise RegistryValidationError(ROOTLESS_PROFILE_FORBIDDEN)
        search_config = SearchConfig.model_validate(search_config)
    if isinstance(selected_policy, Mapping):
        selected_policy = parse_selected_policy(selected_policy)
    if stage == "main" and selected_policy is None:
        return StageGateResult(stage=stage, reason_code="SELECTED_POLICY_REQUIRED")
    if selected_policy is not None:
        _validate_selected_policy(search_config, selected_policy)
    return StageGateResult(stage=stage, reason_code=None)


def parse_selected_policy(payload: Mapping[str, JsonValue]) -> SelectedPolicy:
    if has_forbidden_rootless_profile(payload):
        raise RegistryValidationError(ROOTLESS_PROFILE_FORBIDDEN)
    return SelectedPolicy.model_validate(payload)


def _validate_selected_policy(search: SearchConfig, policy: SelectedPolicy) -> None:
    if policy.search_config_id != search.registry_id or policy.search_config_hash != search.search_config_hash:
        raise RegistryValidationError("SELECTED_POLICY_SEARCH_CONFIG_MISMATCH")
    allowed = (
        (policy.operational_probe_suite_id, {item.operational_probe_suite_id for item in search.suite_candidates}),
        (policy.kappa_id, {item.kappa_id for item in search.kappa_candidates}),
        (policy.coverage_contract_id, {item.coverage_contract_id for item in search.coverage_contract_candidates}),
        (policy.replicate_retry_id, {item.replicate_retry_id for item in search.replicate_retry_candidates}),
        (policy.canonicalizer_id, {item.canonicalizer_id for item in search.canonicalizer_candidates}),
        (policy.tolerance_id, {item.tolerance_id for item in search.tolerance_candidates}),
        (policy.paired_evaluability_rate_id, {item.rate_id for item in search.paired_evaluability_candidates}),
        (policy.inclusion_rate_id, {item.rate_id for item in search.inclusion_rate_candidates}),
        (policy.ordinary_route_coverage_id, {item.rate_id for item in search.ordinary_route_coverage_candidates}),
        (policy.budget_cap_id, {item.budget_cap_id for item in search.budget_cap_candidates}),
        (policy.ci_procedure_id, {item.ci_procedure_id for item in search.ci_procedure_candidates}),
        (policy.tie_break_id, {item.tie_break_id for item in search.deterministic_tie_break_candidates}),
    )
    if any(value not in candidates for value, candidates in allowed):
        raise RegistryValidationError("SELECTED_POLICY_REFERENCE_UNKNOWN")
