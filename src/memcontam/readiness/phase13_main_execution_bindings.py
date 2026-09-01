from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Final

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze


EXPECTED_AUTHORITIES: Final = {
    "authority_router": CORE_MAIN_REGISTRY.authority_router_sha256,
    **dict(CORE_MAIN_REGISTRY.authority_stack),
}
EXPECTED_ARTIFACT_PATHS: Final = {
    "mr_p4_manifest": "data/phase13/main/mr_p4/manifest_v1.json",
    "task_seed_orders": "data/phase13/main/mr_p4/task_seed_orders_v1.json",
    "common_checkpoint_registry": (
        "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json"
    ),
    "analysis_window_registry": "data/phase13/main/mr_p4/readiness0_window_proof_v1.json",
    "package_selection": "data/phase13/main/post_cutoff_package_selection_v2.json",
    "common_capacity": "data/phase13/common_capacity_v1.json",
    "legacy_rag_manifest": "data/phase13/rag/legacy/manifest.json",
    "legacy_rag_seal": "data/phase13/rag/legacy_seal_v1.json",
    "observability_manifest": "data/phase13/observability/manifest_v1.json",
    "observability_packet": "data/phase13/observability/registration_packet_v1.json",
    "provider_contract": "data/phase13/openai_luna_provider_contract_v1.json",
    "openai_client": "src/memcontam/clients/openai_responses.py",
    "stage_envelope_registry": "data/phase13/main/cost_envelope_v2/stage_envelope_registry_v1.json",
    "retry_failure_contract": "data/phase13/main/cost_envelope_v2/retry_failure_contract_v1.json",
    "cost_proof": "data/phase13/main/cost_envelope_v2/cost_proof_v1.json",
    "activated_cost_policy": "data/phase13/main/cost_envelope_v2/activated_policy_v1.json",
    "readiness0_acceptance_policy": "src/memcontam/readiness/phase13_readiness0_package.py",
    "readiness0_evidence_policy": (
        "src/memcontam/readiness/phase13_readiness0_evidence_validate.py"
    ),
    "ordinary_runtime": "src/memcontam/experiment/phase13_ordinary_runtime.py",
    "production_observability_adapter": (
        "src/memcontam/readiness/phase13_production_observability.py"
    ),
    "production_runtime_join": "src/memcontam/readiness/phase13_production_runtime_join.py",
    "production_runtime_evidence": (
        "src/memcontam/readiness/phase13_production_runtime_evidence.py"
    ),
    "production_runtime_memory": "src/memcontam/readiness/phase13_production_runtime_memory.py",
    "production_runtime_models": "src/memcontam/readiness/phase13_production_runtime_models.py",
    "logging_schema": "src/memcontam/logging/schema.py",
    "recording_client": "src/memcontam/clients/recording.py",
    "main_runner": "src/memcontam/readiness/phase13_main_runner.py",
    "main_runner_ledger": "src/memcontam/readiness/phase13_main_runner_ledger.py",
    "main_runner_models": "src/memcontam/readiness/phase13_main_runner_models.py",
    "main_runner_store": "src/memcontam/readiness/phase13_main_runner_store.py",
    "main_runner_cli": "src/memcontam/readiness/phase13_main_runner_cli.py",
    "main_production": "src/memcontam/readiness/phase13_main_production.py",
    "main_production_backend": (
        "src/memcontam/readiness/phase13_main_production_backend.py"
    ),
    "main_live_contract": "src/memcontam/readiness/phase13_main_live_contract.py",
    "main_live_contract_artifact": "data/phase13/main/main_live_contract_v1.json",
    "main_live_evidence": "src/memcontam/readiness/phase13_main_live_evidence.py",
    "main_live_dispatch": "src/memcontam/readiness/phase13_main_live_dispatch.py",
    "main_live_cli": "src/memcontam/readiness/phase13_main_live_cli.py",
    "main_live_runtime": "src/memcontam/readiness/phase13_main_live_runtime.py",
    "main_live_runtime_support": (
        "src/memcontam/readiness/phase13_main_live_runtime_support.py"
    ),
    "main_execution": "src/memcontam/readiness/phase13_main_execution.py",
    "main_execution_models": "src/memcontam/readiness/phase13_main_execution_models.py",
    "main_execution_bindings": "src/memcontam/readiness/phase13_main_execution_bindings.py",
}
PRODUCTION_ROLES: Final = (
    "openai_client",
    "production_observability_adapter",
    "production_runtime_join",
    "production_runtime_evidence",
    "production_runtime_memory",
    "production_runtime_models",
    "main_production_backend",
    "main_live_evidence",
    "main_live_dispatch",
    "main_live_runtime",
    "main_live_runtime_support",
)
RUNNER_ROLES: Final = (
    "main_runner",
    "main_runner_ledger",
    "main_runner_models",
    "main_runner_store",
    "main_runner_cli",
    "main_production",
    "main_production_backend",
    "main_live_contract",
    "main_live_evidence",
    "main_live_dispatch",
    "main_live_cli",
    "main_live_runtime",
    "main_live_runtime_support",
)


class MainExecutionBindingError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_hash(value: JsonValue) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact_bindings(
    package: MainExecutionFreeze,
    repository_root: Path,
) -> dict[str, Path]:
    authorities = {binding.role: binding.sha256 for binding in package.authority}
    if (
        len(package.authority) != len(EXPECTED_AUTHORITIES)
        or len(authorities) != len(package.authority)
        or authorities != EXPECTED_AUTHORITIES
    ):
        raise MainExecutionBindingError("MAIN_EXECUTION_AUTHORITY_MISMATCH")
    if len(package.artifacts) != len(EXPECTED_ARTIFACT_PATHS):
        raise MainExecutionBindingError("MAIN_EXECUTION_ARTIFACT_SET_INVALID")
    paths: dict[str, Path] = {}
    for binding in package.artifacts:
        expected = EXPECTED_ARTIFACT_PATHS.get(binding.role)
        if binding.role in paths or expected is None:
            raise MainExecutionBindingError("MAIN_EXECUTION_ARTIFACT_SET_INVALID")
        if binding.path != expected:
            raise MainExecutionBindingError("MAIN_EXECUTION_ARTIFACT_PATH_INVALID")
        relative = PurePosixPath(binding.path)
        path = repository_root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(repository_root.resolve()):
            raise MainExecutionBindingError("MAIN_EXECUTION_ARTIFACT_PATH_INVALID")
        if hashlib.sha256(read_regular_nofollow(path)).hexdigest() != binding.sha256:
            raise MainExecutionBindingError("MAIN_EXECUTION_ARTIFACT_HASH_MISMATCH")
        paths[binding.role] = path
    if set(paths) != set(EXPECTED_ARTIFACT_PATHS):
        raise MainExecutionBindingError("MAIN_EXECUTION_ARTIFACT_SET_INVALID")
    return paths


def validate_semantic_joins(package: MainExecutionFreeze, paths: dict[str, Path]) -> None:
    bindings = {binding.role: binding.sha256 for binding in package.artifacts}
    if package.observability.packet_sha256 != bindings["observability_packet"]:
        raise MainExecutionBindingError("MAIN_EXECUTION_OBSERVABILITY_BINDING_INVALID")
    production_hash = canonical_hash([bindings[role] for role in PRODUCTION_ROLES])
    if production_hash != package.observability.production_reconstruction_binding_sha256:
        raise MainExecutionBindingError("MAIN_EXECUTION_OBSERVABILITY_BINDING_INVALID")
    runner_hash = canonical_hash([bindings[role] for role in RUNNER_ROLES])
    if runner_hash != package.execution_control.runner_code_sha256:
        raise MainExecutionBindingError("MAIN_EXECUTION_RUNNER_BINDING_INVALID")
    provider = json.loads(read_regular_nofollow(paths["provider_contract"]))
    stage = json.loads(read_regular_nofollow(paths["stage_envelope_registry"]))
    retry = json.loads(read_regular_nofollow(paths["retry_failure_contract"]))
    capacity = json.loads(read_regular_nofollow(paths["common_capacity"]))
    cost = json.loads(read_regular_nofollow(paths["cost_proof"]))
    selection = json.loads(read_regular_nofollow(paths["package_selection"]))
    if (
        provider["schema_version"] != package.runtime.request_contract_id
        or provider["requested_model_id"] != package.runtime.model
        or stage["registry_id"] != package.runtime.execution_envelope_registry_id
        or retry["contract_id"] != package.runtime.transport_contract_id
        or retry["terminal_failure_contract_id"] != package.runtime.terminal_failure_contract_id
        or retry["maximum_transport_attempts_per_semantic_call"]
        != package.runtime.maximum_transport_attempts
        or retry["retries_after_initial_attempt"] != package.runtime.retries_after_initial_attempt
    ):
        raise MainExecutionBindingError("MAIN_EXECUTION_RUNTIME_BINDING_INVALID")
    if (
        capacity["capacity_law_id"] != package.capacity_law_id
        or capacity["capacity_unit"] != package.capacity_unit
        or capacity["B_mem_tokens"] != package.B_mem_tokens
        or capacity["L_DC_tokens"] != package.L_DC_tokens
    ):
        raise MainExecutionBindingError("MAIN_EXECUTION_CAPACITY_BINDING_INVALID")
    guard = package.cost_guard
    if (
        cost["cost_envelope_id"] != guard.cost_envelope_id
        or cost["cost_envelope_sha256"] != guard.cost_envelope_sha256
        or cost["semantic_calls"] != guard.semantic_calls
        or cost["budget"]["total_budget_ceiling_krw"] != guard.total_budget_ceiling_krw
        or cost["budget"]["core_authorization_gate_krw"] != guard.core_authorization_gate_krw
        or cost["cmax_main_krw"] != guard.cmax_main_krw
        or cost["margin_to_core_gate_krw"] != guard.margin_krw
        or guard.reserve_krw != guard.total_budget_ceiling_krw - guard.core_authorization_gate_krw
    ):
        raise MainExecutionBindingError("MAIN_EXECUTION_COST_BINDING_INVALID")
    selected = selection["selected_current_main"]
    if (
        selected["package_id"] != package.selected_package_id
        or selected["attempted_seed_count_per_task"] != len(package.dispatch.concrete_seed_ids)
        or selected["H_run"] != package.H_run
        or selected["H_primary"] != package.H_primary
        or selected["primary_analysis_window_id"] != package.primary_analysis_window_id
    ):
        raise MainExecutionBindingError("MAIN_EXECUTION_SELECTION_BINDING_INVALID")
