from __future__ import annotations

import copy
from typing import Any

from memcontam.baselines.contracts import (
    BASELINE_EXECUTION_CONTRACT_V2,
    BASELINE_FIDELITY_V2,
    FAILURE_TAXONOMY_V2,
)
from memcontam.clients.provider_profile import ProviderProfile, provider_profile_id


_SECRET_MARKERS = ("api_key", "authorization", "credential", "password", "secret", "token")
_FIDELITY_GATE_LAYERS = {"structural", "source_contract", "real_retriever"}


def validate_fidelity_contract(config: dict[str, Any]) -> bool:
    run = config.get("run", {})
    logging = config.get("logging", {})
    versions = {
        "logging.memory_policy_version": logging.get("memory_policy_version"),
        "logging.prompt_version": logging.get("prompt_version"),
        "run.retry_policy_version": run.get("retry_policy_version"),
        "run.baseline_execution_contract_version": run.get("baseline_execution_contract_version"),
        "run.failure_taxonomy_version": run.get("failure_taxonomy_version"),
    }
    is_v2 = any(value == BASELINE_FIDELITY_V2 for value in versions.values())
    if not is_v2:
        return False
    if any(value != BASELINE_FIDELITY_V2 for value in versions.values()):
        raise ValueError("complete Baseline-Fidelity-V2 version tuple is required")
    if run.get("fidelity_gate_layer") not in _FIDELITY_GATE_LAYERS:
        raise ValueError(
            "run.fidelity_gate_layer must be structural, source_contract, or real_retriever"
        )
    return True


def _redact(value: Any, key: str = "") -> Any:
    if any(marker in key.lower() for marker in _SECRET_MARKERS) and key.lower() != "api_key_env":
        return "[REDACTED]"
    if isinstance(value, dict):
        return {item_key: _redact(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def resolve_run_config(
    config: dict[str, Any], *, provider_profile: ProviderProfile
) -> dict[str, Any]:
    resolved = _redact(copy.deepcopy(config))
    run = resolved.setdefault("run", {})
    legacy_live_smoke = resolved.get("live_smoke", {}).get("enabled", False)
    run.setdefault("stage", "pilot" if legacy_live_smoke else "replay")
    run.setdefault("execution_class", "live" if legacy_live_smoke else "offline_contract_replay")
    run["provider"] = provider_profile.provider
    run.setdefault("scientific_result", False)
    run.setdefault("scientific_gate_id", None)
    run["provider_profile_id"] = provider_profile_id(provider_profile)
    if validate_fidelity_contract(resolved):
        run["baseline_execution_contract_version"] = BASELINE_EXECUTION_CONTRACT_V2
        run["failure_taxonomy_version"] = FAILURE_TAXONOMY_V2
    else:
        run["failure_taxonomy_version"] = "baseline_fidelity_v1"
    resolved["provider_config"] = provider_profile.to_dict()
    return resolved
