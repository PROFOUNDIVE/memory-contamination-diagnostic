from __future__ import annotations

import copy
from typing import Any

from memcontam.clients.provider_profile import ProviderProfile, provider_profile_id


_SECRET_MARKERS = ("api_key", "authorization", "credential", "password", "secret", "token")


def _redact(value: Any, key: str = "") -> Any:
    if any(marker in key.lower() for marker in _SECRET_MARKERS) and key.lower() != "api_key_env":
        return "[REDACTED]"
    if isinstance(value, dict):
        return {item_key: _redact(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def resolve_run_config(config: dict[str, Any], *, provider_profile: ProviderProfile) -> dict[str, Any]:
    resolved = _redact(copy.deepcopy(config))
    run = resolved.setdefault("run", {})
    legacy_live_smoke = resolved.get("live_smoke", {}).get("enabled", False)
    run.setdefault("stage", "pilot" if legacy_live_smoke else "replay")
    run.setdefault("execution_class", "live" if legacy_live_smoke else "offline_contract_replay")
    run["provider"] = provider_profile.provider
    run.setdefault("scientific_result", False)
    run.setdefault("scientific_gate_id", None)
    run["provider_profile_id"] = provider_profile_id(provider_profile)
    run["failure_taxonomy_version"] = "baseline_fidelity_v1"
    resolved["provider_config"] = provider_profile.to_dict()
    return resolved
