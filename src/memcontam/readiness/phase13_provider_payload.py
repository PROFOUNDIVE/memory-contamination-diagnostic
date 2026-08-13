from __future__ import annotations

from typing import Final

from memcontam.readiness.phase13_provider_models import JsonScalar, ProviderAccountingError

ALLOWED_CONFIG: Final = frozenset({
    "temperature", "max_output_tokens", "top_p", "seed", "service_tier", "store",
    "arm", "baseline", "model", "run_id", "sample_id", "tool_mode", "fh_mode",
    "context_budget_id", "context_window_tokens", "execution_owner_id",
    "execution_template_id", "session_id", "intervention_id",
})
INTERNAL_CONFIG: Final = frozenset({"embedding_provider", "_phase12_reflection_hook"})


def provider_config(config: dict) -> dict[str, JsonScalar]:
    unknown = set(config) - ALLOWED_CONFIG - INTERNAL_CONFIG
    if unknown or any(_forbidden_token(key) for key in config):
        raise ProviderAccountingError("PROVIDER_CONFIG_LEAKAGE")
    payload: dict[str, JsonScalar] = {}
    for key in ALLOWED_CONFIG & set(config):
        value = config[key]
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise ProviderAccountingError("PROVIDER_CONFIG_LEAKAGE")
        payload[key] = value
    return payload


def validate_messages(messages: list[dict[str, str]]) -> None:
    if any(
        set(message) != {"role", "content"}
        or any(_forbidden_token(value) for value in message.values())
        for message in messages
    ):
        raise ProviderAccountingError("PROVIDER_PROMPT_LEAKAGE")


def _forbidden_token(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    return any(
        token in normalized
        for token in ("future", "horizon", "analysis_window", "task", "window")
    )
