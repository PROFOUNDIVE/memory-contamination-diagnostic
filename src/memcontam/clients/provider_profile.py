from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping, TypedDict
from urllib.parse import urlsplit, urlunsplit

from memcontam.clients.base import LLMClient
from memcontam.clients.config import ProviderConfig
from memcontam.clients.replay import ReplayClient

ProviderBindingError = Literal[
    "PROVIDER_MODEL_MISMATCH",
    "PROVIDER_SERVICE_TIER_MISMATCH",
    "ANSWER_OUTPUT_BUDGET_MISMATCH",
    "PROVIDER_CLIENT_MISMATCH",
]


class ProviderProfileError(ValueError):
    pass


class ProviderProfileDict(TypedDict):
    provider: str
    normalized_base_url: str | None
    api_key_env: str | None
    timeout_seconds: int | None
    max_retries: int | None
    service_tier: str
    store: bool
    served_models: list[str]
    model_snapshots: dict[str, str]


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    provider: Literal["replay", "openai_compatible", "openai_responses"]
    normalized_base_url: str | None
    api_key_env: str | None
    timeout_seconds: int | None
    max_retries: int | None
    service_tier: str
    store: bool
    served_models: tuple[str, ...]
    model_snapshots: Mapping[str, str]

    def to_dict(self) -> ProviderProfileDict:
        return {
            "provider": self.provider,
            "normalized_base_url": self.normalized_base_url,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "service_tier": self.service_tier,
            "store": self.store,
            "served_models": list(self.served_models),
            "model_snapshots": dict(self.model_snapshots),
        }


def _normalize_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.hostname:
        raise ProviderProfileError("provider base_url must include a scheme and host")
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme.lower(), f"{host}{port}", parsed.path, "", ""))


def normalize_provider_profile(
    config: ProviderConfig,
    *,
    served_models: list[str],
    model_snapshots: Mapping[str, str],
) -> ProviderProfile:
    return ProviderProfile(
        provider=config.provider,
        normalized_base_url=_normalize_url(config.base_url),
        api_key_env=config.api_key_env,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        service_tier=config.service_tier,
        store=config.store,
        served_models=tuple(sorted(served_models)),
        model_snapshots={model: model_snapshots[model] for model in sorted(model_snapshots)},
    )


def provider_profile_id(profile: ProviderProfile) -> str:
    payload = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_client_binding_error(
    model: str,
    client: LLMClient,
    allow_test_client: bool,
) -> ProviderBindingError | None:
    if model not in {"replay", "gpt-5.6-luna"}:
        return "PROVIDER_MODEL_MISMATCH"
    if allow_test_client:
        return None
    if model == "replay" and not isinstance(client, ReplayClient):
        return "PROVIDER_CLIENT_MISMATCH"
    if model == "gpt-5.6-luna":
        from memcontam.clients.openai_responses import OpenAIResponsesClient

        if not isinstance(client, OpenAIResponsesClient):
            return "PROVIDER_CLIENT_MISMATCH"
    return None


def request_binding_error(
    service_tier: str,
    max_output_tokens: int,
) -> ProviderBindingError | None:
    if service_tier != "default":
        return "PROVIDER_SERVICE_TIER_MISMATCH"
    if max_output_tokens != 4096:
        return "ANSWER_OUTPUT_BUDGET_MISMATCH"
    return None
