from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

from memcontam.clients.config import ProviderConfig


@dataclass(frozen=True)
class ProviderProfile:
    provider: Literal["replay", "openai_compatible"]
    normalized_base_url: str | None
    api_key_env: str | None
    timeout_seconds: int | None
    max_retries: int | None
    served_models: tuple[str, ...]
    model_snapshots: Mapping[str, str]

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "normalized_base_url": self.normalized_base_url,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "served_models": list(self.served_models),
            "model_snapshots": dict(self.model_snapshots),
        }


def _normalize_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("provider base_url must include a scheme and host")
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
        served_models=tuple(sorted(served_models)),
        model_snapshots={model: model_snapshots[model] for model in sorted(model_snapshots)},
    )


def provider_profile_id(profile: ProviderProfile) -> str:
    payload = json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
