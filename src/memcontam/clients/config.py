from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProviderConfig:
    provider: Literal["replay", "openai_compatible"]
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int | None = None
    max_retries: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    @classmethod
    def from_run_config(cls, config: dict) -> ProviderConfig:
        run = config.get("run", {})
        live_smoke = config.get("live_smoke", {})
        values = config.get("provider_config", {})
        provider = run.get("provider")
        if provider is None:
            provider = "openai_compatible" if live_smoke.get("enabled") else "replay"
        if provider == "replay":
            return cls(provider="replay")
        if provider != "openai_compatible":
            raise ValueError(f"unsupported provider: {provider}")
        return cls(
            provider="openai_compatible",
            base_url=values.get("base_url", values.get("normalized_base_url", live_smoke.get("base_url"))),
            api_key_env=values.get("api_key_env", live_smoke.get("api_key_env", "OPENAI_API_KEY")),
            timeout_seconds=values.get("timeout_seconds"),
            max_retries=values.get("max_retries"),
        )
