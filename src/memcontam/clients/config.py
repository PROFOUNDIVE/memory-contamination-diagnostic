from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProviderConfig:
    provider: Literal["replay", "openai_compatible", "openai_responses"]
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: int | None = None
    max_retries: int | None = None
    live_calls_enabled: bool = False
    service_tier: str = "default"
    store: bool = False
    max_output_tokens: int = 2048
    retries_after_initial_attempt: int = 3
    retry_delays_seconds: tuple[float, ...] = (1, 2, 4)
    input_per_million_usd: float = 2.50
    cached_input_per_million_usd: float = 1.25
    output_per_million_usd: float = 10.00

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.max_output_tokens < 0:
            raise ValueError("max_output_tokens must be non-negative")
        if self.retries_after_initial_attempt < 0:
            raise ValueError("retries_after_initial_attempt must be non-negative")
        if len(self.retry_delays_seconds) < self.retries_after_initial_attempt:
            raise ValueError("retry_delays_seconds must cover every retry")
        if any(delay < 0 for delay in self.retry_delays_seconds):
            raise ValueError("retry delays must be non-negative")

    @classmethod
    def from_run_config(cls, config: dict) -> ProviderConfig:
        run = config.get("run", {})
        live_smoke = config.get("live_smoke", {})
        values = config.get("provider_config", {})
        provider_values = config.get("provider", {})
        provider_values = provider_values if isinstance(provider_values, dict) else {}
        live_calls = config.get("live_calls", {})
        retry = config.get("retry", {})
        cost = config.get("cost", {})
        decoding = config.get("decoding", {})
        provider = run.get("provider")
        if provider is None:
            provider = values.get("provider", provider_values.get("provider"))
        endpoint = values.get("endpoint", provider_values.get("endpoint"))
        if provider == "openai":
            provider = "openai_responses" if endpoint == "responses" else "openai_compatible"
        if provider is None:
            provider = "openai_compatible" if live_smoke.get("enabled") else "replay"
        if provider == "openai_compatible" and endpoint == "responses":
            provider = "openai_responses"
        if provider == "replay":
            return cls(provider="replay")
        if provider not in {"openai_compatible", "openai_responses"}:
            raise ValueError(f"unsupported provider: {provider}")
        return cls(
            provider=provider,
            base_url=values.get(
                "base_url", values.get("normalized_base_url", live_smoke.get("base_url"))
            ),
            api_key_env=values.get("api_key_env", live_smoke.get("api_key_env", "OPENAI_API_KEY")),
            timeout_seconds=values.get("timeout_seconds"),
            max_retries=values.get("max_retries"),
            live_calls_enabled=bool(live_calls.get("enabled", False)),
            service_tier=values.get("service_tier", provider_values.get("service_tier", "default")),
            store=bool(values.get("store", provider_values.get("store", False))),
            max_output_tokens=decoding.get("max_output_tokens", values.get("max_output_tokens", 2048)),
            retries_after_initial_attempt=retry.get(
                "retries_after_initial_attempt", values.get("retries_after_initial_attempt", 3)
            ),
            retry_delays_seconds=tuple(
                retry.get("backoff_seconds", values.get("retry_delays_seconds", (1, 2, 4)))
            ),
            input_per_million_usd=cost.get("input_per_1m_tokens", 2.50),
            cached_input_per_million_usd=cost.get("cached_input_per_1m_tokens", 1.25),
            output_per_million_usd=cost.get("output_per_1m_tokens", 10.00),
        )
