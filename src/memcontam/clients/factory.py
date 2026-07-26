from __future__ import annotations

from dataclasses import replace

from memcontam.clients.base import LLMClient
from memcontam.clients.config import ProviderConfig
from memcontam.clients.openai_compatible import OpenAICompatibleClient
from memcontam.clients.openai_responses import LiveCallNotAuthorized, OpenAIResponsesClient
from memcontam.clients.replay import ReplayClient


def validate_provider_selection(
    config: ProviderConfig, *, stage: str, execution_class: str
) -> None:
    if (stage, execution_class, config.provider) == ("replay", "offline_contract_replay", "replay"):
        return
    if (
        stage in {"pilot", "main"}
        and execution_class == "live"
        and config.provider in {"openai_compatible", "openai_responses"}
    ):
        return
    raise ValueError("unsupported provider configuration")


def build_llm_client(
    config: ProviderConfig,
    *,
    stage: str,
    execution_class: str,
    replay_responses: list[str] | None = None,
    allow_live_calls: bool = False,
    legacy_live_smoke: bool = False,
) -> LLMClient:
    validate_provider_selection(config, stage=stage, execution_class=execution_class)
    if config.provider == "replay":
        return ReplayClient(replay_responses)
    if not config.live_calls_enabled and not legacy_live_smoke:
        raise LiveCallNotAuthorized("live calls require config.live_calls.enabled=true")
    if not allow_live_calls and not legacy_live_smoke:
        raise LiveCallNotAuthorized("live calls require --allow-live-calls")
    if legacy_live_smoke:
        config = replace(config, live_calls_enabled=True)
    if config.provider == "openai_responses":
        return OpenAIResponsesClient(config, allow_live_calls=True)
    return OpenAICompatibleClient(config, allow_live_calls=True)
