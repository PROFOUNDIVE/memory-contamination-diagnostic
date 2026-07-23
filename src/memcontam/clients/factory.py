from __future__ import annotations

from memcontam.clients.base import LLMClient
from memcontam.clients.config import ProviderConfig
from memcontam.clients.openai_compatible import OpenAICompatibleClient
from memcontam.clients.replay import ReplayClient


def validate_provider_selection(
    config: ProviderConfig, *, stage: str, execution_class: str
) -> None:
    if (stage, execution_class, config.provider) == ("replay", "offline_contract_replay", "replay"):
        return
    if (
        stage in {"pilot", "main"}
        and execution_class == "live"
        and config.provider == "openai_compatible"
    ):
        return
    raise ValueError("unsupported provider configuration")


def build_llm_client(
    config: ProviderConfig,
    *,
    stage: str,
    execution_class: str,
    replay_responses: list[str] | None = None,
) -> LLMClient:
    validate_provider_selection(config, stage=stage, execution_class=execution_class)
    if config.provider == "replay":
        return ReplayClient(replay_responses)
    return OpenAICompatibleClient(config)
