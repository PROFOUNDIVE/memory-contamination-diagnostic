from __future__ import annotations

import pytest

from memcontam.clients.config import ProviderConfig
from memcontam.clients.factory import build_llm_client
from memcontam.clients.openai_responses import LiveCallNotAuthorized
from memcontam.clients.replay import ReplayClient


def test_responses_factory_rejects_disabled_live_config_before_client_construction(monkeypatch) -> None:
    def unexpected_client(*_args, **_kwargs):
        raise AssertionError("live client should not be constructed")

    monkeypatch.setattr("memcontam.clients.factory.OpenAIResponsesClient", unexpected_client)

    with pytest.raises(LiveCallNotAuthorized, match="live_calls.enabled"):
        build_llm_client(
            ProviderConfig(provider="openai_responses", live_calls_enabled=False),
            stage="pilot",
            execution_class="live",
            allow_live_calls=True,
        )


def test_responses_factory_rejects_missing_caller_authorization_before_client_construction(
    monkeypatch,
) -> None:
    def unexpected_client(*_args, **_kwargs):
        raise AssertionError("live client should not be constructed")

    monkeypatch.setattr("memcontam.clients.factory.OpenAIResponsesClient", unexpected_client)

    with pytest.raises(LiveCallNotAuthorized, match="allow-live-calls"):
        build_llm_client(
            ProviderConfig(provider="openai_responses", live_calls_enabled=True),
            stage="pilot",
            execution_class="live",
        )


def test_replay_factory_behavior_is_unchanged_without_live_authorization() -> None:
    client = build_llm_client(
        ProviderConfig(provider="replay"),
        stage="replay",
        execution_class="offline_contract_replay",
        replay_responses=["final: 24"],
    )

    assert isinstance(client, ReplayClient)
    assert client.chat([], "replay", {}).content == "final: 24"
