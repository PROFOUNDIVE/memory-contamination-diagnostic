from __future__ import annotations

import pytest

from memcontam.clients.config import ProviderConfig
from memcontam.clients.factory import build_llm_client
from memcontam.clients.replay import ReplayClient


def test_provider_factory_dispatches_replay_only_for_offline_contract_replay() -> None:
    client = build_llm_client(
        ProviderConfig(provider="replay"),
        stage="replay",
        execution_class="offline_contract_replay",
        replay_responses=["final: 24"],
    )

    assert isinstance(client, ReplayClient)


@pytest.mark.parametrize(
    ("provider", "stage", "execution_class"),
    [
        ("openai_compatible", "replay", "offline_contract_replay"),
        ("replay", "pilot", "live"),
        ("replay", "main", "live"),
        ("openai_compatible", "pilot", "offline_contract_replay"),
    ],
)
def test_provider_factory_rejects_invalid_stage_execution_provider_combinations(
    provider: str, stage: str, execution_class: str
) -> None:
    with pytest.raises(ValueError, match="unsupported provider configuration"):
        build_llm_client(
            ProviderConfig(provider=provider),  # type: ignore[arg-type]
            stage=stage,
            execution_class=execution_class,
        )
