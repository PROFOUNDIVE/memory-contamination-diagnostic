from __future__ import annotations

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.clients.replay import ReplayClient


def test_replay_client_consumes_named_stages_in_order() -> None:
    client = ReplayClient(
        responses_by_sample={
            "sample_1": {
                "rag_generate": "rag response",
                "bot_problem_distill": "distill response",
                "bot_instantiate_solve": "solve response",
                "bot_thought_distill": "thought response",
                "bot_novelty_decide": ["yes", "no", "yes"],
            }
        }
    )

    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={"sample_id": "sample_1", "method_stage": "rag_generate"},
    ) == LLMResponse(
        content="rag response",
        raw={"replay": True, "messages": [{"role": "user", "content": "prompt"}]},
        token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        latency_ms=0,
    )

    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={"sample_id": "sample_1", "method_stage": "bot_problem_distill"},
    ).content == "distill response"

    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={"sample_id": "sample_1", "method_stage": "bot_instantiate_solve"},
    ).content == "solve response"

    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={"sample_id": "sample_1", "method_stage": "bot_thought_distill"},
    ).content == "thought response"

    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={"sample_id": "sample_1", "method_stage": "bot_novelty_decide"},
    ).content == "yes"
    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={"sample_id": "sample_1", "method_stage": "bot_novelty_decide"},
    ).content == "no"
    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={"sample_id": "sample_1", "method_stage": "bot_novelty_decide"},
    ).content == "yes"


def test_replay_client_rejects_missing_required_stage() -> None:
    client = ReplayClient(
        responses_by_sample={
            "sample_1": {
                "rag_generate": "rag response",
            }
        }
    )

    with pytest.raises(ValueError, match="sample_1.*bot_problem_distill"):
        client.chat(
            [{"role": "user", "content": "prompt"}],
            model="gpt-4o",
            config={"sample_id": "sample_1", "method_stage": "bot_problem_distill"},
        )


def test_replay_client_flat_list_fallback_without_stage() -> None:
    client = ReplayClient(responses=["first", "second", "third"])

    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={},
    ).content == "first"
    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={},
    ).content == "second"
    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="gpt-4o",
        config={},
    ).content == "third"
