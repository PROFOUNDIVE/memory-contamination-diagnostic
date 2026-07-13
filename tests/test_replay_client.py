from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from memcontam.clients.base import LLMResponse
from memcontam.clients.replay import ReplayClient


V0_5_FIXTURE_PATH = Path("data/replay/g0_fh_reflexion_dc_faithful_v1.yaml")


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


def test_replay_client_uses_v0_5_fixture_stage_responses() -> None:
    fixture = yaml.safe_load(V0_5_FIXTURE_PATH.read_text(encoding="utf-8"))
    client = ReplayClient(responses_by_sample=fixture["responses_by_sample"])
    sample_id = "game24_pilot_001"
    stages = fixture["responses_by_sample"][sample_id]

    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "full_history_generate"},
    ).content == stages["full_history_generate"]
    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "reflexion_generate"},
    ).content == stages["reflexion_generate"]
    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "reflexion_reflect"},
    ).content == stages["reflexion_reflect"]
    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "dynamic_cheatsheet_generate"},
    ).content == stages["dynamic_cheatsheet_generate"]
    assert client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "dynamic_cheatsheet_curate"},
    ).content == stages["dynamic_cheatsheet_curate"]


def test_replay_client_respects_reflexion_and_dc_stage_order() -> None:
    fixture = yaml.safe_load(V0_5_FIXTURE_PATH.read_text(encoding="utf-8"))
    client = ReplayClient(responses_by_sample=fixture["responses_by_sample"])
    sample_id = "game24_pilot_001"

    reflexion_generate = client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "reflexion_generate"},
    ).content
    reflexion_reflect = client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "reflexion_reflect"},
    ).content

    dc_generate = client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "dynamic_cheatsheet_generate"},
    ).content
    dc_curate = client.chat(
        [{"role": "user", "content": "prompt"}],
        model="replay",
        config={"sample_id": sample_id, "method_stage": "dynamic_cheatsheet_curate"},
    ).content

    stages = fixture["responses_by_sample"][sample_id]
    assert reflexion_generate == stages["reflexion_generate"]
    assert reflexion_reflect == stages["reflexion_reflect"]
    assert dc_generate == stages["dynamic_cheatsheet_generate"]
    assert dc_curate == stages["dynamic_cheatsheet_curate"]
