from __future__ import annotations

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder


class _FakeClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, str]], str, dict]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        self.calls.append((messages, model, config))
        if not self.responses:
            raise RuntimeError("exhausted fake responses")
        return self.responses.pop(0)


class _ExplodingClient:
    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        raise ConnectionError("boom")


def test_recorder_preserves_stage_order_and_usage() -> None:
    inner = _FakeClient(
        [
            LLMResponse(
                content="first",
                raw={"replay": True},
                token_usage={"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
                latency_ms=42,
            ),
            LLMResponse(
                content="second",
                raw={"replay": True},
                token_usage={"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
                latency_ms=100,
            ),
        ]
    )
    recorder = MethodCallRecorder(inner)

    first = recorder.chat(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-replay",
        config={
            "method_stage": "rag_generate",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 128,
        },
    )
    assert first.content == "first"

    second = recorder.chat(
        messages=[{"role": "user", "content": "world"}],
        model="gpt-replay",
        config={
            "method_stage": "bot_distill",
            "temperature": 0.7,
        },
    )
    assert second.content == "second"

    records = recorder.get_records()
    assert len(records) == 2
    assert records[0].stage == "rag_generate"
    assert records[1].stage == "bot_distill"
    assert records[0].raw_response == "first"
    assert records[1].raw_response == "second"
    assert records[0].messages == [{"role": "user", "content": "hello"}]
    assert records[1].messages == [{"role": "user", "content": "world"}]
    assert records[0].model == "gpt-replay"
    assert records[1].model == "gpt-replay"
    assert records[0].temperature == 0.2
    assert records[0].top_p == 0.9
    assert records[0].max_tokens == 128
    assert records[1].temperature == 0.7
    assert records[1].top_p is None
    assert records[1].max_tokens is None
    assert records[0].latency_ms == 42
    assert records[1].latency_ms == 100
    assert records[0].token_usage == {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
    assert records[1].token_usage == {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
    assert records[0].retry_count == 0
    assert records[1].retry_count == 0
    assert records[0].error_type is None
    assert records[1].error_type is None

    assert inner.calls[0][2].get("method_stage") == "rag_generate"
    assert inner.calls[1][2].get("method_stage") == "bot_distill"


def test_recorder_resets_after_failed_trial() -> None:
    recorder = MethodCallRecorder(_ExplodingClient())

    with pytest.raises(ConnectionError, match="boom"):
        recorder.chat([{"role": "user", "content": "x"}], "m", {"method_stage": "fail"})

    error_records = recorder.get_records()
    assert len(error_records) == 1
    assert error_records[0].stage == "fail"
    assert error_records[0].error_type == "ConnectionError"
    assert error_records[0].raw_response == ""

    recorder.reset()
    assert recorder.get_records() == []

    inner = _FakeClient(
        [
            LLMResponse(
                content="ok",
                raw={},
                token_usage={"total_tokens": 1},
                latency_ms=1,
            )
        ]
    )
    recorder = MethodCallRecorder(inner)
    response = recorder.chat([{"role": "user", "content": "y"}], "m", {})
    assert response.content == "ok"
    assert len(recorder.get_records()) == 1
    assert recorder.get_records()[0].stage == "unknown"
