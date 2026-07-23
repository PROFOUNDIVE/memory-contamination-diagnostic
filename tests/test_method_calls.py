from __future__ import annotations

import json
from typing import Any

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder, summarize_calls
from memcontam.logging.schema import CallEvent, PromptSourceSpan


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
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or ConnectionError("boom")

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        raise self.error


def _trial_context(trial_id: str = "trial-1", trial_seq: int = 0) -> dict[str, Any]:
    return {
        "run_metadata_id": "run-meta-1",
        "run_id": "run-1",
        "trial_id": trial_id,
        "trial_seq": trial_seq,
        "stage": "replay",
    }


def _span(message_index: int = 0) -> PromptSourceSpan:
    return PromptSourceSpan(
        message_index=message_index,
        start=0,
        end=5,
        rendered_hash="sha256:hello",
        entry_id="entry-1",
        source_ids=["entry-1"],
        parent_ids=[],
        lineage_id="lineage-1",
        version="v1",
        origin="retrieval",
        clean_or_contaminated="clean",
    )


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
    assert records[0].token_usage == {
        "prompt_tokens": 3,
        "completion_tokens": 7,
        "total_tokens": 10,
    }
    assert records[1].token_usage == {
        "prompt_tokens": 5,
        "completion_tokens": 9,
        "total_tokens": 14,
    }
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
    assert error_records[0].raw_response is None
    assert error_records[0].call_id == "unknown:call:1"

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


def test_recorder_emits_callback_before_return_with_call_id() -> None:
    events: list[CallEvent] = []
    inner = _FakeClient(
        [
            LLMResponse(
                content="ok",
                raw={"secret": "should-not-appear"},
                token_usage={"total_tokens": 5},
                latency_ms=10,
            )
        ]
    )
    recorder = MethodCallRecorder(
        inner,
        event_callback=events.append,
        trial_context=_trial_context(),
    )

    recorder.chat(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-replay",
        config={
            "method_stage": "rag_generate",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 128,
            "retry_count": 2,
        },
    )

    assert len(events) == 1
    event = events[0]
    assert event.call_id == "trial-1:call:1"
    assert event.run_metadata_id == "run-meta-1"
    assert event.run_id == "run-1"
    assert event.trial_id == "trial-1"
    assert event.trial_seq == 0
    assert event.stage == "replay"
    assert event.messages == [{"role": "user", "content": "hello"}]
    assert event.model == "gpt-replay"
    assert event.decoding_params == {"temperature": 0.2, "top_p": 0.9, "max_tokens": 128}
    assert event.response_text == "ok"
    assert event.token_usage == {"total_tokens": 5}
    assert event.latency_ms == 10
    assert event.retry_count == 2
    assert event.source_spans == []
    assert "raw" not in event.model_dump(mode="json")

    records = recorder.get_records()
    assert len(records) == 1
    assert records[0].call_id == event.call_id


def test_recorder_increments_call_index_per_trial_context() -> None:
    events: list[CallEvent] = []
    inner = _FakeClient(
        [
            LLMResponse(content="a", raw={}, token_usage={"total_tokens": 1}, latency_ms=1),
            LLMResponse(content="b", raw={}, token_usage={"total_tokens": 1}, latency_ms=1),
        ]
    )
    recorder = MethodCallRecorder(
        inner,
        event_callback=events.append,
        trial_context=_trial_context(),
    )

    recorder.chat([{"role": "user", "content": "a"}], "m", {"method_stage": "s1"})
    recorder.chat([{"role": "user", "content": "b"}], "m", {"method_stage": "s2"})

    assert [event.call_id for event in events] == ["trial-1:call:1", "trial-1:call:2"]
    assert [record.call_id for record in recorder.get_records()] == [
        "trial-1:call:1",
        "trial-1:call:2",
    ]


def test_recorder_carries_source_spans_from_config() -> None:
    events: list[CallEvent] = []
    span = _span()
    inner = _FakeClient(
        [LLMResponse(content="ok", raw={}, token_usage={"total_tokens": 1}, latency_ms=1)]
    )
    recorder = MethodCallRecorder(
        inner,
        event_callback=events.append,
        trial_context=_trial_context(),
    )

    recorder.chat(
        [{"role": "user", "content": "hello"}],
        "m",
        {"method_stage": "rag_generate", "source_spans": [span.model_dump(mode="json")]},
    )

    assert len(events) == 1
    assert len(events[0].source_spans) == 1
    assert events[0].source_spans[0].entry_id == "entry-1"
    record = recorder.get_records()[0]
    assert len(record.source_spans) == 1
    assert record.source_spans[0].entry_id == "entry-1"


def test_recorder_emits_failed_call_before_rethrow_with_no_secrets() -> None:
    events: list[CallEvent] = []
    inner = _ExplodingClient(ConnectionError("secret-token-12345"))
    recorder = MethodCallRecorder(
        inner,
        event_callback=events.append,
        trial_context=_trial_context(),
    )

    config = {
        "method_stage": "rag_generate",
        "api_key": "super-secret",
        "authorization": "Bearer token",
        "headers": {"X-Custom": "secret"},
        "retry_count": 3,
    }
    with pytest.raises(ConnectionError, match="secret-token-12345"):
        recorder.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-replay",
            config=config,
        )

    assert len(events) == 1
    event = events[0]
    assert event.call_id == "trial-1:call:1"
    assert event.response_text is None
    assert event.retry_count == 3
    assert event.error_type == "ConnectionError"
    assert event.failure_function == "chat"
    assert event.failure_module == "memcontam.clients.recording"
    assert event.failure_line is not None
    assert event.failure_line > 0

    dumped = json.dumps(event.model_dump(mode="json"))
    assert "secret-token" not in dumped
    assert "super-secret" not in dumped
    assert "Bearer" not in dumped
    assert "X-Custom" not in dumped
    assert "raw" not in dumped
    assert "ConnectionError" in dumped

    record = recorder.get_records()[0]
    assert record.call_id == event.call_id
    assert record.error_type == "ConnectionError"


def test_summarize_calls_sums_latency_tokens_and_max_retry() -> None:
    base = {
        "run_metadata_id": "run-meta-1",
        "run_id": "run-1",
        "trial_id": "trial-1",
        "trial_seq": 0,
        "event_seq": 0,
        "stage": "replay",
        "messages": [{"role": "user", "content": "hi"}],
        "model": "m",
        "decoding_params": {},
        "response_text": "ok",
        "created_at": "2026-07-16T00:00:00Z",
    }
    calls = [
        CallEvent(
            call_id="trial-1:call:1",
            token_usage={"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
            latency_ms=42,
            retry_count=1,
            source_spans=[],
            **base,
        ),
        CallEvent(
            call_id="trial-1:call:2",
            token_usage={"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
            latency_ms=100,
            retry_count=3,
            source_spans=[],
            **base,
        ),
    ]

    summary = summarize_calls(calls)

    assert summary == {
        "latency_ms": 142,
        "token_usage": {
            "prompt_tokens": 8,
            "completion_tokens": 16,
            "total_tokens": 24,
        },
        "retry_count": 3,
    }
