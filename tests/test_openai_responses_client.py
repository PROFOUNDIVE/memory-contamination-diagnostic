from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memcontam.clients.config import ProviderConfig
from memcontam.clients.cost_guard import CostGuard, CostLimitExceeded, MissingUsageError
from memcontam.clients import openai_responses as responses_module
from memcontam.clients.openai_responses import OpenAIResponsesClient


class _Usage:
    def __init__(self) -> None:
        self.input_tokens = 7
        self.input_tokens_details = SimpleNamespace(cached_tokens=2)
        self.output_tokens = 11
        self.total_tokens = 18

    def model_dump(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "input_tokens_details": {"cached_tokens": self.input_tokens_details.cached_tokens},
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class _Response:
    id = "resp_123"
    model = "gpt-4o-2024-11-20"
    output_text = "final: 24"
    usage = _Usage()
    service_tier = "default"


class _Responses:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _OpenAI:
    outcomes: list[object] = []
    instance: "_OpenAI"

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs
        self.responses = _Responses(list(self.outcomes))
        type(self).instance = self


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _client(monkeypatch, outcomes: list[object], **config_values: Any) -> OpenAIResponsesClient:
    _OpenAI.outcomes = outcomes
    monkeypatch.setattr(responses_module, "OpenAI", _OpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ProviderConfig(
        provider="openai_responses",
        live_calls_enabled=True,
        timeout_seconds=30,
        **config_values,
    )
    return OpenAIResponsesClient(config, allow_live_calls=True, sleep=lambda _: None)


def test_responses_client_preserves_messages_and_records_safe_response_metadata(monkeypatch) -> None:
    client = _client(monkeypatch, [_Response()])
    messages = [
        {"role": "developer", "content": "Follow the task exactly."},
        {"role": "user", "content": "solve"},
    ]

    response = client.chat(
        messages,
        model="gpt-4o-2024-11-20",
        config={"temperature": 0.2, "top_p": 0.9, "max_tokens": 128, "requested_seed": 0},
    )

    assert response.content == "final: 24"
    assert response.token_usage == {
        "prompt_tokens": 7,
        "cached_prompt_tokens": 2,
        "completion_tokens": 11,
        "total_tokens": 18,
    }
    assert response.raw["response_id"] == "resp_123"
    assert response.raw["model"] == "gpt-4o-2024-11-20"
    assert response.raw["attempts"] == 1
    assert response.raw["service_tier"] == "default"
    assert response.raw["seed_parameter_sent"] is False
    assert "test-key" not in repr(response.raw)
    assert _OpenAI.instance.kwargs["max_retries"] == 0
    assert _OpenAI.instance.kwargs["timeout"] == 30
    assert _OpenAI.instance.responses.calls == [
        {
            "model": "gpt-4o-2024-11-20",
            "input": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 128,
            "service_tier": "default",
            "store": False,
        }
    ]


@pytest.mark.parametrize("error", [TimeoutError(), _StatusError(429), _StatusError(503)])
def test_responses_client_retries_only_retryable_failures(monkeypatch, error: BaseException) -> None:
    client = _client(monkeypatch, [error, _Response()])

    response = client.chat(
        [{"role": "user", "content": "solve"}], "gpt-4o-2024-11-20", {"max_tokens": 16}
    )

    assert response.content == "final: 24"
    assert response.raw["attempts"] == 2
    assert len(_OpenAI.instance.responses.calls) == 2


def test_responses_client_does_not_retry_immediate_client_error(monkeypatch) -> None:
    client = _client(monkeypatch, [_StatusError(400)])

    with pytest.raises(_StatusError):
        client.chat([{"role": "user", "content": "solve"}], "gpt-4o-2024-11-20", {})

    assert len(_OpenAI.instance.responses.calls) == 1


def test_responses_client_retains_attempt_count_on_terminal_retry_failure(monkeypatch) -> None:
    error = _StatusError(503)
    client = _client(
        monkeypatch,
        [error, error, error],
        retries_after_initial_attempt=2,
        retry_delays_seconds=(0.0, 0.0),
    )

    with pytest.raises(_StatusError) as caught:
        client.chat([{"role": "user", "content": "solve"}], "future-model", {})

    assert caught.value.provider_attempts_count == 3
    assert isinstance(caught.value.provider_latency_ms, int)


def test_responses_client_fails_closed_when_provider_omits_usage(monkeypatch) -> None:
    response = _Response()
    setattr(response, "usage", None)
    client = _client(monkeypatch, [response])

    with pytest.raises(MissingUsageError, match="usage"):
        client.chat([{"role": "user", "content": "solve"}], "gpt-4o-2024-11-20", {})


def test_responses_client_rejects_over_budget_before_dispatch(monkeypatch) -> None:
    ledger = CostGuard(hard_ceiling_usd=0.00001)
    client = _client(monkeypatch, [_Response()])
    client.cost_guard = ledger

    with pytest.raises(CostLimitExceeded, match="USD"):
        client.chat([{"role": "user", "content": "solve"}], "gpt-4o-2024-11-20", {"max_tokens": 2})

    assert _OpenAI.instance.responses.calls == []


def test_responses_client_loads_repository_dotenv_without_overriding_exported_key(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
    _OpenAI.outcomes = []
    monkeypatch.setattr(responses_module, "OpenAI", _OpenAI)
    monkeypatch.setattr(responses_module, "_repository_root", lambda: tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "exported-key")

    OpenAIResponsesClient(
        ProviderConfig(provider="openai_responses", live_calls_enabled=True),
        allow_live_calls=True,
    )

    assert _OpenAI.instance.kwargs["api_key"] == "exported-key"
