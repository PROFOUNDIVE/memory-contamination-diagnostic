from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from memcontam.clients import openai_compatible as openai_compatible_module
from memcontam.clients.config import ProviderConfig
from memcontam.clients.openai_compatible import OpenAICompatibleClient
from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder
from memcontam.cli import load_config, run_config


class _FakeUsage:
    def model_dump(self) -> dict[str, int]:
        return {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}


class _FakeResponse:
    def __init__(self) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content="final: 24"))]
        self.usage = _FakeUsage()

    def model_dump(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": "final: 24"}}], "usage": {"total_tokens": 18}}


class _FakeChatCompletions:
    def create(self, **kwargs):  # noqa: ANN003
        self.last_kwargs = kwargs
        return _FakeResponse()


class _FakeOpenAI:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def test_openai_compatible_client_mocked_chat(monkeypatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(openai_compatible_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = OpenAICompatibleClient(
        ProviderConfig(
            provider="openai_compatible",
            base_url="https://example.invalid/v1",
            api_key_env="OPENAI_API_KEY",
            timeout_seconds=30,
            max_retries=2,
        )
    )
    response = client.chat([{"role": "user", "content": "solve"}], model="gpt-4o", config={})

    assert isinstance(response, LLMResponse)
    assert response.content == "final: 24"
    assert response.token_usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
    assert isinstance(response.latency_ms, int)
    assert response.latency_ms >= 0


def test_openai_compatible_client_missing_api_key_raises_runtime_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="missing API key env var"):
        OpenAICompatibleClient(ProviderConfig(provider="openai_compatible", api_key_env="OPENAI_API_KEY"))


def test_openai_compatible_client_uses_custom_api_key_env(monkeypatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(openai_compatible_module, "OpenAI", _FakeOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "custom-key")

    client = OpenAICompatibleClient(
        ProviderConfig(
            provider="openai_compatible",
            base_url="https://example.invalid/v1",
            api_key_env="OPENAI_COMPATIBLE_API_KEY",
        )
    )

    assert client.client.api_key == "custom-key"
    assert client.client.base_url == "https://example.invalid/v1"
    assert client.client.timeout is None
    assert client.client.max_retries is None


def test_replay_config_runs_without_provider_env_vars(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    config = load_config(repo_root / "configs/pilot_multitask_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")
    config["baselines"] = [
        baseline
        for baseline in config["baselines"]
        if baseline not in {"retrieval_rag", "bot_style"}
    ]

    run_dir = run_config(config, run_id="task_T12_full_regression")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert rows
    assert {row["task_name"] for row in rows} == {"game24", "math_equation_balancer", "word_sorting"}
    assert {row["baseline"] for row in rows} == {"no_memory", "full_history", "reflexion_style"}


def test_recorder_does_not_persist_secrets_or_raw_payload(monkeypatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(openai_compatible_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "live-secret-key")

    inner = OpenAICompatibleClient(
        ProviderConfig(
            provider="openai_compatible",
            base_url="https://example.invalid/v1",
            api_key_env="OPENAI_API_KEY",
        )
    )
    events = []
    recorder = MethodCallRecorder(
        inner,
        event_callback=events.append,
        trial_context={
            "run_metadata_id": "run-meta-1",
            "run_id": "run-1",
            "trial_id": "trial-1",
            "trial_seq": 0,
            "stage": "main",
        },
    )

    response = recorder.chat(
        messages=[{"role": "user", "content": "solve"}],
        model="gpt-4o",
        config={
            "method_stage": "rag_generate",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 128,
            "retry_count": 1,
            "api_key": "should-not-be-logged",
            "authorization": "Bearer should-not-be-logged",
            "headers": {"X-Secret": "should-not-be-logged"},
        },
    )

    assert response.content == "final: 24"
    assert len(events) == 1
    event = events[0]
    assert event.call_id == "trial-1:call:1"
    assert event.model == "gpt-4o"
    assert event.decoding_params == {"temperature": 0.2, "top_p": 0.9, "max_tokens": 128}
    assert event.retry_count == 1
    assert event.response_text == "final: 24"
    assert event.token_usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}

    dumped = json.dumps(event.model_dump(mode="json"))
    assert "live-secret-key" not in dumped
    assert "should-not-be-logged" not in dumped
    assert "api_key" not in dumped
    assert "authorization" not in dumped
    assert "headers" not in dumped
    assert '"raw":' not in dumped

    record = recorder.get_records()[0]
    record_dumped = json.dumps(record.model_dump(mode="json"))
    assert "live-secret-key" not in record_dumped
    assert "should-not-be-logged" not in record_dumped
    assert "api_key" not in record_dumped
    assert "authorization" not in record_dumped
    assert "headers" not in record_dumped
    assert '"raw":' not in record_dumped
