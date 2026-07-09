from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from memcontam.clients import openai_compatible as openai_compatible_module
from memcontam.clients.openai_compatible import OpenAICompatibleClient
from memcontam.clients.base import LLMResponse
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
    def __init__(self, *, api_key: str, base_url: str | None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def test_openai_compatible_client_mocked_chat(monkeypatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(openai_compatible_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = OpenAICompatibleClient(base_url="https://example.invalid/v1", api_key_env="OPENAI_API_KEY")
    response = client.chat([{"role": "user", "content": "solve"}], model="gpt-4o", config={})

    assert isinstance(response, LLMResponse)
    assert response.content == "final: 24"
    assert response.token_usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
    assert isinstance(response.latency_ms, int)
    assert response.latency_ms >= 0


def test_openai_compatible_client_missing_api_key_raises_runtime_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="missing API key env var"):
        OpenAICompatibleClient(base_url=None, api_key_env="OPENAI_API_KEY")


def test_openai_compatible_client_uses_custom_api_key_env(monkeypatch) -> None:
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(openai_compatible_module, "OpenAI", _FakeOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "custom-key")

    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key_env="OPENAI_COMPATIBLE_API_KEY",
    )

    assert client.client.api_key == "custom-key"
    assert client.client.base_url == "https://example.invalid/v1"


def test_replay_config_runs_without_provider_env_vars(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    config = load_config(repo_root / "configs/pilot_multitask_replay.yaml")
    config["logging"]["output_dir"] = str(tmp_path / "runs")

    run_dir = run_config(config, run_id="task_T12_full_regression")
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]

    assert rows
    assert {row["task_name"] for row in rows} == {"game24", "math_equation_balancer", "word_sorting"}
