from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import openai

from memcontam.clients import openai_compatible as openai_compatible_module
from memcontam.clients.config import ProviderConfig
from memcontam.clients.openai_compatible import OpenAICompatibleClient
from memcontam.clients.provider_profile import normalize_provider_profile, provider_profile_id
from memcontam.cli import load_config
from memcontam.config.resolution import resolve_run_config


class _Usage:
    def model_dump(self) -> dict[str, int]:
        return {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}


class _Response:
    choices = [SimpleNamespace(message=SimpleNamespace(content="final: 24"))]
    usage = _Usage()

    def model_dump(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": "final: 24"}}]}


class _Completions:
    last_kwargs: dict[str, object] = {}

    def create(self, **kwargs):  # noqa: ANN003
        _Completions.last_kwargs = kwargs
        return _Response()


class _OpenAITransport:
    def __init__(self, **_kwargs) -> None:  # noqa: ANN003
        self.chat = SimpleNamespace(completions=_Completions())


def test_f1c_mocked_live_dispatch_joins_provider_profile_and_resolved_config(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "baseline_fidelity_v2_bge_smoke.yaml")
    provider_config = ProviderConfig.from_run_config(config)
    profile = normalize_provider_profile(
        provider_config,
        served_models=config["models"],
        model_snapshots=config["run"]["model_snapshots"],
    )
    resolved = resolve_run_config(config, provider_profile=profile)

    monkeypatch.setattr(openai, "OpenAI", _OpenAITransport)
    monkeypatch.setattr(openai_compatible_module, "OpenAI", _OpenAITransport)
    monkeypatch.setenv("F1C_MOCKED_LIVE_API_KEY", "mocked-transport-only")
    client = OpenAICompatibleClient(provider_config)
    response = client.chat(
        [{"role": "user", "content": "solve"}], config["models"][0], {"max_tokens": 32}
    )

    resolved_hash = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert response.content == "final: 24"
    assert _Completions.last_kwargs["model"] == config["models"][0]
    assert resolved["run"]["provider_profile_id"] == provider_profile_id(profile)
    assert resolved["run"]["provider_profile_id"] == provider_profile_id(
        normalize_provider_profile(
            ProviderConfig.from_run_config(resolved),
            served_models=resolved["models"],
            model_snapshots=resolved["run"]["model_snapshots"],
        )
    )
    assert resolved_hash
    assert "mocked-transport-only" not in json.dumps(resolved, sort_keys=True)
