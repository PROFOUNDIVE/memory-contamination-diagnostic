from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.readiness import phase13_cli
from memcontam.readiness.phase13_provider_models import ExecutionTemplateIdentity
from memcontam.readiness.phase13_provider_runtime import (
    Phase13V2ProviderRuntime,
    Phase13V2RuntimeError,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_live_dispatch_factory_returns_mandatory_owned_client() -> None:
    class _Client:
        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            raise AssertionError("not dispatched")

    client = phase13_cli.build_calibration_v2_provider(
        _Client(),
        ROOT,
        ExecutionTemplateIdentity(task="game24", baseline="bot_style", arm_key="Contam"),
    )

    assert isinstance(client, Phase13V2ProviderRuntime)


def test_blocked_v2_cli_never_constructs_owned_or_provider_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions = 0

    def forbidden() -> None:
        nonlocal constructions
        constructions += 1

    monkeypatch.setattr(phase13_cli, "build_calibration_v2_provider", forbidden)

    with pytest.raises(SystemExit, match="CALIBRATION_V2_EXTERNAL_BLOCK"):
        phase13_cli.run(
            argparse.Namespace(
                phase13_command="run-calibration-v2",
                config=Path("configs/phase13/pre_main_calibration_v2.yaml"),
            )
        )

    assert constructions == 0


def test_v2_runtime_constructs_owned_client_before_actual_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Client:
        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            del messages, model
            events.append("provider")
            assert config["execution_owner_id"] == "phase13-h10-execution-owner-v1"
            return LLMResponse(
                "final: 24",
                {"attempts": 1, "cost_usd": 0.0, "storage_bytes": 0},
                {"prompt_tokens": 1, "completion_tokens": 1},
                1,
            )

    original = phase13_cli.build_calibration_v2_provider

    def observed_factory(client, root, identity):  # noqa: ANN001, ANN202
        events.append("factory")
        return original(client, root, identity)

    monkeypatch.setattr(phase13_cli, "build_calibration_v2_provider", observed_factory)
    runtime = phase13_cli.build_calibration_v2_provider(
        _Client(),
        ROOT,
        ExecutionTemplateIdentity(task="game24", baseline="bot_style", arm_key="Contam"),
    )

    runtime.chat([{"role": "user", "content": "solve"}], "model", {})

    assert events == ["factory", "provider"]
    assert isinstance(runtime, Phase13V2ProviderRuntime)
    assert runtime.reconcile().totals.semantic_calls == 1


def test_v2_runtime_rejects_direct_raw_client_injection() -> None:
    class _Client:
        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            raise AssertionError("must not dispatch")

    with pytest.raises(Phase13V2RuntimeError, match="OWNED_PROVIDER_REQUIRED"):
        Phase13V2ProviderRuntime(_Client())
