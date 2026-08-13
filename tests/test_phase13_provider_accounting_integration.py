from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.readiness import phase13_cli
from memcontam.readiness.phase13_provider_accounting import OwnedProviderAccounting
from memcontam.readiness.phase13_provider_models import ExecutionTemplateIdentity


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

    assert isinstance(client, OwnedProviderAccounting)


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
