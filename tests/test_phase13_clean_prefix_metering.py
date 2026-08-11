from __future__ import annotations

from pathlib import Path

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_clean_prefix import (
    Phase13CalibrationError,
    load_clean_prefix_config,
)
from memcontam.readiness.phase13_clean_prefix_metering import MeteredClient


CONFIG = Path("configs/phase13/clean_prefix_calibration_v1.yaml")


def test_metered_client_records_observed_usage_once_per_successful_response() -> None:
    class _TwoAttemptClient:
        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            del messages, model, config
            return LLMResponse(
                content="final: 24",
                raw={"attempts": 2, "cost_usd": 0.025},
                token_usage={"prompt_tokens": 3, "completion_tokens": 2},
            )

    metered = MeteredClient(_TwoAttemptClient(), load_clean_prefix_config(CONFIG))
    metered.chat([{"role": "user", "content": "solve"}], "model", {"max_output_tokens": 2})

    assert metered.transport_attempts == 2
    assert metered.input_tokens == 3
    assert metered.output_tokens == 2
    assert metered.cost_usd == 0.025


def test_metered_client_rejects_oversized_prompt_before_dispatch() -> None:
    class _CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            del messages, model, config
            self.calls += 1
            return LLMResponse(
                content="final: 24",
                raw={"attempts": 1, "cost_usd": 0.0},
                token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

    client = _CountingClient()
    metered = MeteredClient(client, load_clean_prefix_config(CONFIG))

    with pytest.raises(
        Phase13CalibrationError, match="CALIBRATION_INPUT_TOKEN_CEILING_EXCEEDED"
    ):
        metered.chat(
            [{"role": "user", "content": "x" * 4097}],
            "model",
            {"max_output_tokens": 1},
        )

    assert client.calls == 0


def test_metered_client_uses_utf8_byte_bound_before_dispatch() -> None:
    class _CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            del messages, model, config
            self.calls += 1
            raise AssertionError("dispatch must not occur")

    client = _CountingClient()
    metered = MeteredClient(client, load_clean_prefix_config(CONFIG))

    with pytest.raises(
        Phase13CalibrationError, match="CALIBRATION_INPUT_TOKEN_CEILING_EXCEEDED"
    ):
        metered.chat(
            [{"role": "user", "content": "가" * 1400}],
            "model",
            {"max_output_tokens": 1},
        )

    assert client.calls == 0


def test_metered_client_uses_registered_output_limit_when_stage_omits_one() -> None:
    class _CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            del messages, model, config
            self.calls += 1
            return LLMResponse(
                content="final: 24",
                raw={"attempts": 1, "cost_usd": 0.0},
                token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

    client = _CountingClient()
    metered = MeteredClient(client, load_clean_prefix_config(CONFIG))

    metered.chat([{"role": "user", "content": "solve"}], "model", {})

    assert client.calls == 1


def test_metered_client_allows_low_cost_calls_under_lower_hard_ceiling() -> None:
    class _Client:
        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            del messages, model, config
            return LLMResponse(
                content="final: 24",
                raw={"attempts": 1, "cost_usd": 0.0},
                token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

    config = load_clean_prefix_config(CONFIG)
    config["budget"]["hard_ceiling_microusd"] = 15_000_000
    metered = MeteredClient(_Client(), config)
    for _ in range(metered.maximum_semantic_calls):
        metered.chat(
            [{"role": "user", "content": "solve"}],
            "model",
            {"max_output_tokens": 1},
        )

    assert metered.reserved_max_cost_usd == 0.0


def test_metered_client_retains_failed_attempt_liability_after_success() -> None:
    class _RetryingClient:
        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            del messages, model, config
            return LLMResponse(
                content="final: 24",
                raw={"attempts": 2, "cost_usd": 0.000001},
                token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

    metered = MeteredClient(_RetryingClient(), load_clean_prefix_config(CONFIG))
    metered.chat([{"role": "user", "content": "solve"}], "model", {"max_output_tokens": 1})

    assert metered.reserved_max_cost_microusd == 30_721
