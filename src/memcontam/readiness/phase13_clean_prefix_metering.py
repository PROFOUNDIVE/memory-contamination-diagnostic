from __future__ import annotations

from decimal import ROUND_CEILING, Decimal
from typing import Any

from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.readiness.phase13_clean_prefix import Phase13CalibrationError


class MeteredClient:
    def __init__(self, client: LLMClient, config: dict[str, Any]) -> None:
        self.client = client
        budget = config["budget"]
        retry = config["retry"]
        decoding = config["decoding"]
        self.maximum_semantic_calls = budget["maximum_semantic_calls"]
        self.maximum_transport_attempts = budget["maximum_transport_attempts"]
        self.maximum_input_tokens = budget["maximum_input_tokens"]
        self.maximum_output_tokens = budget["maximum_output_tokens"]
        self.maximum_input_tokens_per_attempt = budget["maximum_input_tokens_per_attempt"]
        self.maximum_output_tokens_per_attempt = decoding["max_output_tokens"]
        self.maximum_attempts_per_semantic_call = 1 + retry["retries_after_initial_attempt"]
        self.hard_ceiling_microusd = budget["hard_ceiling_microusd"]
        self.hard_ceiling_usd = self.hard_ceiling_microusd / 1_000_000
        self.maximum_cost_per_transport_attempt_microusd = int(
            self.maximum_input_tokens_per_attempt * budget["input_per_1m_tokens"]
            + self.maximum_output_tokens_per_attempt * budget["output_per_1m_tokens"]
        )
        self.maximum_cost_per_semantic_call_microusd = (
            self.maximum_attempts_per_semantic_call
            * self.maximum_cost_per_transport_attempt_microusd
        )
        self.semantic_calls = 0
        self.semantic_calls_dispatched = 0
        self.transport_attempts = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.reserved_max_cost_microusd = 0
        self.reserved_max_cost_usd = 0.0
        self.call_records: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        if self.semantic_calls_dispatched >= self.maximum_semantic_calls:
            raise Phase13CalibrationError("CALIBRATION_SEMANTIC_CALL_CEILING_EXCEEDED")
        projected_input_tokens = sum(
            len(message.get("content", "").encode("utf-8")) + 16 for message in messages
        )
        if projected_input_tokens > self.maximum_input_tokens_per_attempt:
            raise Phase13CalibrationError("CALIBRATION_INPUT_TOKEN_CEILING_EXCEEDED")
        requested_output_tokens = config.get(
            "max_output_tokens", config.get("max_tokens", self.maximum_output_tokens_per_attempt)
        )
        if (
            type(requested_output_tokens) is not int
            or requested_output_tokens < 0
            or requested_output_tokens > self.maximum_output_tokens_per_attempt
        ):
            raise Phase13CalibrationError("CALIBRATION_OUTPUT_TOKEN_CEILING_EXCEEDED")
        if (
            self.reserved_max_cost_microusd
            + self.maximum_cost_per_semantic_call_microusd
            > self.hard_ceiling_microusd
        ):
            raise Phase13CalibrationError("CALIBRATION_COST_CEILING_EXCEEDED")
        self.semantic_calls_dispatched += 1
        self.reserved_max_cost_microusd += self.maximum_cost_per_semantic_call_microusd
        self.reserved_max_cost_usd = self.reserved_max_cost_microusd / 1_000_000
        response = self.client.chat(messages, model, config)
        attempts = response.raw.get("attempts")
        prompt_tokens = response.token_usage.get("prompt_tokens")
        completion_tokens = response.token_usage.get("completion_tokens")
        if (
            type(attempts) is not int
            or not 1 <= attempts <= self.maximum_attempts_per_semantic_call
            or type(prompt_tokens) is not int
            or prompt_tokens < 0
            or type(completion_tokens) is not int
            or completion_tokens < 0
        ):
            raise Phase13CalibrationError("CALIBRATION_PROVIDER_ACCOUNTING_REQUIRED")
        self.semantic_calls += 1
        self.transport_attempts += attempts
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens
        cost = response.raw.get("cost_usd")
        if type(cost) not in {int, float} or cost < 0:
            raise Phase13CalibrationError("CALIBRATION_PROVIDER_ACCOUNTING_REQUIRED")
        self.cost_usd += float(cost)
        observed_cost_microusd = int(
            (Decimal(str(cost)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING)
        )
        settled_max_cost_microusd = (
            (attempts - 1) * self.maximum_cost_per_transport_attempt_microusd
            + observed_cost_microusd
        )
        if (
            self.transport_attempts > self.maximum_transport_attempts
            or self.input_tokens > self.maximum_input_tokens
            or self.output_tokens > self.maximum_output_tokens
            or prompt_tokens > self.maximum_input_tokens_per_attempt
            or completion_tokens > self.maximum_output_tokens_per_attempt
            or self.cost_usd > self.hard_ceiling_usd
            or settled_max_cost_microusd > self.maximum_cost_per_semantic_call_microusd
        ):
            raise Phase13CalibrationError("CALIBRATION_PROVIDER_BUDGET_EXCEEDED")
        self.reserved_max_cost_microusd += (
            settled_max_cost_microusd - self.maximum_cost_per_semantic_call_microusd
        )
        self.reserved_max_cost_usd = self.reserved_max_cost_microusd / 1_000_000
        self.call_records.append(
            {
                "sequence": self.semantic_calls,
                "sample_id": config.get("sample_id"),
                "method_stage": config.get("method_stage"),
                "attempts": attempts,
                "prompt_tokens_observed": prompt_tokens,
                "completion_tokens_observed": completion_tokens,
                "cost_usd_observed": float(cost),
            }
        )
        return response
