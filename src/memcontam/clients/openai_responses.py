from __future__ import annotations

import inspect
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, cast

import httpx
from openai import APIStatusError, APITimeoutError, OpenAI
from openai.resources.responses.responses import Responses

from memcontam.clients.base import LLMResponse
from memcontam.clients.config import ProviderConfig
from memcontam.clients.cost_guard import CostGuard


class LiveCallNotAuthorized(RuntimeError):
    """Raised before an unapproved live request can reach the provider."""


class LunaContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class OpenAIResponsesClient:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        allow_live_calls: bool,
        cost_guard: CostGuard | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.provider != "openai_responses":
            raise ValueError("OpenAIResponsesClient requires provider=openai_responses")
        self._config = config
        self._allow_live_calls = allow_live_calls
        self._sleep = sleep
        self.cost_guard = cost_guard or CostGuard(
            input_per_million_usd=config.input_per_million_usd,
            cached_input_per_million_usd=config.cached_input_per_million_usd,
            output_per_million_usd=config.output_per_million_usd,
        )
        _load_repository_dotenv()
        api_key_env = config.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key env var: {api_key_env}")
        options: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if config.base_url is not None:
            options["base_url"] = config.base_url
        if config.timeout_seconds is not None:
            options["timeout"] = config.timeout_seconds
        self.client = OpenAI(**options)

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        self._assert_live_call_authorized()
        max_output_tokens = config.get("max_output_tokens", config.get("max_tokens"))
        if not isinstance(max_output_tokens, int) or max_output_tokens < 0:
            max_output_tokens = self._config.max_output_tokens
        self.cost_guard.check_before_dispatch(
            self.cost_guard.estimate_cost(
                input_tokens=_conservative_input_tokens(messages),
                output_tokens=max_output_tokens,
            )
        )
        request = {
            "model": model,
            "input": cast(Any, messages),
            "temperature": config.get("temperature", 0),
            "top_p": config.get("top_p", 1),
            "max_output_tokens": max_output_tokens,
            "service_tier": self._config.service_tier,
            "store": self._config.store,
        }
        if model == "gpt-5.6-luna":
            if (
                self._config.timeout_seconds != 180
                or self._config.retries_after_initial_attempt != 2
                or self._config.store
                or "previous_response_id" in config
            ):
                raise LunaContractError("LUNA_RUNTIME_CONTRACT_MISMATCH")
            expected_output_tokens = (
                8192 if config.get("method_stage") == "dc_rs_synthesize" else 4096
            )
            if max_output_tokens != expected_output_tokens:
                raise LunaContractError("LUNA_OUTPUT_CONTRACT_MISMATCH")
            request.update(
                reasoning={"mode": "standard", "effort": "none", "context": "current_turn"},
                tools=[],
                store=False,
            )
        seed_parameter_sent = False
        if "requested_seed" in config and _responses_support_seed():
            request["seed"] = config["requested_seed"]
            seed_parameter_sent = True

        start = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self.client.responses.create(**cast(Any, request))
                break
            except Exception as error:
                if not _is_retryable(error) or attempts > self._config.retries_after_initial_attempt:
                    setattr(error, "provider_attempts_count", attempts)
                    setattr(error, "provider_latency_ms", int((time.perf_counter() - start) * 1000))
                    raise
                self._sleep(self._config.retry_delays_seconds[attempts - 1])

        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = _usage_dict(getattr(response, "usage", None))
        token_usage = _token_usage(usage)
        cost_usd = self.cost_guard.record_usage(usage)
        return LLMResponse(
            content=_output_text(response),
            raw={
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", model),
                "usage": usage,
                "latency_ms": latency_ms,
                "attempts": attempts,
                "service_tier": getattr(response, "service_tier", self._config.service_tier),
                "cost_usd": cost_usd,
                "seed_parameter_sent": seed_parameter_sent,
            },
            token_usage=token_usage,
            latency_ms=latency_ms,
        )

    def _assert_live_call_authorized(self) -> None:
        if not self._config.live_calls_enabled:
            raise LiveCallNotAuthorized("live calls require config.live_calls.enabled=true")
        if not self._allow_live_calls:
            raise LiveCallNotAuthorized("live calls require --allow-live-calls")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_repository_dotenv() -> None:
    path = _repository_root() / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _conservative_input_tokens(messages: list[dict[str, str]]) -> int:
    return sum(len(message.get("content", "")) + 16 for message in messages)


def _responses_support_seed() -> bool:
    return "seed" in inspect.signature(Responses.create).parameters


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (APITimeoutError, TimeoutError, httpx.TimeoutException)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and (status_code == 429 or status_code >= 500)


def _usage_dict(usage: object) -> dict[str, object] | None:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        return dict(usage)
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        dumped = dump()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    return {
        key: getattr(usage, key)
        for key in ("input_tokens", "input_tokens_details", "output_tokens", "total_tokens")
        if hasattr(usage, key)
    }


def _token_usage(usage: Mapping[str, object] | None) -> dict[str, int]:
    if usage is None:
        return {}
    details = usage.get("input_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0
    values = {
        "prompt_tokens": usage.get("input_tokens"),
        "cached_prompt_tokens": cached,
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    return {key: value for key, value in values.items() if isinstance(value, int) and not isinstance(value, bool)}


def _output_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    output = getattr(response, "output", [])
    fragments = []
    for item in output if isinstance(output, list) else []:
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if isinstance(text, str):
                fragments.append(text)
    return "".join(fragments)
