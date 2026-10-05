from __future__ import annotations

# ruff: noqa: B010
import hashlib
import inspect
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, __version__ as openai_version
from openai.resources.responses.responses import Responses

from memcontam.clients.base import LLMResponse
from memcontam.clients.config import ProviderConfig
from memcontam.clients.cost_guard import CostGuard
from memcontam.readiness.phase13_readiness0_budget import BudgetedResponses, ResponsesResource
from memcontam.readiness.phase13_v3_count import (
    CountIdentityV3, CountReceiptV3, count_operation, count_projection, validate_count_receipt,
)
from memcontam.readiness.phase13_v3_request import (
    STAGES,
    CompiledProviderRequestV3,
    PackageBindingV3,
    compile_request_bytes,
)


class LiveCallNotAuthorized(RuntimeError):
    """Raised before an unapproved live request can reach the provider."""


class LunaContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        self.provider_attempts_count = 0
        super().__init__(code)


class OpenAIResponsesClient:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        allow_live_calls: bool,
        cost_guard: CostGuard | None = None,
        maximum_provider_calls: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        v3_binding: PackageBindingV3 | None = None,
    ) -> None:
        if config.provider != "openai_responses":
            raise ValueError("OpenAIResponsesClient requires provider=openai_responses")
        self._config = config
        self._allow_live_calls = allow_live_calls
        self._sleep = sleep
        self._v3_binding = v3_binding
        self.cost_guard = cost_guard or CostGuard(
            input_per_million_usd=config.input_per_million_usd,
            cached_input_per_million_usd=config.cached_input_per_million_usd,
            output_per_million_usd=config.output_per_million_usd,
        )
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
        self._responses = (
            self.client.responses
            if maximum_provider_calls is None
            else BudgetedResponses(
                cast(ResponsesResource, self.client.responses), maximum_calls=maximum_provider_calls
            )
        )

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
        compiled_v3 = config.get("_phase13_compiled_v3")
        bound_v3 = isinstance(compiled_v3, CompiledProviderRequestV3) and (
            self._v3_binding is not None and compiled_v3.binding == self._v3_binding
        )
        if self._v3_binding is not None and not bound_v3:
            raise LunaContractError("MAIN_COUNT_RECEIPT_REQUIRED")
        if compiled_v3 is not None and not bound_v3:
            raise LunaContractError("LUNA_RUNTIME_CONTRACT_MISMATCH")
        registered_cost_policy = False
        if model == "gpt-5.6-luna":
            registered_cost_policy = bound_v3 or (
                config.get("_phase13_execution_envelope_id")
                == "CORE_EXECUTION_ENVELOPE_REGISTRY_V2"
            )
            if (
                self._config.timeout_seconds != 180
                or self._config.retries_after_initial_attempt != (0 if bound_v3 else 2)
                or self._config.service_tier != "default"
                or self._config.store
                or "previous_response_id" in config
            ):
                raise LunaContractError("LUNA_RUNTIME_CONTRACT_MISMATCH")
            if registered_cost_policy:
                registered_attempts = config.get("_phase13_maximum_transport_attempts")
                if registered_attempts != 1 or max_output_tokens not in {384, 512, 8192}:
                    raise LunaContractError("LUNA_RUNTIME_CONTRACT_MISMATCH")
            elif max_output_tokens != (
                8192 if config.get("method_stage") == "dc_rs_synthesize" else 4096
            ):
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
        request_contract = _request_contract(request, messages)
        authority_contract = _authority_contract(config, max_output_tokens)
        if bound_v3:
            assert isinstance(compiled_v3, CompiledProviderRequestV3)
            self._require_count_receipt(compiled_v3)
            request = json.loads(compiled_v3.request_bytes)

        start = time.perf_counter()
        attempts = 0
        retries_after_initial_attempt = (
            0 if registered_cost_policy else self._config.retries_after_initial_attempt
        )
        while True:
            attempts += 1
            if bound_v3:
                assert isinstance(compiled_v3, CompiledProviderRequestV3)
                config["_phase13_before_request"]()
                self._require_count_receipt(compiled_v3, started=True)
                if request != json.loads(compiled_v3.request_bytes):
                    raise LunaContractError("MAIN_COUNT_REQUEST_BINDING_MISMATCH")
            try:
                response = self._responses.create(**cast(Any, request))
                break
            except Exception as transport_error:
                if not _is_retryable(transport_error) or attempts > retries_after_initial_attempt:
                    if registered_cost_policy:
                        retry_class = _phase13_retry_class(transport_error)
                        if retry_class is not None:
                            setattr(transport_error, "phase13_retry_class", retry_class)
                            if (isinstance(transport_error, APIStatusError)
                                    and (transport_error.response.status_code == 429
                                         or 500 <= transport_error.response.status_code <= 599)):
                                setattr(transport_error, "provider_failure_acknowledged", True)
                    setattr(transport_error, "provider_attempts_count", attempts)
                    setattr(transport_error, "provider_latency_ms", int((time.perf_counter() - start) * 1000))
                    setattr(transport_error, "provider_request_contract", request_contract)
                    setattr(transport_error, "provider_authority_contract", authority_contract)
                    setattr(transport_error, "provider_service_tier", self._config.service_tier)
                    raise
                self._sleep(self._config.retry_delays_seconds[attempts - 1])

        latency_ms = int((time.perf_counter() - start) * 1000)
        usage = _usage_dict(getattr(response, "usage", None))
        token_usage = _token_usage(usage)
        cost_usd = (
            None if bound_v3 and not {"prompt_tokens", "completion_tokens"} <= token_usage.keys()
            else self.cost_guard.record_usage(usage)
        )
        authoritative_cost = getattr(response, "cost_usd", None)
        if not isinstance(authoritative_cost, (int, float)) or isinstance(
            authoritative_cost, bool
        ):
            authoritative_cost = None
        selected_cost = float(authoritative_cost) if authoritative_cost is not None else cost_usd
        cost_source = (
            "AUTHORITATIVE_PROVIDER"
            if authoritative_cost is not None
            else "DERIVED_FROM_PROVIDER_USAGE"
        )
        if registered_cost_policy and getattr(response, "status", None) == "incomplete":
            error = LunaContractError("LUNA_PROVIDER_INCOMPLETE")
            setattr(error, "provider_attempts_count", attempts)
            setattr(error, "provider_latency_ms", latency_ms)
            setattr(error, "provider_status", "incomplete")
            setattr(
                error,
                "provider_incomplete_reason",
                getattr(getattr(response, "incomplete_details", None), "reason", None),
            )
            setattr(error, "provider_usage", usage)
            setattr(error, "provider_token_usage", token_usage)
            setattr(error, "provider_cost_usd", selected_cost)
            setattr(error, "provider_response_id", getattr(response, "id", None))
            setattr(error, "provider_service_tier", getattr(response, "service_tier", self._config.service_tier))
            setattr(error, "provider_returned_model", getattr(response, "model", model))
            setattr(error, "provider_response_status", getattr(response, "status", None))
            setattr(error, "provider_request_contract", request_contract)
            setattr(error, "provider_authority_contract", authority_contract)
            setattr(error, "authoritative_provider_cost_usd", authoritative_cost)
            setattr(error, "derived_cost_usd", cost_usd)
            setattr(error, "provider_cost_source", cost_source)
            raise error
        return LLMResponse(
            content=_output_text(response),
            raw={
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", model),
                "status": getattr(response, "status", None),
                "usage": usage,
                "latency_ms": latency_ms,
                "attempts": attempts,
                "service_tier": getattr(response, "service_tier", self._config.service_tier),
                "cost_usd": selected_cost,
                "authoritative_provider_cost_usd": authoritative_cost,
                "derived_cost_usd": cost_usd,
                "cost_source": cost_source,
                "request_contract": request_contract,
                "authority_contract": authority_contract,
                "seed_parameter_sent": seed_parameter_sent,
            },
            token_usage=token_usage,
            latency_ms=latency_ms,
        )

    def send_compiled_v3(
        self, compiled: CompiledProviderRequestV3, before_request: Callable[[], None],
    ) -> LLMResponse:
        self._require_count_receipt(compiled)
        if (
            self._v3_binding is None or compiled.binding != self._v3_binding
            or compiled.request_bytes != compile_request_bytes(compiled.key, compiled.material)
            or type(compiled.token_count) is not int or compiled.token_count < 0
        ):
            raise LunaContractError("LUNA_RUNTIME_CONTRACT_MISMATCH")
        return self.chat(
            [message.model_dump() for message in compiled.material.messages], "gpt-5.6-luna", {
                "temperature": compiled.material.temperature, "top_p": compiled.material.top_p,
                "max_output_tokens": STAGES[compiled.key.stage][1],
                "_phase13_maximum_transport_attempts": 1,
                "_phase13_execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V4",
                "_phase13_execution_envelope_sha256": "5796df90795ff7f753aad753abc1a70499c083fdb40dabb7a26afe011e58be38",
                "_phase13_maximum_input_tokens": STAGES[compiled.key.stage][0],
                "_phase13_terminal_failure_contract_id": "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V2",
                "_phase13_terminal_failure_contract_sha256": "599eb322efdfea397c227fdad25f2d9371c444392eb68ef943a04748467d16bd",
                "_phase13_compiled_v3": compiled, "_phase13_before_request": before_request,
            },
        )

    def count_identity_v3(self) -> CountIdentityV3:
        from pathlib import Path

        from memcontam.readiness import phase13_v3_count, phase13_v3_request

        source = b"".join(Path(filename).read_bytes()
                          for filename in (__file__, phase13_v3_count.__file__, phase13_v3_request.__file__))
        schema = json.dumps(CountReceiptV3.model_json_schema(), sort_keys=True).encode()
        runtime = json.dumps([sys.executable, sys.version, openai_version,
                              self._config.timeout_seconds, self._config.retries_after_initial_attempt],
                             separators=(",", ":")).encode()
        account = json.dumps([self.client.api_key, self.client.organization, self.client.project],
                             separators=(",", ":")).encode()
        return CountIdentityV3(base_url=str(self.client.base_url), sdk_version=openai_version,
            account_sha256=hashlib.sha256(account).hexdigest(),
            source_sha256=hashlib.sha256(source).hexdigest(),
            runtime_sha256=hashlib.sha256(runtime).hexdigest(), schema_sha256=hashlib.sha256(schema).hexdigest())

    def count_compiled_v3(self, compiled: CompiledProviderRequestV3,
                          before_count: Callable[[], None]) -> CountReceiptV3:
        self._assert_live_call_authorized()
        if (self._v3_binding != compiled.binding or self._config.timeout_seconds != 180
                or self._config.retries_after_initial_attempt != 0):
            raise LunaContractError("MAIN_COUNT_RUNTIME_CONTRACT_MISMATCH")
        operation = count_operation(compiled, self.count_identity_v3())
        projection = json.loads(count_projection(compiled))
        before_count()
        if operation != count_operation(compiled, self.count_identity_v3()):
            raise LunaContractError("MAIN_COUNT_REQUEST_BINDING_MISMATCH")
        response = self.client.responses.input_tokens.count(**projection)
        if (getattr(response, "object", None) != "response.input_tokens"
                or type(getattr(response, "input_tokens", None)) is not int
                or response.input_tokens < 0):
            raise LunaContractError("MAIN_COUNT_RESPONSE_INVALID")
        return CountReceiptV3(operation=operation, object=response.object,
            input_tokens=response.input_tokens, monetary_cost_usd=getattr(response, "cost_usd", None))

    def _require_count_receipt(self, compiled: CompiledProviderRequestV3, *, started: bool = False) -> None:
        if compiled.count_receipt is None or compiled.verify_durable_count is None:
            raise LunaContractError("MAIN_COUNT_RECEIPT_REQUIRED")
        compiled.verify_durable_count(started)
        validate_count_receipt(compiled, compiled.count_receipt, self.count_identity_v3())

    def _assert_live_call_authorized(self) -> None:
        if not self._config.live_calls_enabled:
            raise LiveCallNotAuthorized("live calls require config.live_calls.enabled=true")
        if not self._allow_live_calls:
            raise LiveCallNotAuthorized("live calls require --allow-live-calls")


def _conservative_input_tokens(messages: list[dict[str, str]]) -> int:
    return sum(len(message.get("content", "")) + 16 for message in messages)


def _request_contract(request: dict[str, object], messages: list[dict[str, str]]) -> dict[str, object]:
    return {
        "model": request["model"],
        "input_sha256": hashlib.sha256(json.dumps(
            messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest(),
        "temperature": request["temperature"], "top_p": request["top_p"],
        "reasoning": request.get("reasoning"),
        "previous_response_id": request.get("previous_response_id"),
        "service_tier": request["service_tier"], "store": request["store"],
        "tools": request.get("tools", []), "max_output_tokens": request["max_output_tokens"],
    }


def _authority_contract(config: dict, max_output_tokens: int) -> dict[str, object]:
    return {
        "maximum_input_tokens": config.get("_phase13_maximum_input_tokens"),
        "maximum_output_tokens": max_output_tokens,
        "execution_envelope_id": config.get("_phase13_execution_envelope_id"),
        "execution_envelope_sha256": config.get("_phase13_execution_envelope_sha256"),
        "failure_contract_id": config.get("_phase13_failure_contract_id"),
        "failure_contract_sha256": config.get("_phase13_failure_contract_sha256"),
        "terminal_failure_contract_id": config.get("_phase13_terminal_failure_contract_id"),
        "terminal_failure_contract_sha256": config.get("_phase13_terminal_failure_contract_sha256"),
        "rate_card_sha256": config.get("_phase13_rate_card_sha256"),
    }


def _responses_support_seed() -> bool:
    return "seed" in inspect.signature(Responses.create).parameters


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (APITimeoutError, TimeoutError, httpx.TimeoutException)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and (status_code == 429 or status_code >= 500)


def _phase13_retry_class(error: Exception) -> str | None:
    if isinstance(error, (APITimeoutError, TimeoutError, httpx.TimeoutException)):
        return "TIMEOUT_BEFORE_SEMANTIC_PAYLOAD"
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return "HTTP_429_BEFORE_SEMANTIC_PAYLOAD"
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return "REGISTERED_PROVIDER_5XX_BEFORE_SEMANTIC_PAYLOAD"
    if isinstance(error, (APIConnectionError, ConnectionError, httpx.NetworkError)):
        return "CONNECTION_FAILURE_BEFORE_SEMANTIC_PAYLOAD"
    return None


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
