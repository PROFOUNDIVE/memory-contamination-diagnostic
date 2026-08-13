from __future__ import annotations

from collections.abc import Mapping
import json
import time
from typing import Literal

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_authority import JsonValue
from memcontam.readiness.phase13_provider_models import (
    ProviderAccountingError,
    ProviderTotals,
    TransportAttempt,
)


def normalize_openai_response(
    call_id: str,
    owner_id: str,
    template_id: str,
    response: LLMResponse,
) -> tuple[tuple[TransportAttempt, ...], ProviderTotals]:
    attempts = response.raw.get("attempts")
    cost = response.raw.get("cost_usd", 0.0)
    storage = response.raw.get("storage_bytes", 0)
    prompt = response.token_usage.get("prompt_tokens")
    completion = response.token_usage.get("completion_tokens")
    if (
        type(attempts) is not int
        or attempts < 1
        or type(cost) not in {int, float}
        or type(storage) is not int
        or storage < 0
        or response.latency_ms is None
        or type(prompt) is not int
        or type(completion) is not int
    ):
        raise ProviderAccountingError("PROVIDER_ACCOUNTING_REQUIRED")
    rows = tuple(
        _attempt(
            call_id=call_id,
            owner_id=owner_id,
            template_id=template_id,
            number=number,
            status="completed" if number == attempts else "failed",
            input_tokens=prompt if number == attempts else 0,
            output_tokens=completion if number == attempts else 0,
            cost_microusd=int(round(float(cost) * 1_000_000)) if number == attempts else 0,
            latency_ms=response.latency_ms if number == attempts else 0,
            storage_bytes=storage if number == attempts else 0,
            evidence=response.raw,
        )
        for number in range(1, attempts + 1)
    )
    return rows, sum_attempts(rows)


def normalize_exception(
    call_id: str,
    owner_id: str,
    template_id: str,
    error: Exception,
    started: float,
) -> tuple[tuple[TransportAttempt, ...], ProviderTotals]:
    attempts = getattr(error, "provider_attempts_count", 1)
    elapsed = int((time.perf_counter() - started) * 1000)
    latency = getattr(error, "provider_latency_ms", elapsed)
    if type(attempts) is not int or attempts < 1 or type(latency) is not int or latency < 0:
        attempts, latency = 1, elapsed
    rows = tuple(
        _attempt(
            call_id=call_id,
            owner_id=owner_id,
            template_id=template_id,
            number=number,
            status="failed",
            input_tokens=0,
            output_tokens=0,
            cost_microusd=0,
            latency_ms=latency if number == attempts else 0,
            storage_bytes=0,
            evidence={"error_type": type(error).__name__, "error": str(error)},
        )
        for number in range(1, attempts + 1)
    )
    return rows, sum_attempts(rows)


def normalize_malformed_provider_failure(
    call_id: str,
    owner_id: str,
    template_id: str,
    error: Exception,
    raw_attempts: tuple[Mapping[str, JsonValue], ...],
    started: float,
) -> tuple[tuple[TransportAttempt, ...], ProviderTotals]:
    rows, _ = normalize_exception(call_id, owner_id, template_id, error, started)
    evidence = {
        "error_type": type(error).__name__,
        "error": str(error),
        "malformed_provider_attempts": json.dumps(raw_attempts, sort_keys=True),
    }
    terminal = rows[-1].model_copy(update={"raw_evidence": evidence})
    normalized = (*rows[:-1], terminal)
    return normalized, sum_attempts(normalized)


def sum_attempts(attempts: tuple[TransportAttempt, ...]) -> ProviderTotals:
    return ProviderTotals(
        transport_attempts=len(attempts),
        retries=max(0, len(attempts) - 1),
        input_tokens=sum(row.input_tokens for row in attempts),
        output_tokens=sum(row.output_tokens for row in attempts),
        cost_microusd=sum(row.cost_microusd for row in attempts),
        latency_ms=sum(row.latency_ms for row in attempts),
        storage_bytes=sum(row.storage_bytes for row in attempts),
    )


def _attempt(
    *,
    call_id: str,
    owner_id: str,
    template_id: str,
    number: int,
    status: Literal["completed", "failed", "partial"],
    input_tokens: int,
    output_tokens: int,
    cost_microusd: int,
    latency_ms: int,
    storage_bytes: int,
    evidence: Mapping[str, JsonValue],
) -> TransportAttempt:
    return TransportAttempt(
        attempt_id=f"{call_id}:attempt:{number}",
        semantic_call_id=call_id,
        execution_owner_id=owner_id,
        execution_template_id=template_id,
        attempt_number=number,
        status=status,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_microusd=cost_microusd,
        latency_ms=latency_ms,
        storage_bytes=storage_bytes,
        provider_error=None if status == "completed" else "transport_failure",
        raw_evidence={
            key: value
            if isinstance(value, (type(None), bool, int, float, str))
            else json.dumps(value, sort_keys=True)
            for key, value in evidence.items()
        },
    )
