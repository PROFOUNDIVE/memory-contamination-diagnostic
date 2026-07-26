from __future__ import annotations

from decimal import Decimal
from threading import Lock
from typing import Mapping
import warnings


class CostGuardError(RuntimeError):
    """Base error for fail-closed live-call accounting."""


class MissingUsageError(CostGuardError):
    """Raised when a completed provider response cannot be accounted for."""


class CostLimitExceeded(CostGuardError):
    """Raised before a request would exceed the run-level budget."""


class CostGuard:
    def __init__(
        self,
        *,
        input_per_million_usd: float = 2.50,
        cached_input_per_million_usd: float = 1.25,
        output_per_million_usd: float = 10.00,
        warning_usd: float = 3.00,
        hard_ceiling_usd: float = 5.00,
    ) -> None:
        self._input_rate = _decimal(input_per_million_usd)
        self._cached_input_rate = _decimal(cached_input_per_million_usd)
        self._output_rate = _decimal(output_per_million_usd)
        self._warning = _decimal(warning_usd)
        self._hard_ceiling = _decimal(hard_ceiling_usd)
        self._spent = Decimal()
        self._warned = False
        self._lock = Lock()

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return float(self._spent)

    def estimate_cost(self, *, input_tokens: int, output_tokens: int) -> float:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token estimates must be non-negative")
        return float(
            (Decimal(input_tokens) * self._input_rate + Decimal(output_tokens) * self._output_rate)
            / 1_000_000
        )

    def check_before_dispatch(self, projected_cost_usd: float) -> None:
        projected_cost = _decimal(projected_cost_usd)
        if projected_cost < 0:
            raise ValueError("projected cost must be non-negative")
        with self._lock:
            if self._spent + projected_cost > self._hard_ceiling:
                raise CostLimitExceeded(f"projected run cost exceeds USD {self._hard_ceiling}")

    def record_usage(self, usage: Mapping[str, object] | None) -> float:
        if not usage:
            raise MissingUsageError("provider response is missing usage")
        input_tokens = _usage_int(usage, "input_tokens")
        output_tokens = _usage_int(usage, "output_tokens")
        details = usage.get("input_tokens_details")
        cached_tokens = _usage_int(details, "cached_tokens", default=0)
        if cached_tokens > input_tokens:
            raise MissingUsageError("provider response has invalid cached usage")
        cost = (
            Decimal(input_tokens - cached_tokens) * self._input_rate
            + Decimal(cached_tokens) * self._cached_input_rate
            + Decimal(output_tokens) * self._output_rate
        ) / 1_000_000
        with self._lock:
            self._spent += cost
            if self._spent >= self._warning and not self._warned:
                warnings.warn(f"live run cost reached USD {self._warning}", RuntimeWarning, stacklevel=2)
                self._warned = True
            return float(cost)


def _usage_int(value: object, key: str, *, default: int | None = None) -> int:
    if not isinstance(value, Mapping) or key not in value:
        if default is not None:
            return default
        raise MissingUsageError("provider response is missing usage")
    result = value[key]
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise MissingUsageError("provider response has invalid usage")
    return result


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))
