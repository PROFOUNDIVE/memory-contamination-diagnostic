from __future__ import annotations

from threading import Lock
from typing import Protocol

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


class ProviderCallBudgetError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ProviderResponse(Protocol):
    id: str
    output_text: str


class ResponsesResource(Protocol):
    def create(self, **kwargs: JsonValue) -> ProviderResponse: ...


class ProviderCallBudget:
    def __init__(self, maximum_calls: int) -> None:
        self._maximum_calls = maximum_calls
        self._issued = 0
        self._lock = Lock()

    @property
    def issued(self) -> int:
        return self._issued

    def consume(self) -> None:
        with self._lock:
            if self._issued >= self._maximum_calls:
                raise ProviderCallBudgetError("READINESS0_PROVIDER_CALL_CEILING_EXCEEDED")
            self._issued += 1


class BudgetedResponses:
    def __init__(self, delegate: ResponsesResource, *, maximum_calls: int) -> None:
        self._delegate = delegate
        self._budget = ProviderCallBudget(maximum_calls)

    @property
    def issued(self) -> int:
        return self._budget.issued

    def create(self, **kwargs: JsonValue) -> ProviderResponse:
        self._budget.consume()
        return self._delegate.create(**kwargs)


__all__ = ["BudgetedResponses", "ProviderCallBudget", "ProviderCallBudgetError"]
