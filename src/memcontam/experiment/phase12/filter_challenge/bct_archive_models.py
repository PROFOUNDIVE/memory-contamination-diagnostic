from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal


_Stage = Literal["screening", "bct"]


class LedgerError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    calls: int
    input_tokens: int
    output_tokens: int
    microusd: int
    wall_seconds: int


STAGE_CAPS = {
    "screening": ResourceBudget(90, 368_640, 57_600, 2_000_000, 3_600),
    "bct": ResourceBudget(480, 1_966_080, 307_200, 8_000_000, 7_200),
}


def resource_fits(limit: ResourceBudget, requested: ResourceBudget) -> bool:
    return all(
        available >= used
        for available, used in zip(
            (limit.calls, limit.input_tokens, limit.output_tokens, limit.microusd, limit.wall_seconds),
            (requested.calls, requested.input_tokens, requested.output_tokens, requested.microusd, requested.wall_seconds),
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class ProcessReservation:
    reservation_id: str
    stage: _Stage
    resources: ResourceBudget

    @property
    def reserved_wall_seconds(self) -> int:
        return self.resources.wall_seconds


@dataclass(frozen=True, slots=True)
class ProcessDeadline:
    deadline_monotonic: float

    def clamp_timeout(self, requested_seconds: float, monotonic_now: float | None = None) -> float:
        if requested_seconds < 0:
            raise LedgerError("REQUEST_TIMEOUT_INVALID")
        now = time.monotonic() if monotonic_now is None else monotonic_now
        return min(requested_seconds, max(0.0, self.deadline_monotonic - now))


@dataclass(frozen=True, slots=True)
class ArchiveValidation:
    valid: bool
    reason_code: str | None = None
