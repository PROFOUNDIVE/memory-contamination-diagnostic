from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EstimabilityRule:
    minimum_seeds: int = 2

    def __post_init__(self) -> None:
        if type(self.minimum_seeds) is not int or self.minimum_seeds < 1:
            raise ValueError("INVALID_ESTIMABILITY_RULE")


@dataclass(frozen=True)
class EstimabilityDecision:
    estimable: bool
    code: str
    eligible_seed_count: int
    required_seed_count: int


def evaluate_estimability(
    panel: Mapping[Any, Any] | Sequence[Any], rule: EstimabilityRule
) -> EstimabilityDecision:
    values = tuple(panel.values()) if isinstance(panel, Mapping) else tuple(panel)
    complete = [
        value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(complete) != len(values):
        return EstimabilityDecision(
            False, "INCOMPLETE_SEED_PANEL", len(complete), rule.minimum_seeds
        )
    if not complete:
        return EstimabilityDecision(False, "NO_ELIGIBLE_SEEDS", 0, rule.minimum_seeds)
    if len(complete) < rule.minimum_seeds:
        return EstimabilityDecision(False, "INSUFFICIENT_SEEDS", len(complete), rule.minimum_seeds)
    return EstimabilityDecision(True, "ESTIMABLE", len(complete), rule.minimum_seeds)


__all__ = ["EstimabilityDecision", "EstimabilityRule", "evaluate_estimability"]
