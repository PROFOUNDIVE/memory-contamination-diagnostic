from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Any, Literal


NOT_ESTIMABLE = "not_estimable"


@dataclass(frozen=True)
class BootstrapConfig:
    replicates: int = 2_000
    confidence_level: float = 0.95
    random_seed: int = 0
    unit: Literal["seed", "trial"] = "seed"

    def __post_init__(self) -> None:
        if type(self.replicates) is not int or self.replicates < 1:
            raise ValueError("INVALID_BOOTSTRAP_CONFIG")
        if not 0 < self.confidence_level < 1:
            raise ValueError("INVALID_BOOTSTRAP_CONFIG")


@dataclass(frozen=True)
class Interval:
    estimate: float | str
    lower: float | str
    upper: float | str
    status: str = "estimable"


@dataclass(frozen=True)
class IntervalSet:
    intervals: Mapping[str, Interval]
    resampling_unit: Literal["seed"] = "seed"


def bootstrap_seeds(panel: Mapping[Any, Any], config: BootstrapConfig) -> IntervalSet:
    if config.unit != "seed":
        raise _aggregate_error("TRIAL_RESAMPLING_FORBIDDEN")
    metrics = _metrics(panel)
    generator = Random(config.random_seed)
    intervals = {
        name: _interval(values, config, generator)
        for name, values in metrics.items()
    }
    return IntervalSet(intervals)


def _metrics(panel: Mapping[Any, Any]) -> dict[str, Mapping[Any, Any]]:
    if not isinstance(panel, Mapping) or not panel:
        raise _aggregate_error("TRIAL_RESAMPLING_FORBIDDEN")
    values = tuple(panel.values())
    if all(isinstance(value, Mapping) for value in values):
        return {str(name): value for name, value in panel.items()}
    if any(isinstance(value, (Mapping, list, tuple, set)) for value in values):
        raise _aggregate_error("TRIAL_RESAMPLING_FORBIDDEN")
    return {"estimate": panel}


def _interval(values: Mapping[Any, Any], config: BootstrapConfig, generator: Random) -> Interval:
    numeric = tuple(values.values())
    if (
        not numeric
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in numeric)
        or len(numeric) < 2
    ):
        return Interval(NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE)
    samples = sorted(
        mean(generator.choice(numeric) for _ in numeric)
        for _ in range(config.replicates)
    )
    tail = (1 - config.confidence_level) / 2
    return Interval(
        mean(numeric),
        samples[round((len(samples) - 1) * tail)],
        samples[round((len(samples) - 1) * (1 - tail))],
    )


def _aggregate_error(code: str) -> Exception:
    from memcontam.evaluation.phase12_aggregate import AggregateError

    return AggregateError(code)


__all__ = ["BootstrapConfig", "Interval", "IntervalSet", "bootstrap_seeds"]
