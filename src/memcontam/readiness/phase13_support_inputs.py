from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from memcontam.readiness.phase13_analysis_models import AnalysisRegistry


class SupportInputRow(Protocol):
    @property
    def task(self) -> str: ...

    @property
    def support_population_id(self) -> str: ...


class SupportInputError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def validate_route_support_inputs(
    rows: Sequence[SupportInputRow], analysis: AnalysisRegistry
) -> None:
    tasks = {family.task for family in analysis.inference.families}
    populations = {
        *(row.support_population_id for row in analysis.support.level_1),
        *(row.support_population_id for row in analysis.support.level_2),
        analysis.support.level_3.support_population_id,
    }
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if row.task not in tasks:
            raise SupportInputError("ROUTE_SUPPORT_TASK_UNREGISTERED")
        if row.support_population_id not in populations:
            raise SupportInputError("ROUTE_SUPPORT_POPULATION_UNREGISTERED")
        key = (row.task, row.support_population_id)
        if key in keys:
            raise SupportInputError("ROUTE_SUPPORT_DUPLICATE")
        keys.add(key)


__all__ = ("SupportInputError", "validate_route_support_inputs")
