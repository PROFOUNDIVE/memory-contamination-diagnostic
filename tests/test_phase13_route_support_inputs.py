from __future__ import annotations

import pytest

from memcontam.readiness.phase13_support_planning import (
    PlanningError,
    RoutePlanningRequest,
    StochasticSupportInput,
    plan_route,
)
from .test_phase13_support_planning import L1, ROOT, _required_inputs, _rows


@pytest.mark.parametrize(
    ("extra", "code"),
    [
        (
            StochasticSupportInput("game24", "caller-defined-support", _rows(12)),
            "ROUTE_SUPPORT_POPULATION_UNREGISTERED",
        ),
        (
            StochasticSupportInput("other_task", L1[0], _rows(12)),
            "ROUTE_SUPPORT_TASK_UNREGISTERED",
        ),
        (
            StochasticSupportInput("game24", L1[0], _rows(12)),
            "ROUTE_SUPPORT_DUPLICATE",
        ),
    ],
)
def test_route_rejects_unregistered_cross_task_and_duplicate_support_rows(
    extra: StochasticSupportInput, code: str
) -> None:
    route_request = RoutePlanningRequest("3w", (*_required_inputs(), extra))

    with pytest.raises(PlanningError) as caught:
        plan_route(route_request, ROOT)

    assert caught.value.code == code
