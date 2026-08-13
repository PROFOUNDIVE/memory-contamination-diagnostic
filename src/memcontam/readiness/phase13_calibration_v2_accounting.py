from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from memcontam.readiness.phase13_calibration_v2_runtime_models import TrajectoryRequest
from memcontam.readiness.phase13_provider_models import SettledCall


@dataclass(frozen=True, slots=True)
class TemplateClosure:
    template_id: str
    session_id: str
    expected_semantic_calls: int
    settled_semantic_calls: int
    settled_transport_attempts: int
    calls: tuple[SettledCall, ...]


@dataclass(frozen=True, slots=True)
class AccountingClosure:
    status: Literal["closed_complete", "closed_partial"]
    expected_semantic_calls: int
    settled_semantic_calls: int
    settled_transport_attempts: int
    templates: tuple[TemplateClosure, ...]


def close_accounting(request: TrajectoryRequest) -> AccountingClosure:
    rows: list[TemplateClosure] = []
    for provider in request.providers.values():
        report = provider.reconcile()
        template = next(
            row
            for row in request.verified.execution.execution_templates
            if row.template_id == provider.execution_template_id
        )
        expected = request.verified.execution.timing.H_run * template.nominal_semantic_calls_per_trial
        rows.append(
            TemplateClosure(
                template.template_id,
                request.session_id,
                expected,
                report.totals.semantic_calls,
                report.totals.transport_attempts,
                report.calls,
            )
        )
    total_expected = sum(row.expected_semantic_calls for row in rows)
    total_settled = sum(row.settled_semantic_calls for row in rows)
    attempts = sum(row.settled_transport_attempts for row in rows)
    status: Literal["closed_complete", "closed_partial"] = (
        "closed_complete"
        if all(row.settled_semantic_calls == row.expected_semantic_calls for row in rows)
        else "closed_partial"
    )
    return AccountingClosure(status, total_expected, total_settled, attempts, tuple(rows))


def empty_closure(request: TrajectoryRequest) -> AccountingClosure:
    expected = sum(
        request.verified.execution.timing.H_run * row.nominal_semantic_calls_per_trial
        for row in request.verified.execution.execution_templates
        if row.task == request.task
    )
    return AccountingClosure("closed_partial", expected, 0, 0, ())
