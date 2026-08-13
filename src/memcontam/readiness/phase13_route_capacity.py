from __future__ import annotations

from dataclasses import dataclass

from memcontam.readiness.phase13_execution_models import ExecutionRegistry


class CapacityPlanningError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    nominal_semantic_calls: int
    raw_maximum_semantic_calls: int
    reserved_semantic_calls: int
    raw_maximum_transport_attempts: int
    reserved_transport_attempts: int
    maximum_input_tokens: int
    maximum_output_tokens: int


def recompute_capacity(
    execution: ExecutionRegistry, attempted: tuple[tuple[str, int], ...]
) -> CapacityPlan:
    prefix = next(row for row in execution.call_components if row.owner_kind == "prefix")
    trial = next(row for row in execution.call_components if row.owner_kind == "execution")
    nominal_by_task = {
        task: sum(
            row.nominal_semantic_calls_per_trial
            for row in execution.execution_templates
            if row.task == task
        )
        for task, _ in attempted
    }
    raw_by_task = {
        task: sum(
            row.raw_maximum_semantic_calls_per_trial
            for row in execution.execution_templates
            if row.task == task
        )
        for task, _ in attempted
    }
    if any(
        nominal_by_task[task] != trial.nominal_calls_per_activation
        or raw_by_task[task] != trial.raw_maximum_calls_per_activation
        for task, _ in attempted
    ):
        raise CapacityPlanningError("CAPACITY_COMPONENT_MISMATCH")
    nominal = sum(
        count
        * (prefix.nominal_calls_per_activation + execution.timing.H_run * nominal_by_task[task])
        for task, count in attempted
    )
    raw = sum(
        count
        * (prefix.raw_maximum_calls_per_activation + execution.timing.H_run * raw_by_task[task])
        for task, count in attempted
    )
    reserve = execution.planning_illustrations.reserve_percent
    reserved = (raw * (100 + reserve) + 99) // 100
    transports = execution.planning_illustrations.maximum_transport_attempts_per_semantic_call
    return CapacityPlan(
        nominal_semantic_calls=nominal,
        raw_maximum_semantic_calls=raw,
        reserved_semantic_calls=reserved,
        raw_maximum_transport_attempts=raw * transports,
        reserved_transport_attempts=reserved * transports,
        maximum_input_tokens=(
            reserved
            * transports
            * execution.planning_illustrations.maximum_input_tokens_per_transport_attempt
        ),
        maximum_output_tokens=(
            reserved
            * transports
            * execution.planning_illustrations.maximum_output_tokens_per_transport_attempt
        ),
    )


__all__ = ("CapacityPlan", "CapacityPlanningError", "recompute_capacity")
