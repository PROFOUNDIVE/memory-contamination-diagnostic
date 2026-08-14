from __future__ import annotations

from dataclasses import dataclass
import math

from memcontam.readiness.phase13_execution_contract import ExecutionRegistry


class CapacityPlanningError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


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
    execution: ExecutionRegistry,
    attempted_seed_counts: dict[str, int],
) -> CapacityPlan:
    if set(attempted_seed_counts) != set(execution.tasks) or any(
        type(count) is not int or count < 0 for count in attempted_seed_counts.values()
    ):
        raise CapacityPlanningError("ATTEMPTED_SEED_COUNTS_INVALID")
    nominal_by_task = {
        task: sum(
            row.nominal_semantic_calls_per_trial
            for row in execution.templates
            if row.task == task
        )
        for task in execution.tasks
    }
    maximum_by_task = {
        task: sum(
            row.maximum_semantic_calls_per_trial
            for row in execution.templates
            if row.task == task
        )
        for task in execution.tasks
    }
    nominal = sum(
        count
        * (
            execution.capacity.prefix_nominal_calls_per_seed
            + execution.H_run * nominal_by_task[task]
        )
        for task, count in attempted_seed_counts.items()
    )
    maximum = sum(
        count
        * (
            execution.capacity.prefix_maximum_calls_per_seed
            + execution.H_run * maximum_by_task[task]
        )
        for task, count in attempted_seed_counts.items()
    )
    reserved = math.ceil(maximum * (100 + execution.capacity.reserve_percent) / 100)
    attempts = execution.capacity.maximum_transport_attempts_per_semantic_call
    return CapacityPlan(
        nominal,
        maximum,
        reserved,
        maximum * attempts,
        reserved * attempts,
        reserved * attempts * execution.capacity.maximum_input_tokens_per_transport_attempt,
        reserved * attempts * execution.capacity.maximum_output_tokens_per_transport_attempt,
    )
