from __future__ import annotations

from memcontam.tasks.base import TaskInstance


def build_instance(row: dict) -> TaskInstance:
    return TaskInstance(
        sample_id=row["sample_id"],
        task_name="math_equation_balancer",
        input=row["input"],
        verifier_spec=row.get("verifier_spec", {}),
    )
