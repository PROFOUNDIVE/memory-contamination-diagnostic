from __future__ import annotations

from memcontam.tasks.base import TaskInstance


def build_instance(row: dict) -> TaskInstance:
    if "input" not in row:
        raise ValueError("math_equation_balancer row must include 'input'")

    verifier_spec = row.get("verifier_spec")
    if not isinstance(verifier_spec, dict):
        raise ValueError("math_equation_balancer row must include 'verifier_spec' object")
    if "target" not in verifier_spec:
        raise ValueError("math_equation_balancer verifier_spec must include 'target'")
    if "target_value" not in verifier_spec:
        raise ValueError("math_equation_balancer verifier_spec must include 'target_value'")

    return TaskInstance(
        sample_id=row["sample_id"],
        task_name="math_equation_balancer",
        input={"input": row["input"]},
        verifier_spec=dict(verifier_spec),
    )
