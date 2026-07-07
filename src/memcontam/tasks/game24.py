from __future__ import annotations

from memcontam.tasks.base import TaskInstance


def build_instance(row: dict) -> TaskInstance:
    return TaskInstance(
        sample_id=row["sample_id"],
        task_name="game24",
        input={"numbers": row["numbers"]},
        verifier_spec={"target": row.get("target", 24)},
    )
