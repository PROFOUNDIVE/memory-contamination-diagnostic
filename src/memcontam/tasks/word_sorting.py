from __future__ import annotations

from memcontam.tasks.base import TaskInstance


def build_instance(row: dict) -> TaskInstance:
    return TaskInstance(
        sample_id=row["sample_id"],
        task_name="word_sorting",
        input={"words": row["words"]},
        verifier_spec={"sorted_words": row.get("sorted_words") or sorted(row["words"])},
    )
