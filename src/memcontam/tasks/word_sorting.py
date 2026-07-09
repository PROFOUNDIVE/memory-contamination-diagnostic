from __future__ import annotations

from memcontam.tasks.base import TaskInstance


def build_instance(row: dict) -> TaskInstance:
    if "words" not in row:
        raise ValueError("word_sorting row must include 'words'")

    words = row["words"]
    if not isinstance(words, list):
        raise ValueError("word_sorting row 'words' must be a list")

    sorted_words = row.get("sorted_words")
    if sorted_words is None:
        sorted_words = sorted(words)

    return TaskInstance(
        sample_id=row["sample_id"],
        task_name="word_sorting",
        input={"words": words},
        verifier_spec={"sorted_words": sorted_words},
    )
