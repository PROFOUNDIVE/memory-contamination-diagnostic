from __future__ import annotations

import pytest

from memcontam.cli import TASK_DISPATCH
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_core_task_json
from memcontam.tasks.multiple_choice import build_instance


def test_core_multiple_choice_tasks_stay_out_of_legacy_dispatch() -> None:
    for task_name in ("mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond"):
        assert task_name not in TASK_DISPATCH
        task = TaskInstance(
            sample_id=f"{task_name}:1",
            task_name=task_name,
            input={"question": "Which?", "options": ["one", "two"]},
            verifier_spec={"answer_index": 1, "answer_label": "B"},
        )

        built = build_instance(task.model_dump())
        from memcontam.tasks.multiple_choice import verify_answer

        result = verify_answer("(B)", built)

        assert result.is_correct is True
        assert result.parsed_answer == "B"


def test_core_prompt_serialization_excludes_verifier_and_provenance() -> None:
    task = TaskInstance(
        sample_id="gpqa_diamond:secret-record-id",
        task_name="gpqa_diamond",
        input={"question": "Which?", "options": ["one", "two"]},
        verifier_spec={"answer_index": 1, "answer_label": "B"},
        metadata={"dataset_repo": "gated/source", "upstream_record_id": "secret-record-id"},
    )

    assert canonical_core_task_json(task) == (
        '{"input":{"options":["one","two"],"question":"Which?"},'
        '"task_name":"gpqa_diamond"}'
    )


@pytest.mark.parametrize(
    "change",
    [
        {"task_name": "game24"},
        {"input": {"question": "Which?", "options": ["only one"]}},
        {"verifier_spec": {"answer_index": 2, "answer_label": "C"}},
        {"verifier_spec": {"answer_index": 1, "answer_label": "A"}},
    ],
)
def test_core_multiple_choice_build_rejects_malformed_rows(change: dict) -> None:
    row = {
        "sample_id": "gpqa_diamond:1",
        "task_name": "gpqa_diamond",
        "input": {"question": "Which?", "options": ["one", "two"]},
        "verifier_spec": {"answer_index": 1, "answer_label": "B"},
    }
    row.update(change)

    with pytest.raises(ValueError):
        build_instance(row)
