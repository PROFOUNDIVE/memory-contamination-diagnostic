from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.tasks import dispatch
from memcontam.tasks.base import TaskInstance


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "data/phase13/main/mr_p4/corrected_v1"


def _render(task: TaskInstance) -> str:
    renderer = getattr(dispatch, "render_common_task_spec", None)
    assert callable(renderer)
    rendered = renderer(task)
    assert isinstance(rendered, str)
    return rendered


@pytest.mark.parametrize(
    "task",
    (
        TaskInstance(sample_id="g24", task_name="game24", input={"numbers": [3, 3, 8, 8]}),
        TaskInstance(
            sample_id="meb",
            task_name="math_equation_balancer",
            input={"input": "2 ? 3 ? 4 = 14"},
        ),
        TaskInstance(
            sample_id="ws",
            task_name="word_sorting",
            input={"words": ["Beta", "alpha", "alpha!"]},
        ),
        TaskInstance(
            sample_id="mmlu",
            task_name="mmlu_pro_engineering",
            input={"question": "Select the fixed option.", "options": ["first", "second", "third"]},
        ),
    ),
)
def test_common_task_spec_is_independent_of_hidden_gold(task: TaskInstance) -> None:
    with_hidden_gold = task.model_copy(
        update={"verifier_spec": {"target": "SECRET", "answer_index": 1, "answer_label": "B"}}
    )

    assert hashlib.sha256(_render(task).encode()).digest() == hashlib.sha256(
        _render(with_hidden_gold).encode()
    ).digest()


def test_mmlu_task_spec_labels_the_fixed_display_order() -> None:
    task = TaskInstance(
        sample_id="mmlu",
        task_name="mmlu_pro_physics",
        input={"question": "Select the fixed option.", "options": ["first", "second", "third"]},
    )

    option_lines = tuple(line for line in _render(task).splitlines() if line[:2] in {"A.", "B.", "C."})

    assert option_lines == ("A. first", "B. second", "C. third")


def test_retrieval_identity_serialization_remains_distinct_from_model_visible_task_spec() -> None:
    task = TaskInstance(sample_id="g24", task_name="game24", input={"numbers": [3, 3, 8, 8]})

    assert dispatch.canonical_task_json(task) != _render(task)


def test_out_of_scope_task_preserves_legacy_model_visible_serialization() -> None:
    task = TaskInstance(
        sample_id="gpqa",
        task_name="gpqa_diamond",
        input={"question": "Legacy question", "options": ["first", "second"]},
    )

    assert dispatch.render_model_visible_task(task) == dispatch.canonical_task_json(task)


def test_materialized_task_templates_bind_renderer_bytes() -> None:
    cases = (
        (
            TaskInstance(sample_id="g24", task_name="game24", input={"numbers": [3, 3, 8, 8]}),
            "game24_task_prompt_v1.txt",
            {"{numbers}": "[3,3,8,8]"},
        ),
        (
            TaskInstance(
                sample_id="meb",
                task_name="math_equation_balancer",
                input={"input": "2 ? 3 ? 4 = 14"},
            ),
            "meb_task_prompt_v1.txt",
            {"{operator_slot_equation}": "2 ? 3 ? 4 = 14"},
        ),
        (
            TaskInstance(
                sample_id="ws",
                task_name="word_sorting",
                input={"words": ["Beta", "alpha", "alpha!"]},
            ),
            "word_sorting_task_prompt_v1.txt",
            {"{words}": '["Beta","alpha","alpha!"]'},
        ),
    )

    for task, filename, replacements in cases:
        expected = (ARTIFACT_ROOT / filename).read_text(encoding="utf-8").removesuffix("\n")
        for placeholder, value in replacements.items():
            expected = expected.replace(placeholder, value)
        assert _render(task).encode() == expected.encode()


def test_materialized_contract_hashes_every_prompt_artifact() -> None:
    contract = json.loads(
        (ARTIFACT_ROOT / "common_task_spec_contract_v1.json").read_text(encoding="utf-8")
    )

    for binding in contract["templates"].values():
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
