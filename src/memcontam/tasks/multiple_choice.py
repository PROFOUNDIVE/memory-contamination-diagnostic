from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memcontam.logging.schema import VerifierResult
from memcontam.tasks.base import TaskInstance


_TASKS = {"mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond"}
_FIELDS = {"sample_id", "task_name", "input", "verifier_spec", "metadata"}


def build_instance(row: Mapping[str, Any]) -> TaskInstance:
    if set(row) - _FIELDS:
        raise ValueError("multiple-choice row contains unsupported fields")
    if row.get("task_name") not in _TASKS:
        raise ValueError("multiple-choice row has unsupported task_name")
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"].strip():
        raise ValueError("multiple-choice row requires a non-empty sample_id")
    task_input = row.get("input")
    if not isinstance(task_input, Mapping):
        raise ValueError("multiple-choice row requires an input object")
    if set(task_input) != {"question", "options"}:
        raise ValueError("multiple-choice input must contain only question and options")
    question = task_input["question"]
    options = task_input["options"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("multiple-choice question must be non-empty")
    if (
        not isinstance(options, (list, tuple))
        or not 2 <= len(options) <= 26
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError("multiple-choice options must contain 2 to 26 non-empty strings")
    verifier = row.get("verifier_spec")
    if not isinstance(verifier, Mapping) or set(verifier) != {"answer_index", "answer_label"}:
        raise ValueError("multiple-choice verifier_spec is invalid")
    answer_index = verifier["answer_index"]
    answer_label = verifier["answer_label"]
    if (
        type(answer_index) is not int
        or not 0 <= answer_index < len(options)
        or answer_label != chr(65 + answer_index)
    ):
        raise ValueError("multiple-choice answer key is invalid")
    return TaskInstance.model_validate(row)


def verify_answer(answer: str, task: TaskInstance) -> VerifierResult:
    if not isinstance(answer, str):
        return VerifierResult(is_correct=False, reason="malformed_answer")
    normalized = answer.strip().upper()
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
    expected = task.verifier_spec["answer_label"]
    return VerifierResult(
        is_correct=normalized == expected,
        parsed_answer=normalized,
        metadata={"answer_index": task.verifier_spec["answer_index"]},
    )
