from __future__ import annotations

import json
from typing import Any, Final, Literal, Mapping, assert_never

from pydantic import TypeAdapter, ValidationError

from memcontam.tasks.base import TaskInstance


TaskSpecName = Literal[
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
]


class TaskSpecificationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_TASK_SPEC_NAME = TypeAdapter(TaskSpecName)
_TASK_SPEC_NAMES: Final = frozenset(
    {
        "game24",
        "math_equation_balancer",
        "word_sorting",
        "mmlu_pro_engineering",
        "mmlu_pro_physics",
    }
)


def canonical_task_json(task: TaskInstance | Mapping[str, Any]) -> str:
    payload = (
        task.model_dump(mode="json", exclude={"verifier_spec"})
        if isinstance(task, TaskInstance)
        else task
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_core_task_json(task: TaskInstance) -> str:
    payload = {"input": task.input, "task_name": task.task_name}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def render_model_visible_task(task: TaskInstance) -> str:
    if task.task_name not in _TASK_SPEC_NAMES:
        return canonical_task_json(task)
    return render_common_task_spec(task)


def render_common_task_spec(task: TaskInstance) -> str:
    try:
        task_name = _TASK_SPEC_NAME.validate_python(task.task_name, strict=True)
    except ValidationError as error:
        raise TaskSpecificationError("COMMON_TASK_SPEC_TASK_UNSUPPORTED") from error
    match task_name:
        case "game24":
            numbers = json.dumps(
                task.input["numbers"], ensure_ascii=False, separators=(",", ":")
            )
            return f"""Task family: Game24

Numbers: {numbers}
Target: 24

Construct one arithmetic expression whose value is exactly 24.
Use each supplied number occurrence exactly once.
Use only addition (+), subtraction (-), multiplication (*), division (/), and parentheses.
Do not concatenate numbers.
Do not use exponentiation, factorials, or any newly introduced numeric constant.
Non-integer rational intermediate values are allowed.

Answer payload: the arithmetic expression only."""
        case "math_equation_balancer":
            return f"""Task family: MathEquationBalancer

Equation template: {task.input["input"]}

Replace every question-mark (?) operator slot with exactly one of +, -, *, / so that the complete equation is true.
Keep every operand in exactly the supplied order.
Do not delete, duplicate, replace, or reorder an operand.
Do not add parentheses or another operation.
Evaluate multiplication and division before addition and subtraction, with left-to-right evaluation within each precedence level.
Use exact arithmetic.
Return the complete filled equation, including the equality sign and the supplied right-hand target.
Do not return only the numeric target.

Answer payload: the complete operator-filled equation only."""
        case "word_sorting":
            words = json.dumps(
                task.input["words"], ensure_ascii=False, separators=(",", ":")
            )
            return f"""Task family: WordSorting

Words: {words}

Sort the supplied word tokens in ascending lexicographic order.
Compare the supplied token text case-sensitively from left to right.
Keep punctuation as part of each token; do not remove or ignore it.
At the first differing character, the lexically earlier character comes first.
If one token is a strict prefix of another, place the shorter token first.
Preserve exactly the supplied token multiset and preserve each token's spelling.
Return the complete sorted token sequence with exactly one ASCII space between adjacent tokens and no leading or trailing whitespace.

Answer payload: the sorted token sequence only."""
        case "mmlu_pro_engineering" | "mmlu_pro_physics":
            task_family = (
                "MMLU-Pro Engineering"
                if task_name == "mmlu_pro_engineering"
                else "MMLU-Pro Physics"
            )
            options = "\n".join(
                f"{chr(65 + index)}. {option}"
                for index, option in enumerate(task.input["options"])
            )
            return f"""Task family: {task_family}

Question:
{task.input["question"]}

Displayed options:
{options}

Choose exactly one displayed option.
Answer payload: one uppercase displayed option label only."""
        case unreachable:
            assert_never(unreachable)
