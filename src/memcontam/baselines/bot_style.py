from __future__ import annotations

import re
from typing import Any

from memcontam.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider, normalized_dot_top_k
from memcontam.memory.retrieval import RetrievedRecord, retrieve_records
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


_META_DISTILLER_INSTRUCTIONS = """\
As a highly professional and intelligent expert in information distillation, extract the essential information required to solve the problem from the user input query.

Please categorize and extract the crucial information required to solve the problem. The distilled information should include:

1. Key information: values and information of key variables extracted from user input.
2. Restriction: the objective of the problem and corresponding real-world constraints.
3. Distilled task: extend the problem based on the key information and restriction, propose a meta problem that can address the user query, and use the user query input key information as input to solve the problem as an example.
4. Python transformation: try to transform the problem into a Python algorithm problem and provide the input parameters.
5. Answer form: describe the exact output format required.

Important: your task is to distill the problem. Do not give the final result or a possible solution in your response.

Please distill the information following the format below:

Distilled Information:

1. Key information:

2. Restriction:

3. Distilled task:

4. Python transformation:

5. Answer form:
"""


_INSTANTIATION_INSTRUCTIONS = """\
You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a thought template. Your goal is to analyze the user's task and generate a specific solution based on the thought template.

If the instantiated solution involves Python code, provide the code in one fenced block and let the compiler handle it. Otherwise, provide a final answer that is easy to extract from the text.
"""


_SLOT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("key_information", re.compile(r"1\.\s*Key information\s*:?", re.IGNORECASE)),
    ("restriction", re.compile(r"2\.\s*Restriction\s*:?", re.IGNORECASE)),
    ("distilled_task", re.compile(r"3\.\s*Distilled task\s*:?", re.IGNORECASE)),
    ("python_transformation", re.compile(r"4\.\s*Python transformation\s*:?", re.IGNORECASE)),
    ("answer_form", re.compile(r"5\.\s*Answer form\s*:?", re.IGNORECASE)),
]


_SLOT_DISPLAY_NAMES: dict[str, str] = {
    "key_information": "Key information",
    "restriction": "Restriction",
    "distilled_task": "Distilled task",
    "python_transformation": "Python transformation",
    "answer_form": "Answer form",
}


def _build_meta_distiller_prompt(task: TaskInstance) -> str:
    return (
        _META_DISTILLER_INSTRUCTIONS
        + "\n\nUser input:\n"
        + str(task.input)
        + "\n\nProvide the distilled information in the requested format."
    )


def _parse_distilled_slots(text: str) -> dict[str, str]:
    positions: list[tuple[str, int, int]] = []
    for key, pattern in _SLOT_PATTERNS:
        match = pattern.search(text)
        if match is None:
            raise ValueError(
                f"malformed problem distillation: missing slot {_SLOT_DISPLAY_NAMES[key]!r}"
            )
        positions.append((key, match.start(), match.end()))

    positions.sort(key=lambda item: item[1])
    slots: dict[str, str] = {}
    for index, (key, _header_start, content_start) in enumerate(positions):
        end = positions[index + 1][1] if index + 1 < len(positions) else len(text)
        slots[key] = text[content_start:end].strip(": \n")
    return slots


def _retrieve_top1_template(
    query_text: str, entries: list[MemoryEntry], provider: EmbeddingProvider | None = None
) -> dict[str, Any] | None:
    if not entries:
        return None
    provider = provider or FakeEmbeddingProvider()
    query_vector = provider.encode_query(query_text)
    document_vectors = [provider.encode_document(entry.content) for entry in entries]
    document_ids = [entry.entry_id for entry in entries]
    top_k = normalized_dot_top_k(query_vector, document_vectors, document_ids, k=1)
    if not top_k:
        return None
    top_id, score = top_k[0]
    for entry in entries:
        if entry.entry_id == top_id:
            return {
                "entry_id": entry.entry_id,
                "content": entry.content,
                "score": score,
                "memory_entry": entry,
            }
    return None


def _build_instantiation_prompt(
    task: TaskInstance,
    distilled: dict[str, str],
    template: dict[str, Any] | None,
) -> str:
    if template is not None:
        template_section = (
            f"entry_id={template['entry_id']}\n{template['content']}"
        )
        instantiated = (
            "Apply the retrieved thought template to the key information above "
            "and derive the answer following the restriction and answer form."
        )
    else:
        template_section = (
            "No thought template has been retrieved. "
            "Solve the problem from the distilled information alone."
        )
        instantiated = (
            "No template is available; reason directly from the distilled problem."
        )

    return (
        "Distilled information:\n"
        "\n"
        "1. Key information:\n"
        f"{distilled['key_information']}\n"
        "\n"
        "2. Restriction:\n"
        f"{distilled['restriction']}\n"
        "\n"
        "3. Distilled task:\n"
        f"{distilled['distilled_task']}\n"
        "\n"
        "4. Python transformation:\n"
        f"{distilled['python_transformation']}\n"
        "\n"
        "5. Answer form:\n"
        f"{distilled['answer_form']}\n"
        "\n"
        "Retrieved thought template:\n"
        f"{template_section}\n"
        "\n"
        "Instantiated guidance:\n"
        f"{instantiated}\n"
        "\n"
        f"Solve: {task.input}"
    )


def distill_thought_template(
    task: TaskInstance,
    raw_response: str,
    verifier_result: Any,
    retrieved_template: dict[str, Any] | None,
) -> str:
    outcome = "validated" if verifier_result.is_correct else "attempted"
    prior = "with a retrieved prior template" if retrieved_template else "without a retrieved prior template"
    if task.task_name == "game24":
        definition = "Arithmetic target construction from a fixed multiset of numbers."
        relationships = "Track the target value, required numbers, and inverse operations that create useful intermediate values."
        strategy = "Build a compact expression by creating a denominator or subexpression that transforms one given number into the target."
        example = "For similar inputs, inspect fractions, complements, and parenthesized subexpressions before combining all numbers exactly once."
    elif task.task_name == "math_equation_balancer":
        definition = "Direct arithmetic evaluation under standard operator precedence."
        relationships = "Preserve the expression structure, reduce inner operations first, then combine terms in order."
        strategy = "Translate the expression into a deterministic calculation and return only the final value."
        example = "For similar inputs, identify precedence boundaries before simplifying the whole expression."
    elif task.task_name == "word_sorting":
        definition = "Lexicographic ordering of a fixed word list."
        relationships = "Each input word appears exactly once in the output; ordering is alphabetical."
        strategy = "Normalize the list as tokens, sort lexicographically, and emit the sorted sequence without adding commentary."
        example = "For similar inputs, compare words from left to right and keep duplicates only if present in the source list."
    else:
        definition = "Structured reasoning over the task input."
        relationships = "Identify the givens, constraints, and required output form before solving."
        strategy = "Convert the problem into reusable steps, solve those steps, then format the answer exactly as requested."
        example = "For similar inputs, reuse the constraint-first decomposition instead of copying a prior answer."

    return (
        f"### Problem Type: {task.task_name}\n\n"
        f"**Definition**: {definition}\n\n"
        f"**Quantitative Relationships**: {relationships}\n\n"
        f"**Solution Strategy**: {strategy}\n\n"
        f"**Example**: {example}\n\n"
        f"**Update Context**: This {outcome} solve was distilled {prior}."
    )


class BotStylePolicy:
    """BoT-style proxy baseline: distill, retrieve top-1 template, instantiate, solve.

    This is an adapted baseline, not an official Buffer-of-Thoughts reproduction.
    In replay mode distillation is deterministic and mirrors the structural slots
    of the official meta-distiller prompt.
    """

    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        distilled = self._distill(task)
        template = self._retrieve_template(task, memory)
        prompt = self._render_prompt(task, distilled, template)
        return [{"role": "user", "content": prompt}]

    def _distill(self, task: TaskInstance) -> dict[str, str]:
        task_name = task.task_name
        input_data = task.input

        if task_name == "game24":
            numbers = input_data.get("numbers", [])
            target = input_data.get("target", 24)
            return {
                "key_information": f"numbers = {numbers!r}, target = {target!r}",
                "restriction": (
                    "Use each given number exactly once. "
                    "Respect standard arithmetic operator precedence and use parentheses where needed. "
                    "The final expression must evaluate to the target value."
                ),
                "distilled_task": (
                    "Given a list of numbers and a target value, construct an arithmetic expression "
                    "that uses every number exactly once and evaluates to the target."
                ),
                "python_transformation": (
                    f"numbers = {numbers!r}\n"
                    f"target = {target!r}\n"
                    "expression = evaluate_arithmetic_expression(numbers)"
                ),
                "answer_form": (
                    "Output a single arithmetic expression as plain text, prefixed with 'final: '. "
                    "No extra explanation."
                ),
            }

        if task_name == "math_equation_balancer":
            expression = input_data.get("input", "")
            return {
                "key_information": f"expression = {expression!r}",
                "restriction": (
                    "Evaluate the expression using standard arithmetic rules and operator precedence. "
                    "Return only the final numeric result."
                ),
                "distilled_task": (
                    "Compute the value of the provided arithmetic expression."
                ),
                "python_transformation": (
                    f"expression = {expression!r}\n"
                    "result = evaluate_expression(expression)"
                ),
                "answer_form": (
                    "Output the final value, prefixed with 'final: '. No extra explanation."
                ),
            }

        if task_name == "word_sorting":
            words = input_data.get("words", [])
            return {
                "key_information": f"words = {words!r}",
                "restriction": (
                    "Sort the words alphabetically (lexicographic order). "
                    "Preserve each word exactly once and do not add new words."
                ),
                "distilled_task": (
                    "Given a list of words, return them sorted in alphabetical order."
                ),
                "python_transformation": (
                    f"words = {words!r}\n"
                    "sorted_words = sorted(words)"
                ),
                "answer_form": (
                    "Output the sorted words separated by a single space, prefixed with 'final: '. "
                    "No extra explanation."
                ),
            }

        items = [f"{key} = {value!r}" for key, value in input_data.items()]
        return {
            "key_information": "\n".join(items) if items else "(no input data)",
            "restriction": "Follow the real-world rules implied by the problem statement.",
            "distilled_task": "Solve the problem described by the user input.",
            "python_transformation": (
                "\n".join(f"{key} = {value!r}" for key, value in input_data.items())
                or "# no parameters"
            ),
            "answer_form": "Output the answer prefixed with 'final: '. No extra explanation.",
        }

    def _retrieve_template(self, task: TaskInstance, memory: MemoryState) -> RetrievedRecord | None:
        records = retrieve_records(str(task.input), memory.entries, k=1)
        return records[0] if records else None

    def _render_prompt(
        self, task: TaskInstance, distilled: dict[str, str], template: RetrievedRecord | None
    ) -> str:
        if template is not None:
            template_section = template["content"]
            instantiated = (
                "Apply the retrieved thought template to the key information above "
                "and derive the answer following the restriction and answer form."
            )
        else:
            template_section = (
                "No thought template has been retrieved. Solve the problem from the distilled information alone."
            )
            instantiated = (
                "No template is available; reason directly from the distilled problem."
            )

        return (
            "Distilled problem:\n"
            "\n"
            "1. Key information:\n"
            f"{distilled['key_information']}\n"
            "\n"
            "2. Restriction:\n"
            f"{distilled['restriction']}\n"
            "\n"
            "3. Distilled task:\n"
            f"{distilled['distilled_task']}\n"
            "\n"
            "4. Python transformation:\n"
            f"{distilled['python_transformation']}\n"
            "\n"
            "5. Answer form:\n"
            f"{distilled['answer_form']}\n"
            "\n"
            "Retrieved thought template:\n"
            f"{template_section}\n"
            "\n"
            "Instantiated guidance:\n"
            f"{instantiated}\n"
            "\n"
            f"Solve: {task.input}"
        )

    def problem_distillation(
        self,
        task: TaskInstance,
        client: Any,
        model: str,
        config: dict[str, Any],
    ) -> dict[str, str]:
        messages = [
            {"role": "system", "content": "You are an expert information distillation assistant."},
            {"role": "user", "content": _build_meta_distiller_prompt(task)},
        ]
        call_config = dict(config)
        call_config.setdefault("sample_id", task.sample_id)
        call_config["method_stage"] = "bot_problem_distill"
        response = client.chat(messages, model, call_config)
        return _parse_distilled_slots(response.content)

    def template_instantiation_solve(
        self,
        task: TaskInstance,
        distilled: dict[str, str],
        memory: MemoryState,
        client: Any,
        model: str,
        config: dict[str, Any],
        retrieved: dict[str, Any] | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": _INSTANTIATION_INSTRUCTIONS},
            {
                "role": "user",
                "content": _build_instantiation_prompt(task, distilled, retrieved),
            },
        ]
        call_config = dict(config)
        call_config.setdefault("sample_id", task.sample_id)
        call_config["method_stage"] = "bot_instantiate_solve"
        response = client.chat(messages, model, call_config)
        return response.content
