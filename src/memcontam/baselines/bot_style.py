from __future__ import annotations

from memcontam.memory.retrieval import RetrievedRecord, retrieve_records
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


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
