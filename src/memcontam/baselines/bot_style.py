from __future__ import annotations

from typing import Any

from memcontam.baselines.bot_read import DistilledProblem, distill_problem
from memcontam.baselines.bot_solve import parse_bot_solve_result, render_bot_solve_prompt
from memcontam.tasks.base import TaskInstance


_INSTANTIATION_INSTRUCTIONS = """\
Apply the provided thought template to the distilled problem. Return only the requested JSON object.
"""


def distill_thought_template(
    task: TaskInstance,
    raw_response: str,
    verifier_result: Any,
    retrieved_template: dict[str, Any] | None,
) -> str:
    del task, verifier_result, retrieved_template
    return parse_bot_solve_result(raw_response).solution_trace


class BotStylePolicy:
    """BoT read-and-solve facade; thought-template writes are deferred to Task 8."""

    def problem_distillation(
        self,
        task: TaskInstance,
        client: Any,
        model: str,
        config: dict[str, Any],
    ) -> DistilledProblem:
        return distill_problem(task, client, model, config)

    def template_instantiation_solve(
        self,
        task: TaskInstance,
        distilled: DistilledProblem,
        client: Any,
        model: str,
        config: dict[str, Any],
        retrieved: dict[str, Any] | None = None,
    ) -> str:
        content, source_spans = render_bot_solve_prompt(task, distilled, retrieved)
        call_config = dict(config)
        call_config.setdefault("sample_id", task.sample_id)
        call_config["method_stage"] = "bot_instantiate_solve"
        call_config["source_spans"] = source_spans
        response = client.chat(
            [
                {"role": "system", "content": _INSTANTIATION_INSTRUCTIONS},
                {"role": "user", "content": content},
            ],
            model,
            call_config,
        )
        return response.content
