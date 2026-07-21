from __future__ import annotations

import json
from typing import Any

from memcontam.baselines.bot_read import (
    BoTRetrievalDecision,
    DistilledProblem,
    distill_problem,
    retrieve_top_template,
)
from memcontam.baselines.bot_solve import parse_bot_solve_result, render_bot_solve_prompt
from memcontam.memory.embeddings import EmbeddingProvider
from memcontam.memory.stores import MemoryState
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

    def build_prompt(
        self, task: TaskInstance, memory: MemoryState, *, embedding_provider: EmbeddingProvider
    ) -> list[dict[str, str]]:
        problem = DistilledProblem(
            key_information=json.dumps(task.input, sort_keys=True),
            restrictions="Follow the task constraints.",
            distilled_task=f"Solve the {task.task_name} task.",
        )
        content, _ = render_bot_solve_prompt(
            task,
            problem,
            retrieve_top_template(problem, memory.entries, embedding_provider),
        )
        return [{"role": "user", "content": content}]

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
        retrieval_decision: BoTRetrievalDecision,
    ) -> str:
        content, source_spans = render_bot_solve_prompt(task, distilled, retrieval_decision)
        call_config = dict(config)
        call_config.setdefault("sample_id", task.sample_id)
        call_config["method_stage"] = "bot_instantiate_solve"
        call_config["_bot_retrieval_decision"] = retrieval_decision.decision
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
