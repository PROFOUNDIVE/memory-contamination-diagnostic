from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from memcontam.baselines.bot_read import (
    COARSE_THOUGHT_STRUCTURES,
    BoTRetrievalDecision,
    DistilledProblem,
)
from memcontam.baselines.contracts import NonEmptyStr
from memcontam.logging.provenance import PromptSourcePart, build_prompt_with_sources
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


class BoTSolveResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_structure: Literal[
        "retrieved-template", "prompt-based", "procedure-based", "programming-based"
    ]
    solution_trace: NonEmptyStr
    final_answer: NonEmptyStr


def render_bot_solve_prompt(
    task: TaskInstance,
    problem: DistilledProblem,
    retrieval_decision: BoTRetrievalDecision,
    *,
    tool_mode: Literal["text_only", "python_sandbox"] = "text_only",
) -> tuple[str, list[Any]]:
    prefix = (
        "Distilled problem JSON:\n"
        f"{json.dumps(problem.model_dump(), sort_keys=True, separators=(',', ':'))}\n\n"
        "Reasoning structure:\n"
    )
    result_contract = (
        "Return only strict unfenced JSON with exactly these non-empty string fields: "
        "selected_structure, solution_trace, final_answer."
        if tool_mode == "text_only"
        else (
            "Use the Python sandbox only when execution would validate the solution. Return exactly one "
            'JSON action: {"action":"execute_python","code":"..."} or '
            '{"action":"final","answer":"<strict BoT solve JSON>"}. '
            "The final action answer must be strict unfenced JSON with exactly these non-empty string fields: "
            "selected_structure, solution_trace, final_answer."
        )
    )
    suffix = (
        "\n\nTask input:\n"
        f"{canonical_task_json(task)}\n\n"
        + result_contract
    )
    entry = retrieval_decision.matched_entry
    if retrieval_decision.decision == "matched" and isinstance(entry, MemoryEntry):
        selected = "Set selected_structure to retrieved-template.\n\n"
        return build_prompt_with_sources(
            [
                prefix,
                selected,
                PromptSourcePart(f"entry_id={entry.entry_id}\n{entry.content}", entry),
                suffix,
            ],
            message_index=1,
        )
    structures = "\n\n".join(
        f"{name}:\n{description}" for name, description in COARSE_THOUGHT_STRUCTURES
    )
    selection = (
        "Select exactly one coarse structure for selected_structure: "
        "prompt-based, procedure-based, or programming-based.\n\n"
    )
    return prefix + selection + structures + suffix, []


def render_tool_augmented_bot_solve_messages(
    task: TaskInstance,
    problem: DistilledProblem,
    retrieval_decision: BoTRetrievalDecision,
) -> tuple[list[dict[str, str]], list[Any]]:
    content, source_spans = render_bot_solve_prompt(
        task, problem, retrieval_decision, tool_mode="python_sandbox"
    )
    return (
        [
            {
                "role": "system",
                "content": "Apply the thought structure. You may use the Python sandbox for validation.",
            },
            {"role": "user", "content": content},
        ],
        source_spans,
    )


def parse_bot_solve_result(
    response: str, retrieval_decision: BoTRetrievalDecision | None = None
) -> BoTSolveResult:
    try:
        result = BoTSolveResult.model_validate_json(response)
    except ValidationError as error:
        raise ValueError("malformed bot solve result") from error
    if retrieval_decision is not None:
        _validate_selected_structure(result, retrieval_decision)
    return result


def _validate_selected_structure(
    result: BoTSolveResult, retrieval_decision: BoTRetrievalDecision
) -> None:
    if retrieval_decision.decision == "matched":
        if result.selected_structure != "retrieved-template":
            raise ValueError("matched BoT retrieval must select retrieved-template")
        return
    if result.selected_structure == "retrieved-template":
        raise ValueError("BoT retrieval miss must select a coarse structure")
