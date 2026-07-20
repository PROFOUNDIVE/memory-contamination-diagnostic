from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .bot_read import DistilledProblem, FALLBACK_THOUGHT_TEMPLATE
from memcontam.baselines.contracts import NonEmptyStr
from memcontam.logging.provenance import PromptSourcePart, build_prompt_with_sources
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


class BoTSolveResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solution_trace: NonEmptyStr
    final_answer: NonEmptyStr


def render_bot_solve_prompt(
    task: TaskInstance,
    problem: DistilledProblem,
    retrieved: dict[str, Any] | None,
) -> tuple[str, list[Any]]:
    prefix = (
        "Distilled problem JSON:\n"
        f"{json.dumps(problem.model_dump(), sort_keys=True, separators=(',', ':'))}\n\n"
        "Retrieved thought template:\n"
    )
    suffix = (
        "\n\nTask input:\n"
        f"{task.input}\n\n"
        "Return only strict unfenced JSON with exactly these non-empty string fields: "
        "solution_trace, final_answer."
    )
    entry = retrieved.get("memory_entry") if retrieved else None
    if isinstance(entry, MemoryEntry):
        return build_prompt_with_sources(
            [prefix, PromptSourcePart(f"entry_id={entry.entry_id}\n{entry.content}", entry), suffix],
            message_index=1,
        )
    return prefix + FALLBACK_THOUGHT_TEMPLATE + suffix, []


def parse_bot_solve_result(response: str) -> BoTSolveResult:
    try:
        return BoTSolveResult.model_validate_json(response)
    except ValidationError as error:
        raise ValueError("malformed bot solve result") from error
