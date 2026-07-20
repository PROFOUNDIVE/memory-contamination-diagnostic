from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from memcontam.baselines.contracts import NonEmptyStr
from memcontam.memory.embeddings import EmbeddingProvider, normalized_dot_top_k
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


FALLBACK_THOUGHT_TEMPLATE = (
    "Solve the task step by step. Use only justified transformations, obey all stated restrictions, "
    "and return exactly the required final-answer format."
)

_DISTILL_INSTRUCTIONS = """\
Extract the problem information required to solve the user task. Return only strict unfenced JSON with
exactly these non-empty string fields: key_information, restrictions, distilled_task. Do not solve it.
"""


class DistilledProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_information: NonEmptyStr
    restrictions: NonEmptyStr
    distilled_task: NonEmptyStr


def distill_problem(
    task: TaskInstance, client: Any, model: str, config: dict[str, Any]
) -> DistilledProblem:
    call_config = dict(config)
    call_config.setdefault("sample_id", task.sample_id)
    call_config["method_stage"] = "bot_problem_distill"
    response = client.chat(
        [
            {"role": "system", "content": "You are an expert information distillation assistant."},
            {"role": "user", "content": f"{_DISTILL_INSTRUCTIONS}\n\nUser input:\n{task.input}"},
        ],
        model,
        call_config,
    )
    try:
        return DistilledProblem.model_validate_json(response.content)
    except ValidationError as error:
        raise ValueError(f"malformed problem distillation: {error}") from error


def build_distilled_query(problem: DistilledProblem) -> str:
    return json.dumps(problem.model_dump(), sort_keys=True, separators=(",", ":"))


def retrieve_top_template(
    problem: DistilledProblem,
    entries: list[MemoryEntry],
    provider: EmbeddingProvider,
) -> dict[str, Any] | None:
    if not entries:
        return None
    query_vector = provider.encode_query(build_distilled_query(problem))
    descriptions = [
        entry.metadata.get("description", entry.content)
        if isinstance(entry.metadata.get("description", entry.content), str)
        else entry.content
        for entry in entries
    ]
    document_vectors = [provider.encode_document(description) for description in descriptions]
    top_matches = normalized_dot_top_k(
        query_vector, document_vectors, [entry.entry_id for entry in entries], k=1
    )
    if not top_matches:
        return None
    entry_id, score = top_matches[0]
    if score < 0.7:
        return None
    entry = next(entry for entry in entries if entry.entry_id == entry_id)
    return {"entry_id": entry.entry_id, "content": entry.content, "score": score, "memory_entry": entry}
