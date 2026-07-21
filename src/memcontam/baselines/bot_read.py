from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from memcontam.baselines.contracts import NonEmptyStr
from memcontam.memory.embeddings import EmbeddingProvider, normalized_dot_top_k
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


CoarseStructure = Literal["prompt-based", "procedure-based", "programming-based"]

COARSE_THOUGHT_STRUCTURES: tuple[tuple[CoarseStructure, str], ...] = (
    (
        "prompt-based",
        "Restate the objective and constraints, then reason through the required answer format.",
    ),
    (
        "procedure-based",
        "Decompose the task into justified ordered steps and verify each intermediate result.",
    ),
    (
        "programming-based",
        "Formulate a precise executable-style plan, then evaluate it against the stated constraints.",
    ),
)

RETRIEVAL_THRESHOLD = 0.7

_DISTILL_INSTRUCTIONS = """\
Extract the problem information required to solve the user task. Return only strict unfenced JSON with
exactly these non-empty string fields: key_information, restrictions, distilled_task. Do not solve it.
"""


class DistilledProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_information: NonEmptyStr
    restrictions: NonEmptyStr
    distilled_task: NonEmptyStr


@dataclass(frozen=True)
class BoTRetrievalDecision:
    decision: Literal["matched", "miss", "empty_buffer"]
    matched_entry: MemoryEntry | None
    top_similarity: float | None
    threshold: float


def distill_problem(
    task: TaskInstance, client: Any, model: str, config: dict[str, Any]
) -> DistilledProblem:
    call_config = dict(config)
    call_config.setdefault("sample_id", task.sample_id)
    call_config["method_stage"] = "bot_problem_distill"
    response = client.chat(
        [
            {"role": "system", "content": "You are an expert information distillation assistant."},
            {
                "role": "user",
                "content": f"{_DISTILL_INSTRUCTIONS}\n\nUser input:\n{canonical_task_json(task)}",
            },
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
    provider: EmbeddingProvider | None,
) -> BoTRetrievalDecision:
    if provider is None:
        raise ValueError("BoT retrieval requires an explicit embedding_provider")
    if not entries:
        return BoTRetrievalDecision("empty_buffer", None, None, RETRIEVAL_THRESHOLD)
    query_vector = provider.encode_query(build_distilled_query(problem))
    descriptions = [_template_description(entry) for entry in entries]
    document_vectors = [provider.encode_document(description) for description in descriptions]
    top_matches = normalized_dot_top_k(
        query_vector, document_vectors, [entry.entry_id for entry in entries], k=1
    )
    if not top_matches:
        return BoTRetrievalDecision("miss", None, None, RETRIEVAL_THRESHOLD)
    entry_id, score = top_matches[0]
    if score < RETRIEVAL_THRESHOLD:
        return BoTRetrievalDecision("miss", None, score, RETRIEVAL_THRESHOLD)
    entry = next(entry for entry in entries if entry.entry_id == entry_id)
    return BoTRetrievalDecision("matched", entry, score, RETRIEVAL_THRESHOLD)


def _template_description(entry: MemoryEntry) -> str:
    description = entry.metadata.get("description")
    category = entry.metadata.get("category")
    if not isinstance(description, str) or not description.strip() or not isinstance(category, str) or not category.strip():
        raise ValueError("V2 BoT templates require explicit description and category metadata")
    return description
