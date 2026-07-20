from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from memcontam.baselines.contracts import NonEmptyStr
from memcontam.clients.base import LLMClient
from memcontam.memory.stores import MemoryEntry


_THOUGHT_DISTILL_INSTRUCTIONS = """\
Distill the solved task into one reusable thought template. Return only strict unfenced JSON with exactly
these fields: description, template, category, explicitly_used_memory_ids. category must be one of
prompt-based, procedure-based, programming-based. explicitly_used_memory_ids must list only visible memory
entry IDs actually used in the solution.
"""


class TemplateDistillationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: NonEmptyStr
    template: NonEmptyStr
    category: Literal["prompt-based", "procedure-based", "programming-based"]
    explicitly_used_memory_ids: tuple[str, ...]


@dataclass(frozen=True)
class BoTTemplatePayload:
    description: str
    template: str
    category: Literal["prompt-based", "procedure-based", "programming-based"]
    explicitly_used_memory_ids: tuple[str, ...]


def distill_thought_template(
    *,
    solution_trace: str,
    final_answer: str,
    visible_memory_ids: Sequence[str],
    client: LLMClient,
    model: str,
    config: dict[str, Any],
) -> BoTTemplatePayload:
    call_config = dict(config)
    call_config["method_stage"] = "bot_thought_distill"
    response = client.chat(
        [
            {"role": "system", "content": _THOUGHT_DISTILL_INSTRUCTIONS},
            {
                "role": "user",
                "content": f"Solution trace:\n{solution_trace}\n\nFinal answer:\n{final_answer}",
            },
        ],
        model,
        call_config,
    )
    try:
        result = TemplateDistillationResult.model_validate_json(response.content)
    except ValidationError as error:
        raise ValueError("malformed thought distillation") from error
    used_ids = validate_explicitly_used_memory_ids(
        result.explicitly_used_memory_ids, visible_memory_ids
    )
    return BoTTemplatePayload(
        description=result.description,
        template=result.template,
        category=result.category,
        explicitly_used_memory_ids=used_ids,
    )


def validate_explicitly_used_memory_ids(
    used_ids: Sequence[str], visible_memory_ids: Sequence[str]
) -> tuple[str, ...]:
    if len(used_ids) != len(set(used_ids)):
        raise ValueError("explicitly used memory IDs must be unique")
    visible_ids = set(visible_memory_ids)
    unknown_ids = [entry_id for entry_id in used_ids if entry_id not in visible_ids]
    if unknown_ids:
        raise ValueError(f"unknown explicitly used memory IDs: {unknown_ids}")
    return tuple(used_ids)


def build_template_entry(
    *,
    payload: BoTTemplatePayload,
    source_trial_id: str,
    visible_entry_ids: Sequence[str],
    clean_or_contaminated: Literal["clean", "contaminated"] = "clean",
) -> MemoryEntry:
    used_ids = validate_explicitly_used_memory_ids(
        payload.explicitly_used_memory_ids, visible_entry_ids
    )
    entry_id = "bot_template:" + hashlib.sha256(
        f"{source_trial_id}:{payload.description}:{payload.template}".encode("utf-8")
    ).hexdigest()[:12]
    used_id_list = list(used_ids)
    return MemoryEntry(
        entry_id=entry_id,
        content=payload.template,
        memory_type="thought_template",
        clean_or_contaminated=clean_or_contaminated,
        source_trial_id=source_trial_id,
        metadata={
            "description": payload.description,
            "category": payload.category,
            "explicitly_used_memory_ids": used_id_list,
            "declared_updater_context_ids": list(visible_entry_ids),
            "memory_support_ids": used_id_list,
            "direct_parent_ids": used_id_list,
            "parent_entry_ids": used_id_list,
            "source_entry_ids": used_id_list,
            "creation_origin": "thought_template",
        },
    )
