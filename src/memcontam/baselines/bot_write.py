from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from memcontam.baselines.bot_read import BoTRetrievalDecision, DistilledProblem
from memcontam.baselines.contracts import NonEmptyStr
from memcontam.clients.base import LLMClient
from memcontam.memory.stores import MemoryEntry


_THOUGHT_DISTILL_INSTRUCTIONS = """\
Distill the solved task into one reusable thought template. Return only strict unfenced JSON with exactly
these fields: description, template, category, explicitly_used_memory_ids. category must be one of
prompt-based, procedure-based, programming-based. explicitly_used_memory_ids must list only visible memory
entry IDs actually used in the solution. description must be a core-task summary. template must give a
reusable procedure and general answer form. Do not include an instance-specific final answer. Do not invent IDs;
when Visible memory JSON is [], explicitly_used_memory_ids must be [].
"""


class TemplateDistillationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: NonEmptyStr
    template: NonEmptyStr
    category: Literal["prompt-based", "procedure-based", "programming-based"]
    explicitly_used_memory_ids: tuple[str, ...]


class BoTToolContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class BoTTemplatePayload:
    description: str
    template: str
    category: Literal["prompt-based", "procedure-based", "programming-based"]
    explicitly_used_memory_ids: tuple[str, ...]


@dataclass(frozen=True)
class VisibleBoTMemory:
    entry_id: str
    description: str
    template: str


def distill_thought_template(
    *,
    canonical_task: str,
    distilled_problem: DistilledProblem,
    retrieval_decision: BoTRetrievalDecision,
    selected_structure: str,
    solution_trace: str,
    final_answer: str,
    visible_memory: Sequence[VisibleBoTMemory],
    client: LLMClient,
    model: str,
    config: dict[str, Any],
    executed_trajectory: Sequence[dict[str, Any]] = (),
    require_executed_programming: bool = False,
) -> BoTTemplatePayload:
    rendered_memory = tuple(visible_memory)
    if rendered_memory != visible_memory_for_retrieval_decision(retrieval_decision):
        raise ValueError("visible BoT memory must exactly match the retrieval decision")
    rendered_trajectory = [dict(item) for item in executed_trajectory]
    call_config = dict(config)
    call_config["method_stage"] = "bot_thought_distill"
    response = client.chat(
        [
            {"role": "system", "content": _THOUGHT_DISTILL_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"Canonical task JSON:\n{canonical_task}\n\n"
                    f"Distilled problem JSON:\n{json.dumps(distilled_problem.model_dump(), sort_keys=True, separators=(',', ':'))}\n\n"
                    f"Retrieval decision JSON:\n{render_retrieval_decision(retrieval_decision)}\n\n"
                    f"Selected reasoning structure:\n{selected_structure}\n\n"
                    f"Solution trace:\n{solution_trace}\n\n"
                    f"Executed trajectory JSON:\n{json.dumps(rendered_trajectory, sort_keys=True, separators=(',', ':'))}\n\n"
                    f"Final answer:\n{final_answer}\n\n"
                    f"Visible memory JSON:\n{render_visible_bot_memory(rendered_memory)}"
                ),
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
        result.explicitly_used_memory_ids, [entry.entry_id for entry in rendered_memory]
    )
    if (
        require_executed_programming
        and result.category == "programming-based"
        and not rendered_trajectory
    ):
        raise BoTToolContractError("BOT_UNEXECUTED_VALIDATION")
    return BoTTemplatePayload(
        description=result.description,
        template=result.template,
        category=result.category,
        explicitly_used_memory_ids=used_ids,
    )


def visible_memory_for_retrieval_decision(
    retrieval_decision: BoTRetrievalDecision,
) -> tuple[VisibleBoTMemory, ...]:
    if retrieval_decision.decision != "matched":
        return ()
    entry = retrieval_decision.matched_entry
    if entry is None:
        raise ValueError("matched BoT retrieval requires an entry")
    description = entry.metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("matched BoT retrieval requires an explicit description")
    return (VisibleBoTMemory(entry.entry_id, description, entry.content),)


def render_visible_bot_memory(visible_memory: Sequence[VisibleBoTMemory]) -> str:
    return json.dumps(
        [asdict(entry) for entry in visible_memory], sort_keys=True, separators=(",", ":")
    )


def render_retrieval_decision(retrieval_decision: BoTRetrievalDecision) -> str:
    return json.dumps(
        {"decision": retrieval_decision.decision}, sort_keys=True, separators=(",", ":")
    )


def validate_explicitly_used_memory_ids(
    used_ids: Sequence[str], rendered_memory_ids: Sequence[str]
) -> tuple[str, ...]:
    if len(used_ids) != len(set(used_ids)):
        raise ValueError("explicitly used memory IDs must be unique")
    visible_ids = set(rendered_memory_ids)
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
    entry_id = (
        "bot_template:"
        + hashlib.sha256(
            f"{source_trial_id}:{payload.description}:{payload.template}".encode("utf-8")
        ).hexdigest()[:12]
    )
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
