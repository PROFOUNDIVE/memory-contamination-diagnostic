from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memcontam.baselines.prompt_budget import (
    PromptBudgetSpec,
    count_prompt_tokens,
    effective_prompt_budget,
)
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


@dataclass(frozen=True)
class FullHistoryContextDecision:
    messages: list[dict[str, str]]
    records: list[MemoryEntry]
    pre_record_ids: list[str]
    post_record_ids: list[str]
    removed_record_ids: list[str]
    pre_token_count: int | None
    post_token_count: int | None
    current_task_token_count: int | None
    effective_prompt_budget_tokens: int | None

    def telemetry(self) -> dict[str, Any]:
        return {
            "pre_record_ids": self.pre_record_ids,
            "post_record_ids": self.post_record_ids,
            "removed_record_ids": self.removed_record_ids,
            "pre_token_count": self.pre_token_count,
            "post_token_count": self.post_token_count,
            "current_task_token_count": self.current_task_token_count,
            "effective_prompt_budget_tokens": self.effective_prompt_budget_tokens,
        }


def select_visible_history(
    records: Sequence[MemoryEntry],
    task_text: str,
    config: Mapping[str, Any] | None,
) -> FullHistoryContextDecision:
    visible_records = list(records)
    pre_record_ids = [record.entry_id for record in records]
    messages = _messages(visible_records, task_text)
    if not config or config.get("mode") != "context_bounded_pair_atomic":
        return FullHistoryContextDecision(
            messages=messages,
            records=visible_records,
            pre_record_ids=pre_record_ids,
            post_record_ids=pre_record_ids,
            removed_record_ids=[],
            pre_token_count=None,
            post_token_count=None,
            current_task_token_count=None,
            effective_prompt_budget_tokens=None,
        )

    spec = PromptBudgetSpec(
        context_window_tokens=config["context_window_tokens"],
        max_output_tokens=config["max_output_tokens"],
        fixed_prompt_overhead_tokens=config["fixed_prompt_overhead_tokens"],
        safety_margin_tokens=config["safety_margin_tokens"],
    )
    encoding_name = config["token_encoding"]
    current_task_tokens = count_prompt_tokens(_messages([], task_text), encoding_name)
    history_budget = effective_prompt_budget(spec, current_task_tokens)
    total_budget = current_task_tokens + history_budget
    pre_token_count = count_prompt_tokens(messages, encoding_name)
    while visible_records and count_prompt_tokens(messages, encoding_name) > total_budget:
        visible_records.pop(0)
        messages = _messages(visible_records, task_text)
    post_record_ids = [record.entry_id for record in visible_records]
    return FullHistoryContextDecision(
        messages=messages,
        records=visible_records,
        pre_record_ids=pre_record_ids,
        post_record_ids=post_record_ids,
        removed_record_ids=pre_record_ids[: len(pre_record_ids) - len(post_record_ids)],
        pre_token_count=pre_token_count,
        post_token_count=count_prompt_tokens(messages, encoding_name),
        current_task_token_count=current_task_tokens,
        effective_prompt_budget_tokens=history_budget,
    )


def render_context_bounded_history(
    task: TaskInstance,
    records: Sequence[MemoryEntry],
    config: Mapping[str, Any] | None,
) -> FullHistoryContextDecision:
    return select_visible_history(records, canonical_task_json(task), config)


def _messages(records: Sequence[MemoryEntry], task_text: str) -> list[dict[str, str]]:
    history = "\n\n".join(record.content for record in records)
    prefix = f"{history}\n\n" if history else ""
    return [{"role": "user", "content": f"{prefix}TASK:\n{task_text}"}]
