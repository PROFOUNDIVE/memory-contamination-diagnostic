from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class PromptBudgetSpec:
    context_window_tokens: int
    max_output_tokens: int
    fixed_prompt_overhead_tokens: int
    safety_margin_tokens: int

    def __post_init__(self) -> None:
        _require_positive("context_window_tokens", self.context_window_tokens)
        _require_positive("max_output_tokens", self.max_output_tokens)
        _require_nonnegative("fixed_prompt_overhead_tokens", self.fixed_prompt_overhead_tokens)
        _require_nonnegative("safety_margin_tokens", self.safety_margin_tokens)


def count_prompt_tokens(messages: Sequence[Mapping[str, str]], encoding_name: str) -> int:
    """Count role/content messages serialized as ``role\\ncontent`` blocks joined by blank lines."""
    blocks: list[str] = []
    for message in messages:
        try:
            role = message["role"]
            content = message["content"]
        except KeyError as exc:
            raise ValueError("each prompt message requires role and content") from exc
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("prompt message role and content must be strings")
        blocks.append(f"{role}\n{content}")
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode("\n\n".join(blocks), disallowed_special=()))


def effective_prompt_budget(spec: PromptBudgetSpec, current_task_tokens: int) -> int:
    """Return history capacity after reserving output, fixed overhead, safety, and the current task."""
    _require_nonnegative("current_task_tokens", current_task_tokens)
    budget = (
        spec.context_window_tokens
        - spec.max_output_tokens
        - spec.fixed_prompt_overhead_tokens
        - spec.safety_margin_tokens
        - current_task_tokens
    )
    if budget <= 0:
        raise ValueError("effective prompt budget must be positive")
    return budget


def _require_positive(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
