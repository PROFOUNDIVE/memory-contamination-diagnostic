from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    content: str
    raw: dict
    token_usage: dict[str, int]
    latency_ms: int | None = None


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        """Return a chat completion response.

        `config` is implementation-specific. The optional key `method_stage`
        may be used by replay/test clients to select a stage-specific
        response (e.g. ``rag_generate``, ``bot_problem_distill``,
        ``bot_instantiate_solve``, ``bot_thought_distill``,
        ``bot_novelty_decide``). Live clients should ignore it.
        """
        ...
