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
        """Return a chat completion response."""
