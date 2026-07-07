from __future__ import annotations

from memcontam.clients.base import LLMResponse


class ReplayClient:
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["{}"]
        self.index = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        content = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return LLMResponse(content=content, raw={"replay": True, "messages": messages}, token_usage={})
