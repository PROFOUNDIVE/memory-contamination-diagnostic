from __future__ import annotations

from memcontam.clients.base import LLMResponse


class ReplayClient:
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["{}"]
        self.index = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        content = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        token_usage = config.get("token_usage") or {}
        normalized_token_usage = {
            "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(token_usage.get("completion_tokens", 0)),
            "total_tokens": int(token_usage.get("total_tokens", 0)),
        }
        latency_ms = config.get("latency_ms")
        if not isinstance(latency_ms, int) or latency_ms < 0:
            latency_ms = 0
        return LLMResponse(
            content=content,
            raw={"replay": True, "messages": messages},
            token_usage=normalized_token_usage,
            latency_ms=latency_ms,
        )
