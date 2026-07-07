from __future__ import annotations

import os
import time

from openai import OpenAI

from memcontam.clients.base import LLMResponse


class OpenAICompatibleClient:
    def __init__(self, base_url: str | None, api_key_env: str, api_key_default: str | None = None):
        api_key = os.environ.get(api_key_env, api_key_default)
        if api_key is None:
            raise RuntimeError(f"missing API key env var: {api_key_env}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config.get("temperature", 0),
            top_p=config.get("top_p", 1),
            max_tokens=config.get("max_tokens"),
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        message = response.choices[0].message
        usage = response.usage.model_dump() if response.usage else {}
        return LLMResponse(
            content=message.content or "",
            raw=response.model_dump(),
            token_usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
            latency_ms=latency_ms,
        )
