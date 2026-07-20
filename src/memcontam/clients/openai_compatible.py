from __future__ import annotations

import os
import time
from typing import Any, cast

from openai import OpenAI

from memcontam.clients.base import LLMResponse
from memcontam.clients.config import ProviderConfig


class OpenAICompatibleClient:
    def __init__(self, config: ProviderConfig):
        if config.provider != "openai_compatible":
            raise ValueError("OpenAICompatibleClient requires provider=openai_compatible")
        api_key_env = config.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key env var: {api_key_env}")
        options: dict[str, Any] = {"api_key": api_key, "base_url": config.base_url}
        if config.timeout_seconds is not None:
            options["timeout"] = config.timeout_seconds
        if config.max_retries is not None:
            options["max_retries"] = config.max_retries
        self.client = OpenAI(**options)

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=cast(Any, messages),
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
