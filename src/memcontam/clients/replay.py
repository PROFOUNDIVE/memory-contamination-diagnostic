from __future__ import annotations

from typing import Any

from memcontam.clients.base import LLMResponse


class ReplayClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        responses_by_sample: dict[str, Any] | None = None,
    ):
        self.responses = responses or ["{}"]
        self.responses_by_sample = responses_by_sample or {}
        self.index = 0
        self._stage_indices: dict[tuple[str, str], int] = {}

    def _next_flat_response(self) -> str:
        content = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return content

    def _next_stage_response(self, sample_id: str, stage: str) -> str:
        sample_stages = self.responses_by_sample[sample_id]
        if stage not in sample_stages:
            raise ValueError(
                f"missing replay response for sample {sample_id!r} stage {stage!r}"
            )
        stage_response = sample_stages[stage]
        if isinstance(stage_response, list):
            key = (sample_id, stage)
            idx = self._stage_indices.get(key, 0)
            if idx >= len(stage_response):
                raise ValueError(
                    f"exhausted replay responses for sample {sample_id!r} stage {stage!r}"
                )
            content = stage_response[idx]
            self._stage_indices[key] = idx + 1
        else:
            content = stage_response
        return content

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        stage = config.get("method_stage")
        sample_id = config.get("sample_id")
        if stage and sample_id and sample_id in self.responses_by_sample:
            sample_response = self.responses_by_sample[sample_id]
            if isinstance(sample_response, dict):
                content = self._next_stage_response(sample_id, stage)
            elif config.get("_require_stage_keyed_replay"):
                raise ValueError(
                    f"native replay requires stage-keyed responses for sample {sample_id!r}"
                )
            elif isinstance(sample_response, str):
                content = sample_response
            else:
                raise ValueError(f"invalid replay response for sample {sample_id!r}")
        else:
            content = self._next_flat_response()
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
