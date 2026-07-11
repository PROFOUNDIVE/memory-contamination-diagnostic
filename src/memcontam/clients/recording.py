from __future__ import annotations

from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.logging.schema import MethodCall


class MethodCallRecorder:
    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._records: list[MethodCall] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        stage = config.get("method_stage", "unknown")
        try:
            response = self._client.chat(messages, model, config)
        except Exception as exc:
            self._records.append(
                MethodCall(
                    stage=stage,
                    messages=messages,
                    raw_response="",
                    model=model,
                    temperature=config.get("temperature"),
                    top_p=config.get("top_p"),
                    max_tokens=config.get("max_tokens"),
                    latency_ms=None,
                    token_usage={},
                    retry_count=config.get("retry_count", 0),
                    error_type=type(exc).__name__,
                )
            )
            raise

        self._records.append(
            MethodCall(
                stage=stage,
                messages=messages,
                raw_response=response.content,
                model=model,
                temperature=config.get("temperature"),
                top_p=config.get("top_p"),
                max_tokens=config.get("max_tokens"),
                latency_ms=response.latency_ms,
                token_usage=response.token_usage,
                retry_count=config.get("retry_count", 0),
                error_type=None,
            )
        )
        return response

    def get_records(self) -> list[MethodCall]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()
