from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Callable

from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.logging.schema import CallEvent, MethodCall, PromptSourceSpan


class MethodCallRecorder:
    def __init__(
        self,
        client: LLMClient,
        event_callback: Callable[[CallEvent], None] | None = None,
        trial_context: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._event_callback = event_callback
        self._trial_context = trial_context or {}
        self._records: list[MethodCall] = []
        self._call_indices: dict[str, int] = {}

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        stage = config.get("method_stage", "unknown")
        trial_id = self._trial_context.get("trial_id", "unknown")
        call_index = self._next_call_index(trial_id)
        call_id = f"{trial_id}:call:{call_index}"

        decoding_params = {
            key: config[key] for key in ("temperature", "top_p", "max_tokens") if key in config
        }
        retry_count = config.get("retry_count", 0)
        source_spans = self._normalize_source_spans(config.get("source_spans", []))

        call_event = CallEvent(
            call_id=call_id,
            method_stage=stage,
            run_metadata_id=self._trial_context.get("run_metadata_id", ""),
            run_id=self._trial_context.get("run_id", ""),
            trial_id=trial_id,
            trial_seq=self._trial_context.get("trial_seq", 0),
            event_seq=0,
            stage=self._trial_context.get("stage", stage),
            messages=messages,
            model=model,
            decoding_params=decoding_params,
            response_text=None,
            token_usage={},
            latency_ms=None,
            retry_count=retry_count,
            source_spans=source_spans,
            created_at=_timestamp(),
        )

        try:
            response = self._client.chat(
                messages,
                model,
                {key: value for key, value in config.items() if not key.startswith("_logging_")},
            )
        except Exception as exc:
            failure = self._capture_failure()
            call_event = call_event.model_copy(
                update={
                    "response_text": None,
                    "token_usage": {},
                    "latency_ms": None,
                    "error_type": type(exc).__name__,
                    "failure_function": failure["function"],
                    "failure_module": failure["module"],
                    "failure_line": failure["line"],
                    "origin": "provider_call",
                }
            )
            method_call = self._call_event_to_method_call(call_event, stage)
            self._records.append(method_call)
            if self._event_callback is not None:
                self._event_callback(call_event)
            raise

        call_event = call_event.model_copy(
            update={
                "response_text": response.content,
                "token_usage": response.token_usage,
                "latency_ms": response.latency_ms,
            }
        )
        method_call = self._call_event_to_method_call(call_event, stage)
        self._records.append(method_call)
        if self._event_callback is not None:
            self._event_callback(call_event)
        return response

    def get_records(self) -> list[MethodCall]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()
        self._call_indices.clear()

    def _next_call_index(self, trial_id: str) -> int:
        self._call_indices[trial_id] = self._call_indices.get(trial_id, 0) + 1
        return self._call_indices[trial_id]

    def _normalize_source_spans(self, spans: Any) -> list[PromptSourceSpan]:
        if not isinstance(spans, list):
            return []
        result: list[PromptSourceSpan] = []
        for span in spans:
            if isinstance(span, PromptSourceSpan):
                result.append(span)
            elif isinstance(span, dict):
                result.append(PromptSourceSpan(**span))
        return result

    def _capture_failure(self) -> dict[str, Any]:
        frame = inspect.currentframe()
        if frame is None:
            return {"function": None, "module": None, "line": None}
        caller = frame.f_back
        del frame
        if caller is None:
            return {"function": None, "module": None, "line": None}
        try:
            code = caller.f_code
            return {
                "function": code.co_name,
                "module": caller.f_globals.get("__name__"),
                "line": caller.f_lineno,
            }
        finally:
            del caller

    def _call_event_to_method_call(self, event: CallEvent, method_stage: str) -> MethodCall:
        return MethodCall(
            call_id=event.call_id,
            stage=event.method_stage if event.method_stage != "unknown" else method_stage,
            messages=event.messages,
            raw_response=event.response_text,
            model=event.model,
            temperature=event.decoding_params.get("temperature"),
            top_p=event.decoding_params.get("top_p"),
            max_tokens=event.decoding_params.get("max_tokens"),
            latency_ms=event.latency_ms,
            token_usage=event.token_usage,
            retry_count=event.retry_count,
            error_type=event.error_type,
            source_spans=event.source_spans,
        )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def summarize_calls(calls: list[CallEvent]) -> dict[str, Any]:
    total_latency = 0
    token_usage: dict[str, int] = {}
    max_retry = 0
    for call in calls:
        if call.latency_ms is not None:
            total_latency += call.latency_ms
        for key, value in call.token_usage.items():
            token_usage[key] = token_usage.get(key, 0) + value
        if call.retry_count > max_retry:
            max_retry = call.retry_count
    return {
        "latency_ms": total_latency,
        "token_usage": token_usage,
        "retry_count": max_retry,
    }
