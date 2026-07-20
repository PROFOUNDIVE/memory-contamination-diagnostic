from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import (
    BaselineExecutionOutcome,
    ErrorType,
    FailureDisposition,
    ScientificIneligibilityReason,
)
from memcontam.baselines.full_history import FullHistoryPayload, FullHistoryState, render_full_history
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.provenance import PromptSourcePart, build_prompt_with_sources, phase11_lineage_metadata
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


class FullHistoryAdapter:
    def execute(
        self,
        task: TaskInstance,
        state: FullHistoryState,
        *,
        client: LLMClient,
        model: str,
        config: dict[str, Any] | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None = None,
    ) -> BaselineExecutionOutcome:
        config = dict(config or {})
        memory_before = tuple(record.model_dump() for record in state.records)
        messages, source_spans = _messages(task, state)
        trial_id = _trial_id(task, config, model)
        recorder = MethodCallRecorder(
            client,
            event_callback=config.get("_logging_event_callback"),
            trial_context={**config.get("_logging_trial_context", {}), "trial_id": trial_id},
        )
        try:
            response = recorder.chat(
                messages,
                model=model,
                config={
                    **config,
                    "sample_id": config.get("sample_id", task.sample_id),
                    "method_stage": "full_history_generate",
                    "source_spans": source_spans,
                },
            )
        except Exception:
            return _failed_outcome(
                recorder,
                memory_before,
                state,
                answer_call_id=_answer_call_id(recorder),
                error_type="ProviderCallFailure",
                failure_disposition="provider_call_failed",
                scientific_ineligibility_reason="provider_call_failed",
            )

        if response is None:
            return _failed_outcome(
                recorder,
                memory_before,
                state,
                answer_call_id=_answer_call_id(recorder),
                error_type="ProviderCallFailure",
                failure_disposition="provider_call_failed",
                scientific_ineligibility_reason="provider_call_failed",
            )

        entry = _append_response(task, state, response.content, config, model, trial_id)
        memory_write_event = {
            "type": "full_history_append",
            "status": "accepted",
            "new_entry_id": entry.entry_id,
            "source_trial_id": trial_id,
            "source_entry_ids": list(entry.metadata["source_entry_ids"]),
        }
        answer_call_id = _answer_call_id(recorder)
        try:
            parsed_answer = parse_final_answer(response.content)
        except (TypeError, ValueError):
            parsed_answer = ""
        if not parsed_answer:
            return _failed_outcome(
                recorder,
                memory_before,
                state,
                final_response=response.content,
                answer_call_id=answer_call_id,
                memory_write_event=memory_write_event,
                error_type="BaselineOutputError",
                failure_disposition="full_history_invalid_final_answer",
                scientific_ineligibility_reason="invalid_final_answer",
            )

        try:
            verifier_result = _verify(verifier, parsed_answer, task)
        except Exception:
            return _failed_outcome(
                recorder,
                memory_before,
                state,
                final_response=response.content,
                parsed_answer=parsed_answer,
                answer_call_id=answer_call_id,
                memory_write_event=memory_write_event,
                error_type="VerifierContractError",
                failure_disposition="verifier_contract_failed",
                scientific_ineligibility_reason="verifier_contract_failed",
            )

        return BaselineExecutionOutcome(
            status="succeeded",
            final_response=response.content,
            parsed_answer=parsed_answer,
            verifier_result=verifier_result,
            answer_call_id=answer_call_id,
            method_calls=tuple(recorder.get_records()),
            memory_before=memory_before,
            memory_after=tuple(record.model_dump() for record in state.records),
            memory_write_event=memory_write_event,
            metadata={},
        )


def _messages(task: TaskInstance, state: FullHistoryState) -> tuple[list[dict[str, str]], list[Any]]:
    parts: list[str | PromptSourcePart] = []
    for index, record in enumerate(state.records):
        if index:
            parts.append("\n\n")
        parts.append(PromptSourcePart(record.content, record))
    if state.records:
        parts.append("\n\n")
    parts.append(f"TASK:\n{task.input}")
    content, spans = build_prompt_with_sources(parts, message_index=0, entries=state.records)
    return [{"role": "user", "content": content}], spans


def _append_response(
    task: TaskInstance,
    state: FullHistoryState,
    raw_response: str,
    config: dict[str, Any],
    model: str,
    trial_id: str,
) -> MemoryEntry:
    source_entry_ids = [record.entry_id for record in state.records]
    entry_id = f"full_history:{task.task_name}:{task.sample_id}:{uuid4().hex}"
    entry = MemoryEntry(
        entry_id=entry_id,
        content=render_full_history(entry_id, FullHistoryPayload(str(task.input), raw_response)),
        memory_type="full_history_transcript",
        clean_or_contaminated=(
            "contaminated"
            if any(record.clean_or_contaminated == "contaminated" for record in state.records)
            else "clean"
        ),
        source_trial_id=trial_id,
        metadata={"source_entry_ids": source_entry_ids},
    )
    if config.get("_logging_target_contamination_set") or config.get("_logging_target_set_id"):
        entry.metadata.update(
            phase11_lineage_metadata(
                entry,
                [*state.records, entry],
                config.get("_logging_target_contamination_set")
                or config.get("_logging_target_set_id"),
            )
        )
    state.records.append(entry)
    return entry


def _failed_outcome(
    recorder: MethodCallRecorder,
    memory_before: tuple[dict[str, Any], ...],
    state: FullHistoryState,
    *,
    error_type: ErrorType,
    failure_disposition: FailureDisposition,
    scientific_ineligibility_reason: ScientificIneligibilityReason,
    final_response: str | None = None,
    parsed_answer: str | None = None,
    answer_call_id: str | None = None,
    memory_write_event: dict[str, Any] | None = None,
) -> BaselineExecutionOutcome:
    return BaselineExecutionOutcome(
        status="failed",
        final_response=final_response,
        parsed_answer=parsed_answer,
        answer_call_id=answer_call_id,
        method_calls=tuple(recorder.get_records()),
        memory_before=memory_before,
        memory_after=tuple(record.model_dump() for record in state.records),
        memory_write_event=memory_write_event,
        error_type=error_type,
        failure_disposition=failure_disposition,
        scientific_ineligibility_reason=scientific_ineligibility_reason,
        metadata={},
    )


def _verify(
    verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None,
    parsed_answer: str,
    task: TaskInstance,
) -> bool:
    if verifier is None:
        return True
    result = verifier(parsed_answer, task)
    if isinstance(result, VerifierResult):
        return result.is_correct
    if isinstance(result, bool):
        return result
    raise TypeError("full history verifier must return VerifierResult or bool")


def _trial_id(task: TaskInstance, config: dict[str, Any], model: str) -> str:
    return ":".join(
        [
            str(config.get("run_id", "unknown")),
            task.task_name,
            task.sample_id,
            str(config.get("baseline", "full_history")),
            str(config.get("arm", "clean")),
            str(config.get("model", model)),
        ]
    )


def _answer_call_id(recorder: MethodCallRecorder) -> str | None:
    records = recorder.get_records()
    return records[0].call_id if records else None
