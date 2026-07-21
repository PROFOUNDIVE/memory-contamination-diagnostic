from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import (
    BaselineExecutionOutcome,
    ErrorType,
    FailureDisposition,
    NonEmptyStr,
    ReflexionAttemptOutcome,
    ReflexionReflectionEvent,
    ScientificIneligibilityReason,
)
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.provenance import (
    PromptSourcePart,
    build_prompt_with_sources,
    combine_lineage_status,
    derived_source_span,
    phase11_lineage_metadata,
    source_lineage_from_spans,
)
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


@dataclass
class ReflexionState:
    reflections: list[MemoryEntry] = field(default_factory=list)


def visible_reflections(state: ReflexionState) -> list[MemoryEntry]:
    return [entry for entry in state.reflections if entry.memory_type == "verbal_reflection"][-3:]


@dataclass(frozen=True)
class ReflectionPayload:
    reflection_text: str
    explicitly_used_memory_ids: tuple[str, ...]


class ReflectionGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["corrective"]
    failure_class: Literal["incorrect_answer"]
    reflection_text: NonEmptyStr
    explicitly_used_memory_ids: tuple[str, ...]


def record_attempt_outcome(
    attempts: list[ReflexionAttemptOutcome],
    *,
    attempt_id: str,
    attempt_index: int,
    answer_call_id: str,
    verifier_result: bool,
) -> None:
    attempts.append(
        ReflexionAttemptOutcome(
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            answer_call_id=answer_call_id,
            outcome=BaselineExecutionOutcome(status="succeeded", verifier_result=verifier_result),
        )
    )


def record_reflection_event(
    events: list[ReflexionReflectionEvent],
    *,
    attempt_id: str,
    reflection_call_id: str,
    reflection_entry_id: str,
) -> None:
    events.append(ReflexionReflectionEvent(attempt_id, reflection_call_id, reflection_entry_id))


class ReflexionAdapter:
    def execute(
        self,
        task: TaskInstance,
        state: ReflexionState,
        *,
        client: LLMClient,
        model: str,
        config: dict[str, Any] | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult | bool] | None = None,
    ) -> BaselineExecutionOutcome:
        config = dict(config or {})
        max_attempts = config.get("max_attempts", 2)
        if type(max_attempts) is not int or max_attempts not in {1, 2}:
            raise ValueError("reflexion max_attempts must be 1 or 2")
        memory_before = tuple(entry.model_dump() for entry in state.reflections)
        trial_id = _trial_id(task, config, model)
        recorder = MethodCallRecorder(
            client,
            event_callback=config.get("_logging_event_callback"),
            trial_context={**config.get("_logging_trial_context", {}), "trial_id": trial_id},
        )
        attempts: list[ReflexionAttemptOutcome] = []
        reflection_events: list[ReflexionReflectionEvent] = []

        for attempt_index in range(1, max_attempts + 1):
            visible_entries = visible_reflections(state)
            messages, source_spans = _generation_messages(task, visible_entries)
            try:
                response = recorder.chat(
                    messages,
                    model=model,
                    config={
                        **config,
                        "sample_id": config.get("sample_id", task.sample_id),
                        "method_stage": "reflexion_generate",
                        "source_spans": source_spans,
                    },
                )
            except Exception:
                return _failed_outcome(
                    recorder,
                    memory_before,
                    state,
                    attempts,
                    reflection_events,
                    "provider_call_failed",
                    answer_call_id=_answer_call_id(recorder),
                )

            answer_call_id = recorder.get_records()[-1].call_id
            if answer_call_id is None:
                raise AssertionError("recorded generation call must have an ID")
            parsed_answer = _parse_generation(response.content)
            if parsed_answer is None:
                return _failed_outcome(
                    recorder,
                    memory_before,
                    state,
                    attempts,
                    reflection_events,
                    "reflexion_invalid_generation",
                    final_response=response.content,
                    answer_call_id=answer_call_id,
                )
            try:
                verifier_result = _verify(verifier, parsed_answer, task)
            except Exception:
                return _failed_outcome(
                    recorder,
                    memory_before,
                    state,
                    attempts,
                    reflection_events,
                    "verifier_contract_failed",
                    final_response=response.content,
                    parsed_answer=parsed_answer,
                    answer_call_id=answer_call_id,
                )

            attempt_id = f"{trial_id}:attempt:{attempt_index}"
            record_attempt_outcome(
                attempts,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
                answer_call_id=answer_call_id,
                verifier_result=verifier_result,
            )
            if verifier_result:
                return _succeeded_outcome(
                    recorder,
                    memory_before,
                    state,
                    attempts,
                    reflection_events,
                    final_response=response.content,
                    parsed_answer=parsed_answer,
                    verifier_result=True,
                    answer_call_id=answer_call_id,
                )

            try:
                reflection_messages, reflection_spans = _reflection_messages(
                    task,
                    visible_entries,
                    response.content,
                    parsed_answer,
                    failed_actor_call_id=answer_call_id,
                    failed_actor_spans=source_spans,
                    target_set=config.get("_logging_target_contamination_set")
                    or config.get("_logging_target_set_id"),
                )
                reflection_response = recorder.chat(
                    reflection_messages,
                    model=model,
                    config={
                        **config,
                        "sample_id": config.get("sample_id", task.sample_id),
                        "method_stage": "reflexion_reflect",
                        "source_spans": reflection_spans,
                    },
                )
            except Exception:
                return _failed_outcome(
                    recorder,
                    memory_before,
                    state,
                    attempts,
                    reflection_events,
                    "provider_call_failed",
                    final_response=response.content,
                    parsed_answer=parsed_answer,
                    answer_call_id=answer_call_id,
                )
            payload = _parse_reflection(reflection_response.content, visible_entries)
            if payload is None:
                return _failed_outcome(
                    recorder,
                    memory_before,
                    state,
                    attempts,
                    reflection_events,
                    "reflexion_invalid_reflection",
                    final_response=response.content,
                    parsed_answer=parsed_answer,
                    answer_call_id=answer_call_id,
                )
            reflection_call_id = recorder.get_records()[-1].call_id
            if reflection_call_id is None:
                raise AssertionError("recorded reflection call must have an ID")
            entry = _append_reflection(
                task,
                state,
                payload,
                trial_id,
                reflection_call_id,
                answer_call_id,
                visible_entries,
                config,
            )
            record_reflection_event(
                reflection_events,
                attempt_id=attempt_id,
                reflection_call_id=reflection_call_id,
                reflection_entry_id=entry.entry_id,
            )
            if attempt_index == max_attempts:
                return _succeeded_outcome(
                    recorder,
                    memory_before,
                    state,
                    attempts,
                    reflection_events,
                    final_response=response.content,
                    parsed_answer=parsed_answer,
                    verifier_result=False,
                    answer_call_id=answer_call_id,
                )

        raise AssertionError("bounded Reflexion loop must return")


def _parse_generation(response: str) -> str | None:
    try:
        return parse_final_answer(response)
    except ValueError:
        return None


def _parse_reflection(response: str, visible_entries: list[MemoryEntry]) -> ReflectionPayload | None:
    try:
        result = ReflectionGenerationResult.model_validate_json(response)
    except (ValidationError, ValueError, json.JSONDecodeError):
        return None
    used_ids = result.explicitly_used_memory_ids
    visible_ids = {entry.entry_id for entry in visible_entries}
    if len(used_ids) != len(set(used_ids)) or not set(used_ids).issubset(visible_ids):
        return None
    return ReflectionPayload(result.reflection_text, used_ids)


def _generation_messages(
    task: TaskInstance, reflections: list[MemoryEntry]
) -> tuple[list[dict[str, str]], list[Any]]:
    parts: list[str | PromptSourcePart] = [f"Task: {task.task_name}\n\nReflections:\n"]
    if reflections:
        for index, entry in enumerate(reflections):
            if index:
                parts.append("\n")
            parts.append(PromptSourcePart(_render_reflection(entry), entry))
    else:
        parts.append("(none)")
    parts.append(f"\n\nCurrent task:\n{canonical_task_json(task)}")
    content, spans = build_prompt_with_sources(parts, message_index=1)
    return [
        {
            "role": "system",
            "content": (
                f"Solve the {task.task_name} task. Use only the listed reflections when useful. "
                "Return exactly one non-empty line: final: <answer>."
            ),
        },
        {"role": "user", "content": content},
    ], spans


def _reflection_messages(
    task: TaskInstance,
    reflections: list[MemoryEntry],
    response: str,
    parsed_answer: str,
    *,
    failed_actor_call_id: str,
    failed_actor_spans: list[Any],
    target_set: Any,
) -> tuple[list[dict[str, str]], list[Any]]:
    parts: list[str | PromptSourcePart] = [f"Task: {task.task_name}\n\nVisible reflections:\n"]
    if reflections:
        for index, entry in enumerate(reflections):
            if index:
                parts.append("\n")
            parts.append(PromptSourcePart(_render_reflection(entry), entry))
    else:
        parts.append("(none)")
    trajectory_prefix = f"\n\nCurrent task:\n{canonical_task_json(task)}\n\nFailed actor response:\n"
    suffix = f"\n\nParsed answer:\n{parsed_answer}\n\nFailure class:\nincorrect_answer"
    parts.append(trajectory_prefix)
    content, spans = build_prompt_with_sources(parts, message_index=1)
    trajectory_start = len(content)
    content += response + suffix
    spans.append(
        _failed_actor_trajectory_span(
            response,
            message_index=1,
            start=trajectory_start,
            end=trajectory_start + len(response),
            failed_actor_call_id=failed_actor_call_id,
            failed_actor_spans=failed_actor_spans,
            target_set=target_set,
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "Diagnose the failed actor attempt. Return only JSON matching "
                "ReflectionGenerationResult. Set failure_class to incorrect_answer and cite only "
                "reflection IDs shown."
            ),
        },
        {
            "role": "user",
            "content": content,
        },
    ], spans


def _append_reflection(
    task: TaskInstance,
    state: ReflexionState,
    payload: ReflectionPayload,
    trial_id: str,
    reflection_call_id: str,
    failed_actor_call_id: str,
    visible_entries: list[MemoryEntry],
    config: dict[str, Any],
) -> MemoryEntry:
    used_ids = list(payload.explicitly_used_memory_ids)
    entries_by_id = {entry.entry_id: entry for entry in visible_entries}
    used_entries = [entries_by_id[entry_id] for entry_id in used_ids]
    contaminated_source_ids = _contaminated_source_entry_ids(used_entries)
    entry = MemoryEntry(
        entry_id=f"reflexion:{task.task_name}:{task.sample_id}:{uuid4().hex}",
        content=f"Reflection: {payload.reflection_text}",
        memory_type="verbal_reflection",
        clean_or_contaminated="contaminated" if contaminated_source_ids else "clean",
        source_trial_id=trial_id,
        metadata={
            "parent_entry_ids": used_ids,
            "direct_parent_ids": used_ids,
            "memory_support_ids": used_ids,
            "source_entry_ids": used_ids,
            "source_contaminated_entry_ids": contaminated_source_ids,
            "creation_call_id": reflection_call_id,
            "failed_actor_call_id": failed_actor_call_id,
            "parent_call_ids": [failed_actor_call_id],
            "declared_updater_context_ids": [entry.entry_id for entry in visible_entries],
            "parent_call_id": reflection_call_id,
            "reflection_lineage": {"stage": "reflexion_reflect", "source_trial_id": trial_id},
        },
    )
    target_set = config.get("_logging_target_contamination_set") or config.get("_logging_target_set_id")
    if target_set:
        entry.metadata.update(phase11_lineage_metadata(entry, [*state.reflections, entry], target_set))
    state.reflections.append(entry)
    return entry


def _succeeded_outcome(
    recorder: MethodCallRecorder,
    memory_before: tuple[dict[str, Any], ...],
    state: ReflexionState,
    attempts: list[ReflexionAttemptOutcome],
    reflection_events: list[ReflexionReflectionEvent],
    *,
    final_response: str,
    parsed_answer: str,
    verifier_result: bool,
    answer_call_id: str,
) -> BaselineExecutionOutcome:
    return BaselineExecutionOutcome(
        status="succeeded",
        final_response=final_response,
        parsed_answer=parsed_answer,
        verifier_result=verifier_result,
        answer_call_id=answer_call_id,
        method_calls=tuple(recorder.get_records()),
        memory_before=memory_before,
        memory_after=tuple(entry.model_dump() for entry in state.reflections),
        memory_write_event=_memory_write_event(state, reflection_events),
        metadata=_metadata(attempts, reflection_events),
    )


def _failed_outcome(
    recorder: MethodCallRecorder,
    memory_before: tuple[dict[str, Any], ...],
    state: ReflexionState,
    attempts: list[ReflexionAttemptOutcome],
    reflection_events: list[ReflexionReflectionEvent],
    failure_disposition: FailureDisposition,
    *,
    final_response: str | None = None,
    parsed_answer: str | None = None,
    answer_call_id: str | None = None,
) -> BaselineExecutionOutcome:
    error_type, reason = _failure_triple(failure_disposition)
    return BaselineExecutionOutcome(
        status="failed",
        final_response=final_response,
        parsed_answer=parsed_answer,
        answer_call_id=answer_call_id,
        method_calls=tuple(recorder.get_records()),
        memory_before=memory_before,
        memory_after=tuple(entry.model_dump() for entry in state.reflections),
        memory_write_event=_memory_write_event(state, reflection_events),
        error_type=error_type,
        failure_disposition=failure_disposition,
        scientific_ineligibility_reason=reason,
        metadata=_metadata(attempts, reflection_events),
    )


def _memory_write_event(
    state: ReflexionState, events: list[ReflexionReflectionEvent]
) -> dict[str, Any] | None:
    if not events:
        return None
    entry = next(entry for entry in state.reflections if entry.entry_id == events[-1].reflection_entry_id)
    return {
        "type": "reflexion_append",
        "status": "accepted",
        "new_entry_id": entry.entry_id,
        "source_trial_id": entry.source_trial_id,
        "parent_entry_ids": list(entry.metadata["direct_parent_ids"]),
        "direct_parent_ids": list(entry.metadata["direct_parent_ids"]),
        "source_entry_ids": list(entry.metadata["source_entry_ids"]),
    }


def _metadata(
    attempts: list[ReflexionAttemptOutcome], events: list[ReflexionReflectionEvent]
) -> dict[str, Any]:
    return {
        "reflexion_attempt_outcomes": [
            {
                "attempt_id": attempt.attempt_id,
                "attempt_index": attempt.attempt_index,
                "answer_call_id": attempt.answer_call_id,
                "outcome": {
                    "status": attempt.outcome.status,
                    "verifier_result": attempt.outcome.verifier_result,
                },
                **(
                    {"failure_class": "incorrect_answer"}
                    if attempt.outcome.verifier_result is False
                    else {}
                ),
            }
            for attempt in attempts
        ],
        "reflexion_reflection_events": [
            {
                "attempt_id": event.attempt_id,
                "reflection_call_id": event.reflection_call_id,
                "reflection_entry_id": event.reflection_entry_id,
            }
            for event in events
        ],
    }


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
    raise TypeError("Reflexion verifier must return VerifierResult or bool")


def _answer_call_id(recorder: MethodCallRecorder) -> str | None:
    records = recorder.get_records()
    return records[-1].call_id if records else None


def _render_reflection(entry: MemoryEntry) -> str:
    return (
        f"Reflection ID: {entry.entry_id}\n"
        f"Reflection: {entry.content.removeprefix('Reflection:').strip()}"
    )


def _failed_actor_trajectory_span(
    response: str,
    *,
    message_index: int,
    start: int,
    end: int,
    failed_actor_call_id: str,
    failed_actor_spans: list[Any],
    target_set: Any,
) -> Any:
    source_ids, parent_ids, clean_or_contaminated = source_lineage_from_spans(failed_actor_spans)
    direct_parent_ids = _span_entry_ids(failed_actor_spans)
    lineage_status = combine_lineage_status(
        [span.lineage_status or "unavailable" for span in failed_actor_spans]
    )
    injected_root_ids: list[str] = []
    for span in failed_actor_spans:
        if span.lineage_status == "exact":
            _extend_unique(injected_root_ids, span.injected_root_ids)
    target_set_id = _target_set_id(target_set)
    contamination_class = "derived" if injected_root_ids and lineage_status == "exact" else "clean"
    return derived_source_span(
        response,
        message_index=message_index,
        start=start,
        end=end,
        entry_id=f"reflexion_failed_actor:{failed_actor_call_id}",
        parent_call_id=failed_actor_call_id,
        source_ids=source_ids,
        parent_ids=parent_ids,
        lineage_id=failed_actor_call_id,
        version="reflexion_failed_actor_v1",
        origin="reflexion_generate",
        clean_or_contaminated=clean_or_contaminated,
        contamination_class=contamination_class if target_set_id else None,
        injected_root_ids=injected_root_ids,
        lineage_status=lineage_status if target_set_id else None,
        lineage_basis="recorded_parent" if target_set_id else None,
        direct_parent_ids=direct_parent_ids,
        target_set_id=target_set_id,
        is_target_contamination=(
            contamination_class == "derived" and lineage_status == "exact"
            if target_set_id
            else None
        ),
    )


def _span_entry_ids(spans: list[Any]) -> list[str]:
    entry_ids: list[str] = []
    for span in spans:
        if isinstance(span.entry_id, str) and span.entry_id not in entry_ids:
            entry_ids.append(span.entry_id)
    return entry_ids


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _target_set_id(target_set: Any) -> str | None:
    if isinstance(target_set, str) and target_set:
        return target_set
    if isinstance(target_set, dict):
        target_set_id = target_set.get("target_set_id")
        if isinstance(target_set_id, str) and target_set_id:
            return target_set_id
    return None


def _contaminated_source_entry_ids(entries: list[MemoryEntry]) -> list[str]:
    source_ids: list[str] = []
    for entry in entries:
        if entry.clean_or_contaminated != "contaminated":
            continue
        for source_id in entry.metadata.get("source_entry_ids", [entry.entry_id]):
            if isinstance(source_id, str) and source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def _trial_id(task: TaskInstance, config: dict[str, Any], model: str) -> str:
    return ":".join(
        [
            str(config.get("run_id", "unknown")),
            task.task_name,
            task.sample_id,
            str(config.get("baseline", "reflexion_style")),
            str(config.get("arm", "clean")),
            str(config.get("model", model)),
        ]
    )


def _failure_triple(
    failure_disposition: FailureDisposition,
) -> tuple[ErrorType, ScientificIneligibilityReason]:
    triples: dict[FailureDisposition, tuple[ErrorType, ScientificIneligibilityReason]] = {
        "reflexion_invalid_generation": ("BaselineOutputError", "invalid_reflexion_generation"),
        "reflexion_invalid_reflection": ("BaselineOutputError", "invalid_reflection"),
        "provider_call_failed": ("ProviderCallFailure", "provider_call_failed"),
        "verifier_contract_failed": ("VerifierContractError", "verifier_contract_failed"),
    }
    return triples[failure_disposition]
