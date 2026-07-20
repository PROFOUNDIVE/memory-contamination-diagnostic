from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

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
from memcontam.logging.provenance import PromptSourcePart, build_prompt_with_sources, phase11_lineage_metadata
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, apply_keep_last_3
from memcontam.tasks.base import TaskInstance


@dataclass
class ReflexionState:
    reflections: list[MemoryEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reflections[:] = apply_keep_last_3(self.reflections)


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
            messages, source_spans = _generation_messages(task, state.reflections)
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
                reflection_response = recorder.chat(
                    _reflection_messages(task, state.reflections, response.content, parsed_answer),
                    model=model,
                    config={
                        **config,
                        "sample_id": config.get("sample_id", task.sample_id),
                        "method_stage": "reflexion_reflect",
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
            payload = _parse_reflection(reflection_response.content, state.reflections)
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
            entry = _append_reflection(task, state, payload, trial_id, reflection_call_id, config)
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
    stripped = response.strip()
    if not stripped.lower().startswith("final:"):
        return None
    answer = stripped.split(":", 1)[1].strip()
    return answer or None


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
    parts.append(f"\n\nCurrent task input:\n{task.input}")
    content, spans = build_prompt_with_sources(parts, message_index=1)
    return [
        {"role": "system", "content": f"Solve the {task.task_name} task using reflections when useful."},
        {"role": "user", "content": content},
    ], spans


def _reflection_messages(
    task: TaskInstance, reflections: list[MemoryEntry], response: str, parsed_answer: str
) -> list[dict[str, str]]:
    reflection_context = "\n".join(_render_reflection(entry) for entry in reflections) or "(none)"
    return [
        {"role": "system", "content": "Diagnose the failed attempt and return corrective JSON only."},
        {
            "role": "user",
            "content": (
                f"Task: {task.task_name}\n\nReflections:\n{reflection_context}"
                f"\n\nTask input:\n{task.input}\n\nFailed raw response:\n{response}"
                f"\n\nParsed answer:\n{parsed_answer}\n\nCorrect: false"
            ),
        },
    ]


def _append_reflection(
    task: TaskInstance,
    state: ReflexionState,
    payload: ReflectionPayload,
    trial_id: str,
    reflection_call_id: str,
    config: dict[str, Any],
) -> MemoryEntry:
    used_ids = list(payload.explicitly_used_memory_ids)
    entries_by_id = {entry.entry_id: entry for entry in state.reflections}
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
            "parent_call_id": reflection_call_id,
            "reflection_lineage": {"stage": "reflexion_reflect", "source_trial_id": trial_id},
        },
    )
    target_set = config.get("_logging_target_contamination_set") or config.get("_logging_target_set_id")
    if target_set:
        entry.metadata.update(phase11_lineage_metadata(entry, [*state.reflections, entry], target_set))
    state.reflections[:] = apply_keep_last_3([*state.reflections, entry])
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
    return f"Reflection: {entry.content.removeprefix('Reflection:').strip()}"


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
