from __future__ import annotations

import re
from typing import Any, Callable
from uuid import uuid4

from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.provenance import (
    PromptSourcePart,
    build_prompt_with_sources,
    phase11_lineage_metadata,
)
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


def _parse_answer(response: str) -> str:
    response = response.strip()
    if response.lower().startswith("final:"):
        return response.split(":", 1)[1].strip()
    return response


def _reflection_entries(memory: MemoryState) -> list[MemoryEntry]:
    return [
        entry
        for entry in memory.entries
        if entry.memory_type == "verbal_reflection" or entry.content.startswith("Reflection:")
    ][-3:]


def _render_reflection(entry: MemoryEntry) -> str:
    content = entry.content.strip()
    if entry.memory_type == "verbal_reflection":
        return f"Reflection: {content.removeprefix('Reflection:').strip()}"
    return content


def _reflection_context(entries: list[MemoryEntry]) -> str:
    return "\n".join(_render_reflection(entry) for entry in entries) or "(none)"


def _generation_messages(
    task: TaskInstance, reflection_entries: list[MemoryEntry]
) -> tuple[list[dict[str, str]], list[Any]]:
    parts: list[str | PromptSourcePart] = [
        f"Task: {task.task_name}\n\nReflections:\n",
    ]
    if reflection_entries:
        for index, entry in enumerate(reflection_entries):
            if index:
                parts.append("\n")
            parts.append(PromptSourcePart(_render_reflection(entry), entry))
    else:
        parts.append("(none)")
    parts.append(f"\n\nCurrent task input:\n{task.input}")
    content, spans = build_prompt_with_sources(parts, message_index=1)
    return [
        {
            "role": "system",
            "content": f"Solve the {task.task_name} task using reflections when useful.",
        },
        {"role": "user", "content": content},
    ], spans


def _sanitized_feedback(verifier_result: VerifierResult) -> str:
    reason = verifier_result.reason
    if isinstance(reason, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason):
        return reason
    return "verifier_rejected"


def _contaminated_source_entry_ids(entries: list[MemoryEntry]) -> list[str]:
    source_entry_ids: list[str] = []
    for entry in entries:
        if entry.clean_or_contaminated != "contaminated":
            continue
        sources = entry.metadata.get("source_entry_ids", [entry.entry_id])
        if not isinstance(sources, list):
            sources = [entry.entry_id]
        for source_entry_id in sources:
            if isinstance(source_entry_id, str) and source_entry_id not in source_entry_ids:
                source_entry_ids.append(source_entry_id)
    return source_entry_ids


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


class ReflexionStylePolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        reflections = "\n".join(entry.content for entry in memory.entries[-3:])
        return [{"role": "user", "content": f"Reflections:\n{reflections}\n\nSolve: {task.input}"}]

    def run(
        self,
        task: TaskInstance,
        memory: MemoryState,
        *,
        client: LLMClient,
        model: str,
        config: dict[str, Any] | None = None,
        verifier: Callable[[str, TaskInstance], VerifierResult] | None = None,
    ) -> dict[str, Any]:
        call_config = {**(config or {}), "sample_id": (config or {}).get("sample_id", task.sample_id)}
        max_attempts = call_config.get("max_attempts", 1)
        if type(max_attempts) is not int or max_attempts not in {1, 2}:
            raise ValueError("reflexion max_attempts must be 1 or 2")
        memory_before = [entry.model_dump() for entry in memory.entries]
        reflection_entries = _reflection_entries(memory)
        reflection_context = _reflection_context(reflection_entries)
        source_trial_id = _trial_id(task, call_config, model)
        recorder = MethodCallRecorder(
            client,
            event_callback=call_config.get("_logging_event_callback"),
            trial_context={**call_config.get("_logging_trial_context", {}), "trial_id": source_trial_id},
        )
        generation_messages, generation_spans = _generation_messages(task, reflection_entries)
        response = recorder.chat(
            generation_messages,
            model=model,
            config={
                **call_config,
                "method_stage": "reflexion_generate",
                "source_spans": generation_spans,
            },
        )
        answer_call_id = recorder.get_records()[-1].call_id
        parsed_answer = _parse_answer(response.content)
        verifier_result = (
            verifier(parsed_answer, task)
            if verifier is not None
            else VerifierResult(is_correct=True)
        )
        if verifier_result.is_correct:
            return _result(
                response.content,
                parsed_answer,
                verifier_result,
                recorder,
                memory_before,
                memory,
                None,
                answer_call_id,
            )

        feedback = _sanitized_feedback(verifier_result)
        reflection_response = recorder.chat(
            [
                {
                    "role": "system",
                    "content": "Diagnose the failed attempt and write a concise mitigation plan.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task.task_name}\n\nReflections:\n{reflection_context}"
                        f"\n\nTask input:\n{task.input}\n\nFailed raw response:\n{response.content}"
                        f"\n\nParsed answer:\n{parsed_answer}\n\nCorrect: false"
                        f"\n\nVerifier feedback:\n{feedback}"
                    ),
                },
            ],
            model=model,
            config={**call_config, "method_stage": "reflexion_reflect"},
        )
        reflection_call_id = recorder.get_records()[-1].call_id
        parent_entry_ids = [entry.entry_id for entry in reflection_entries]
        source_entry_ids = _contaminated_source_entry_ids(reflection_entries)
        event = {
            "type": "reflexion_append",
            "status": "rejected_empty",
            "source_trial_id": source_trial_id,
            "parent_entry_ids": parent_entry_ids,
            "source_entry_ids": source_entry_ids,
        }
        reflection = reflection_response.content.strip()
        if reflection:
            entry = MemoryEntry(
                entry_id=f"reflexion:{task.task_name}:{task.sample_id}:{uuid4().hex}",
                content=f"Reflection: {reflection}",
                memory_type="verbal_reflection",
                clean_or_contaminated="contaminated" if source_entry_ids else "clean",
                source_trial_id=source_trial_id,
                metadata={
                    "parent_entry_ids": parent_entry_ids,
                    "direct_parent_ids": parent_entry_ids,
                    "source_entry_ids": source_entry_ids,
                    "parent_call_id": reflection_call_id,
                    "reflection_lineage": {
                        "stage": "reflexion_reflect",
                        "parent_entry_ids": parent_entry_ids,
                        "source_trial_id": source_trial_id,
                    },
                },
            )
            entry.metadata.update(
                phase11_lineage_metadata(
                    entry,
                    [*memory.entries, entry],
                    call_config.get("_logging_target_contamination_set")
                    or call_config.get("_logging_target_set_id"),
                )
            )
            memory.entries.append(entry)
            event = {
                "type": "reflexion_append",
                "status": "accepted",
                "new_entry_id": entry.entry_id,
                "source_trial_id": source_trial_id,
                "parent_entry_ids": parent_entry_ids,
                "source_entry_ids": source_entry_ids,
            }
        elif max_attempts == 2:
            event["fidelity_invalid"] = True

        if not reflection or max_attempts == 1:
            return _result(
                response.content,
                parsed_answer,
                verifier_result,
                recorder,
                memory_before,
                memory,
                event,
                answer_call_id,
            )

        retry_messages, retry_spans = _generation_messages(task, _reflection_entries(memory))
        retry_response = recorder.chat(
            retry_messages,
            model=model,
            config={
                **call_config,
                "method_stage": "reflexion_generate",
                "retry_count": 1,
                "source_spans": retry_spans,
            },
        )
        answer_call_id = recorder.get_records()[-1].call_id
        retry_parsed_answer = _parse_answer(retry_response.content)
        retry_verifier_result = (
            verifier(retry_parsed_answer, task)
            if verifier is not None
            else VerifierResult(is_correct=True)
        )
        return _result(
            retry_response.content,
            retry_parsed_answer,
            retry_verifier_result,
            recorder,
            memory_before,
            memory,
            event,
            answer_call_id,
        )


def _result(
    final_response: str,
    parsed_answer: str,
    verifier_result: VerifierResult,
    recorder: MethodCallRecorder,
    memory_before: list[dict[str, Any]],
    memory: MemoryState,
    memory_write_event: dict[str, Any] | None,
    answer_call_id: str | None,
) -> dict[str, Any]:
    return {
        "final_response": final_response,
        "parsed_answer": parsed_answer,
        "verifier_result": verifier_result,
        "retrieved_records": [],
        "retrieved_memory": [],
        "retrieved_scores": [],
        "method_calls": recorder.get_records(),
        "memory_before": memory_before,
        "memory_after": [entry.model_dump() for entry in memory.entries],
        "metadata": {},
        "memory_write_event": memory_write_event,
        "answer_call_id": answer_call_id,
    }
