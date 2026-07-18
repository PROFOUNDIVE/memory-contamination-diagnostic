from __future__ import annotations

import json
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


_MEMORY_SEED_PLACEHOLDER = "<task prompt>"


class FullHistoryPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        history = "\n".join(entry.content for entry in memory.entries)
        return [{"role": "user", "content": f"History:\n{history}\n\nSolve: {task.input}"}]

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
        config = dict(config or {})
        memory_before = [entry.model_dump() for entry in memory.entries]
        messages, source_spans = self._build_messages(task, memory)
        recorder = MethodCallRecorder(
            client,
            event_callback=config.get("_logging_event_callback"),
            trial_context={
                **config.get("_logging_trial_context", {}),
                "trial_id": ":".join(
                    [
                        str(config.get("run_id", "unknown")),
                        task.task_name,
                        task.sample_id,
                        str(config.get("baseline", "full_history")),
                        str(config.get("arm", "clean")),
                        str(config.get("model", model)),
                    ]
                ),
            },
        )
        call_config = {
            **config,
            "sample_id": config.get("sample_id", task.sample_id),
            "method_stage": "full_history_generate",
            "source_spans": source_spans,
        }
        response = recorder.chat(messages, model, call_config)
        parsed_answer = _parse_answer(response.content)
        verifier_result = (
            _call_verifier(verifier, parsed_answer, task)
            if verifier is not None
            else VerifierResult(is_correct=True, parsed_answer=parsed_answer)
        )
        if verifier_result.parsed_answer is None:
            verifier_result.parsed_answer = parsed_answer

        parent_entry_ids = [entry.entry_id for entry in memory.entries]
        source_entry_ids = [
            entry.entry_id for entry in memory.entries if entry.clean_or_contaminated == "contaminated"
        ]
        lineage = "contaminated" if source_entry_ids else "clean"
        trial_id = ":".join(
            [
                str(config.get("run_id", "unknown_run")),
                task.task_name,
                task.sample_id,
                str(config.get("baseline", "full_history")),
                str(config.get("arm", "clean")),
                str(config.get("model", model)),
            ]
        )
        new_entry = MemoryEntry(
            entry_id=f"full_history:{task.task_name}:{task.sample_id}:{uuid4().hex}",
            content=_render_transcript(
                task=task,
                prompt_messages=messages,
                raw_response=response.content,
                parsed_answer=parsed_answer,
                correct=verifier_result.is_correct,
            ),
            memory_type="full_history_transcript",
            clean_or_contaminated=lineage,
            source_trial_id=trial_id,
            metadata={
                "parent_entry_ids": parent_entry_ids,
                "direct_parent_ids": parent_entry_ids,
                "source_entry_ids": source_entry_ids,
                "lineage": lineage,
                "task_input": task.input,
                "prompt_messages": messages,
                "raw_response": response.content,
                "parsed_answer": parsed_answer,
                "correct": verifier_result.is_correct,
                **(
                    {"memory_error_status": "satisfied"}
                    if config.get("_logging_target_set_id") and not verifier_result.is_correct
                    else {}
                ),
            },
        )
        new_entry.metadata.update(
            phase11_lineage_metadata(
                new_entry,
                [*memory.entries, new_entry],
                config.get("_logging_target_contamination_set")
                or config.get("_logging_target_set_id"),
            )
        )
        memory.entries.append(new_entry)

        memory_after = [entry.model_dump() for entry in memory.entries]
        memory_write_event = {
            "type": "full_history_append",
            "status": "accepted",
            "new_entry_id": new_entry.entry_id,
            "source_trial_id": trial_id,
            "parent_entry_ids": parent_entry_ids,
            "source_entry_ids": source_entry_ids,
        }
        method_calls = recorder.get_records()
        answer_call_id = method_calls[0].call_id if method_calls else None
        return {
            "final_response": response.content,
            "parsed_answer": parsed_answer,
            "verifier_result": verifier_result,
            "method_calls": method_calls,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_write_event": memory_write_event,
            "metadata": {
                "parent_entry_ids": parent_entry_ids,
                "source_entry_ids": source_entry_ids,
                "lineage": lineage,
            },
            "retrieved_records": [],
            "retrieved_scores": [],
            "answer_call_id": answer_call_id,
        }

    def _build_messages(
        self, task: TaskInstance, memory: MemoryState
    ) -> tuple[list[dict[str, str]], list[Any]]:
        parts: list[str | PromptSourcePart] = ["History:\n"]
        for index, entry in enumerate(memory.entries):
            if index > 0:
                parts.append("\n\n")
            previous_input = _MEMORY_SEED_PLACEHOLDER
            if entry.memory_type == "full_history_transcript":
                previous_input = str(entry.metadata.get("task_input", _MEMORY_SEED_PLACEHOLDER))
            parts.append(f"Previous input: {previous_input}\nPrevious response: ")
            parts.append(PromptSourcePart(entry.content, entry))
        parts.append("\n\nSolve: ")
        parts.append(str(task.input))
        content, spans = build_prompt_with_sources(parts, message_index=0, entries=memory.entries)
        return [{"role": "user", "content": content}], spans


def _render_memory_entry(entry: MemoryEntry) -> str:
    previous_input = _MEMORY_SEED_PLACEHOLDER
    if entry.memory_type == "full_history_transcript":
        previous_input = str(entry.metadata.get("task_input", _MEMORY_SEED_PLACEHOLDER))
    return f"Previous input: {previous_input}\nPrevious response: {entry.content}"


def _render_transcript(
    *,
    task: TaskInstance,
    prompt_messages: list[dict[str, str]],
    raw_response: str,
    parsed_answer: str,
    correct: bool,
) -> str:
    return (
        f"Previous input: {task.input}\n"
        f"Previous prompt: {json.dumps(prompt_messages, ensure_ascii=False)}\n"
        f"Previous response: {raw_response}\n"
        f"Parsed answer: {parsed_answer}\n"
        f"Correct: {str(correct).lower()}"
    )


def _parse_answer(response: str) -> str:
    stripped = response.strip()
    if stripped.lower().startswith("final:"):
        return stripped.split(":", 1)[1].strip()
    return stripped


def _call_verifier(
    verifier: Callable[[str, TaskInstance], VerifierResult], parsed_answer: str, task: TaskInstance
) -> VerifierResult:
    return verifier(parsed_answer, task)
