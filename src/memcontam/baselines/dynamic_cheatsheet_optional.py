"""Faithful adapted cumulative Dynamic Cheatsheet runtime."""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema import VerifierResult
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


Verifier = Callable[[str, TaskInstance], VerifierResult]


class DynamicCheatsheetOptionalPolicy:
    def run(
        self,
        task: TaskInstance,
        memory: MemoryState,
        *,
        client: LLMClient,
        model: str,
        config: dict[str, Any] | None = None,
        verifier: Verifier | None = None,
    ) -> dict[str, Any]:
        call_config = {**(config or {}), "sample_id": (config or {}).get("sample_id", task.sample_id)}
        cheatsheet, lineage = _current_cheatsheet(memory)
        memory_before = [entry.model_dump() for entry in memory.entries]
        recorder = MethodCallRecorder(client)

        generated = recorder.chat(
            [_generation_message(task, cheatsheet)],
            model=model,
            config={**call_config, "method_stage": "dynamic_cheatsheet_generate"},
        )
        parsed_answer = _parse_answer(generated.content)
        verifier_result = (
            verifier(parsed_answer, task)
            if verifier is not None
            else _default_verifier(parsed_answer)
        )
        curated = recorder.chat(
            [_curation_message(task, cheatsheet, generated.content, parsed_answer, verifier_result.is_correct)],
            model=model,
            config={**call_config, "method_stage": "dynamic_cheatsheet_curate"},
        )
        updated_cheatsheet, status = _extract_cheatsheet(curated.content, cheatsheet)

        event = {
            "type": "dynamic_cheatsheet_update",
            "status": status,
            "previous_entry_ids": [entry.entry_id for entry in memory.entries],
        }
        if status == "accepted":
            trial_id = _trial_id(task, call_config, model)
            entry = MemoryEntry(
                entry_id=f"dc_cheatsheet:{task.task_name}:{uuid4().hex}",
                content=updated_cheatsheet,
                memory_type="dynamic_cheatsheet",
                clean_or_contaminated=(
                    "contaminated" if lineage["source_contaminated_entry_ids"] else "clean"
                ),
                source_trial_id=trial_id,
                metadata=lineage,
            )
            memory_after = [entry.model_dump()]
            event.update(
                {
                    "new_entry_id": entry.entry_id,
                    "parent_entry_ids": lineage["parent_entry_ids"],
                    "source_entry_ids": lineage["source_entry_ids"],
                    "source_contaminated_entry_ids": lineage["source_contaminated_entry_ids"],
                }
            )
        else:
            memory_after = memory_before

        return {
            "final_response": generated.content,
            "parsed_answer": parsed_answer,
            "verifier_result": verifier_result,
            "retrieved_records": [],
            "retrieved_memory": [],
            "retrieved_scores": [],
            "method_calls": recorder.get_records(),
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_write_event": event,
            "metadata": {},
        }


def _current_cheatsheet(memory: MemoryState) -> tuple[str, dict[str, list[str]]]:
    cheatsheet_entries = [entry for entry in memory.entries if _is_cheatsheet(entry)]
    if len(cheatsheet_entries) == 1:
        entry = cheatsheet_entries[0]
        return entry.content, _lineage([entry])
    return "\n".join(f"- {entry.content}" for entry in memory.entries), _lineage(memory.entries)


def _is_cheatsheet(entry: MemoryEntry) -> bool:
    return entry.memory_type == "dynamic_cheatsheet" or entry.entry_id.startswith("dc_cheatsheet:")


def _lineage(entries: list[MemoryEntry]) -> dict[str, list[str]]:
    parent_entry_ids: list[str] = []
    source_entry_ids: list[str] = []
    source_trial_ids: list[str] = []
    for entry in entries:
        metadata = entry.metadata
        _extend_unique(parent_entry_ids, metadata.get("parent_entry_ids", []))
        _extend_unique(parent_entry_ids, [entry.entry_id])
        sources = metadata.get("source_entry_ids", [entry.entry_id])
        _extend_unique(
            source_entry_ids,
            metadata.get("source_contaminated_entry_ids", []),
        )
        if entry.clean_or_contaminated == "contaminated":
            _extend_unique(source_entry_ids, sources)
        _extend_unique(source_trial_ids, metadata.get("source_trial_ids", []))
        if entry.source_trial_id:
            _extend_unique(source_trial_ids, [entry.source_trial_id])
    return {
        "parent_entry_ids": parent_entry_ids,
        "source_entry_ids": source_entry_ids,
        "source_contaminated_entry_ids": source_entry_ids,
        "source_trial_ids": source_trial_ids,
    }


def _trial_id(task: TaskInstance, config: dict[str, Any], model: str) -> str:
    return (
        f"{config.get('run_id', 'unknown_run')}:{task.task_name}:{task.sample_id}:"
        f"{config.get('baseline', 'dynamic_cheatsheet_optional')}:"
        f"{config.get('arm', 'clean')}:{config.get('model', model)}"
    )


def _extend_unique(target: list[str], values: object) -> None:
    if not isinstance(values, list):
        return
    for value in values:
        if isinstance(value, str) and value not in target:
            target.append(value)


def _generation_message(task: TaskInstance, cheatsheet: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Task input: {task.input}\n\nCheatsheet:\n{cheatsheet}\n\n"
            "Solve the task and respond in the normal harness format: final: <answer>."
        ),
    }


def _curation_message(
    task: TaskInstance,
    cheatsheet: str,
    raw_output: str,
    parsed_answer: str,
    is_correct: bool,
) -> dict[str, str]:
    correctness = "true" if is_correct else "false"
    return {
        "role": "user",
        "content": (
            f"Previous cheatsheet:\n{cheatsheet}\n\nTask input: {task.input}\n\n"
            f"Raw output: {raw_output}\nParsed answer: {parsed_answer}\nCorrect: {correctness}\n\n"
            "Return exactly one <cheatsheet>...</cheatsheet> block with the updated cheatsheet."
        ),
    }


def _parse_answer(response: str) -> str:
    response = response.strip()
    if response.lower().startswith("final:"):
        return response.split(":", 1)[1].strip()
    return response


def _default_verifier(answer: str) -> VerifierResult:
    return VerifierResult(is_correct=True, parsed_answer=answer)


def _extract_cheatsheet(text: str, fallback: str) -> tuple[str, str]:
    start = text.find("<cheatsheet>")
    if start < 0:
        return fallback, "preserved_missing_tag"
    start += len("<cheatsheet>")
    end = text.find("</cheatsheet>", start)
    if end < 0:
        return fallback, "preserved_missing_tag"
    cheatsheet = text[start:end].strip()
    if not cheatsheet:
        return fallback, "preserved_empty"
    return cheatsheet, "accepted"
