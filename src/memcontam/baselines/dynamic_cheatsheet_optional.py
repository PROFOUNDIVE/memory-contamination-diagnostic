"""Faithful adapted cumulative Dynamic Cheatsheet runtime."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.provenance import (
    PromptSourcePart,
    build_prompt_with_sources,
    derived_source_span,
    source_lineage_from_spans,
)
from memcontam.logging.schema import VerifierResult
from memcontam.memory.embeddings import EmbeddingProvider
from memcontam.memory.retrieval import DenseIndex
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


Verifier = Callable[[str, TaskInstance], VerifierResult]


class DynamicCheatsheetRetrievalSynthesisPolicy:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.cache_dir = cache_dir

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
        trial_id = _trial_id(task, call_config, model)
        memory_before = [entry.model_dump() for entry in memory.entries]
        pair_entries = [entry for entry in memory.entries if entry.memory_type == "dc_rs_io_pair"]
        cheatsheet_entries = [entry for entry in memory.entries if _is_cheatsheet(entry)]
        cheatsheet = _dc_rs_cheatsheet(cheatsheet_entries)
        retrieved_records = self._retrieve_pairs(task.input, pair_entries, trial_id)
        pairs_by_id = {entry.entry_id: entry for entry in pair_entries}
        retrieved_pairs = [pairs_by_id[record.document_id] for record in retrieved_records]
        recorder = MethodCallRecorder(
            client,
            event_callback=call_config.get("_logging_event_callback"),
            trial_context={**call_config.get("_logging_trial_context", {}), "trial_id": trial_id},
        )

        synthesis_message, synthesis_spans = _synthesis_message_with_sources(
            task, cheatsheet_entries, retrieved_pairs
        )
        synthesized = recorder.chat(
            [synthesis_message],
            model=model,
            config={
                **call_config,
                "method_stage": "dc_rs_synthesize",
                "source_spans": synthesis_spans,
            },
        )
        synthesis_call = recorder.get_records()[-1]
        updated_cheatsheet, parser_status = _extract_cheatsheet(synthesized.content, cheatsheet)
        generation_message, generation_spans = _dc_rs_generation_message(
            task, updated_cheatsheet, synthesis_call.call_id, synthesis_call.source_spans
        )
        generated = recorder.chat(
            [generation_message],
            model=model,
            config={
                **call_config,
                "method_stage": "dc_rs_generate",
                "source_spans": generation_spans,
            },
        )
        answer_call_id = recorder.get_records()[-1].call_id
        parsed_answer = _parse_answer(generated.content)
        verifier_result = (
            verifier(parsed_answer, task)
            if verifier is not None
            else _default_verifier(parsed_answer)
        )

        method_calls = recorder.get_records()
        method_calls[0].retrieved_records = retrieved_records
        lineage = _lineage(memory.entries)
        if parser_status == "accepted":
            replacement = MemoryEntry(
                entry_id=f"dc_rs_cheatsheet:{trial_id}",
                content=updated_cheatsheet,
                memory_type="dynamic_cheatsheet",
                clean_or_contaminated=(
                    "contaminated" if lineage["source_contaminated_entry_ids"] else "clean"
                ),
                source_trial_id=trial_id,
                metadata=lineage,
            )
            state_entries = [entry for entry in memory.entries if not _is_cheatsheet(entry)]
            memory_after_entries = [replacement, *state_entries]
            synthesis_update = {"status": "replaced", "parser_status": parser_status}
        else:
            memory_after_entries = list(memory.entries)
            synthesis_update = {"status": "preserved", "parser_status": parser_status}

        pair = MemoryEntry(
            entry_id=f"dc_rs_pair:{trial_id}",
            content=str(task.input),
            memory_type="dc_rs_io_pair",
            clean_or_contaminated=(
                "contaminated" if lineage["source_contaminated_entry_ids"] else "clean"
            ),
            source_trial_id=trial_id,
            metadata={**lineage, "output_text": parsed_answer},
        )
        memory_after_entries.append(pair)
        event = {
            "type": "dynamic_cheatsheet_rs_update",
            "status": parser_status,
            "source_trial_id": trial_id,
            "previous_entry_ids": [entry.entry_id for entry in memory.entries],
            "synthesis_update": synthesis_update,
            "pair_appended": {
                "entry_id": pair.entry_id,
                "source_trial_id": trial_id,
                "parent_entry_ids": lineage["parent_entry_ids"],
                "source_entry_ids": lineage["source_entry_ids"],
                "source_contaminated_entry_ids": lineage["source_contaminated_entry_ids"],
            },
        }
        return {
            "final_response": generated.content,
            "parsed_answer": parsed_answer,
            "verifier_result": verifier_result,
            "retrieved_records": retrieved_records,
            "retrieved_memory": [
                {**record.model_dump(), "entry_id": record.document_id} for record in retrieved_records
            ],
            "retrieved_scores": [record.score for record in retrieved_records],
            "method_calls": method_calls,
            "memory_before": memory_before,
            "memory_after": [entry.model_dump() for entry in memory_after_entries],
            "memory_write_event": event,
            "answer_call_id": answer_call_id,
            "metadata": {},
        }

    def _retrieve_pairs(
        self, task_input: dict, pair_entries: list[MemoryEntry], trial_id: str
    ) -> list[Any]:
        k = min(3, len(pair_entries))
        if self.cache_dir is None:
            with tempfile.TemporaryDirectory() as cache_dir:
                return DenseIndex(
                    pair_entries, provider=self.embedding_provider, cache_dir=cache_dir
                ).retrieve(str(task_input), k)
        cache_key = hashlib.sha256(
            "\0".join([trial_id, *(entry.entry_id for entry in pair_entries)]).encode("utf-8")
        ).hexdigest()
        return DenseIndex(
            pair_entries,
            provider=self.embedding_provider,
            cache_dir=Path(self.cache_dir) / cache_key,
        ).retrieve(str(task_input), k)


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
        trial_id = _trial_id(task, call_config, model)
        recorder = MethodCallRecorder(
            client,
            event_callback=call_config.get("_logging_event_callback"),
            trial_context={**call_config.get("_logging_trial_context", {}), "trial_id": trial_id},
        )
        cheatsheet_entries = [entry for entry in memory.entries if _is_cheatsheet(entry)]
        generation_entries = cheatsheet_entries if len(cheatsheet_entries) == 1 else memory.entries
        generation_message, generation_spans = _generation_message_with_sources(
            task, cheatsheet, generation_entries
        )

        generated = recorder.chat(
            [generation_message],
            model=model,
            config={
                **call_config,
                "method_stage": "dynamic_cheatsheet_generate",
                "source_spans": generation_spans,
            },
        )
        answer_call_id = recorder.get_records()[-1].call_id
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
            "source_trial_id": trial_id,
            "previous_entry_ids": [entry.entry_id for entry in memory.entries],
        }
        if status == "accepted":
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
            "answer_call_id": answer_call_id,
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


def _generation_message_with_sources(
    task: TaskInstance, cheatsheet: str, entries: list[MemoryEntry]
) -> tuple[dict[str, str], list[Any]]:
    prefix = f"Task input: {task.input}\n\nCheatsheet:\n"
    suffix = "\n\nSolve the task and respond in the normal harness format: final: <answer>."
    if not entries:
        return {"role": "user", "content": prefix + cheatsheet + suffix}, []

    parts: list[str | PromptSourcePart] = [prefix]
    for index, entry in enumerate(entries):
        if index:
            parts.append("\n")
        text = entry.content if len(entries) == 1 and _is_cheatsheet(entry) else f"- {entry.content}"
        parts.append(PromptSourcePart(text, entry))
    parts.append(suffix)
    content, spans = build_prompt_with_sources(parts, message_index=0)
    return {"role": "user", "content": content}, spans


def _dc_rs_cheatsheet(entries: list[MemoryEntry]) -> str:
    if len(entries) == 1:
        return entries[0].content
    return "\n".join(entry.content for entry in entries)


def _synthesis_message(
    task: TaskInstance, cheatsheet: str, pairs: list[MemoryEntry]
) -> dict[str, str]:
    prior_pairs = "\n\n".join(
        f"Prior input:\n{entry.content}\n\nPrior output:\n{entry.metadata['output_text']}"
        for entry in pairs
    )
    return {
        "role": "user",
        "content": (
            f"Existing cheatsheet:\n{cheatsheet}\n\nRetrieved prior input/output pairs:\n"
            f"{prior_pairs}\n\nCurrent task input:\n{task.input}\n\n"
            "Return exactly one <cheatsheet>...</cheatsheet> block for solving the current task."
        ),
    }


def _synthesis_message_with_sources(
    task: TaskInstance,
    cheatsheet_entries: list[MemoryEntry],
    pairs: list[MemoryEntry],
) -> tuple[dict[str, str], list[Any]]:
    parts: list[str | PromptSourcePart] = ["Existing cheatsheet:\n"]
    for index, entry in enumerate(cheatsheet_entries):
        if index:
            parts.append("\n")
        parts.append(PromptSourcePart(entry.content, entry))
    parts.append("\n\nRetrieved prior input/output pairs:\n")
    for index, entry in enumerate(pairs):
        if index:
            parts.append("\n\n")
        parts.append(
            PromptSourcePart(
                f"Prior input:\n{entry.content}\n\nPrior output:\n{entry.metadata['output_text']}",
                entry,
            )
        )
    parts.append(
        f"\n\nCurrent task input:\n{task.input}\n\n"
        "Return exactly one <cheatsheet>...</cheatsheet> block for solving the current task."
    )
    content, spans = build_prompt_with_sources(parts, message_index=0)
    return {"role": "user", "content": content}, spans


def _dc_rs_generation_message(
    task: TaskInstance,
    cheatsheet: str,
    synthesis_call_id: str | None,
    synthesis_spans: list[Any],
) -> tuple[dict[str, str], list[Any]]:
    prefix = f"Task input: {task.input}\n\nCheatsheet:\n"
    suffix = "\n\nSolve the task and respond in the normal harness format: final: <answer>."
    if not cheatsheet or synthesis_call_id is None:
        return {"role": "user", "content": prefix + cheatsheet + suffix}, []
    source_ids, parent_ids, clean_or_contaminated = source_lineage_from_spans(synthesis_spans)
    content = prefix + cheatsheet + suffix
    return {
        "role": "user",
        "content": content,
    }, [
        derived_source_span(
            cheatsheet,
            message_index=0,
            start=len(prefix),
            end=len(prefix) + len(cheatsheet),
            entry_id=f"dc_rs_synthesized:{synthesis_call_id}",
            parent_call_id=synthesis_call_id,
            source_ids=source_ids,
            parent_ids=parent_ids,
            lineage_id=synthesis_call_id,
            version="dc_rs_synthesis_v1",
            origin="dc_rs_synthesize",
            clean_or_contaminated=clean_or_contaminated,
        )
    ]


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
