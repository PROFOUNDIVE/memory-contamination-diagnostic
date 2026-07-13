from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
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
        memory_before = [entry.model_dump() for entry in memory.entries]
        reflection_entries = _reflection_entries(memory)
        reflection_context = _reflection_context(reflection_entries)
        recorder = MethodCallRecorder(client)
        response = recorder.chat(
            [
                {
                    "role": "system",
                    "content": f"Solve the {task.task_name} task using reflections when useful.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task.task_name}\n\nReflections:\n{reflection_context}"
                        f"\n\nCurrent task input:\n{task.input}"
                    ),
                },
            ],
            model=model,
            config={**call_config, "method_stage": "reflexion_generate"},
        )
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
            )

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
                    ),
                },
            ],
            model=model,
            config={**call_config, "method_stage": "reflexion_reflect"},
        )
        parent_entry_ids = [entry.entry_id for entry in reflection_entries]
        source_entry_ids = _contaminated_source_entry_ids(reflection_entries)
        event = {
            "type": "reflexion_append",
            "status": "rejected_empty",
            "parent_entry_ids": parent_entry_ids,
            "source_entry_ids": source_entry_ids,
        }
        reflection = reflection_response.content.strip()
        if reflection:
            source_trial_id = _trial_id(task, call_config, model)
            entry = MemoryEntry(
                entry_id=f"reflexion:{task.task_name}:{task.sample_id}:{uuid4().hex}",
                content=f"Reflection: {reflection}",
                memory_type="verbal_reflection",
                clean_or_contaminated="contaminated" if source_entry_ids else "clean",
                source_trial_id=source_trial_id,
                metadata={
                    "parent_entry_ids": parent_entry_ids,
                    "source_entry_ids": source_entry_ids,
                    "reflection_lineage": {
                        "stage": "reflexion_reflect",
                        "parent_entry_ids": parent_entry_ids,
                        "source_trial_id": source_trial_id,
                    },
                },
            )
            memory.entries.append(entry)
            event = {
                "type": "reflexion_append",
                "status": "accepted",
                "new_entry_id": entry.entry_id,
                "parent_entry_ids": parent_entry_ids,
                "source_entry_ids": source_entry_ids,
            }
        return _result(
            response.content,
            parsed_answer,
            verifier_result,
            recorder,
            memory_before,
            memory,
            event,
        )


def _result(
    final_response: str,
    parsed_answer: str,
    verifier_result: VerifierResult,
    recorder: MethodCallRecorder,
    memory_before: list[dict[str, Any]],
    memory: MemoryState,
    memory_write_event: dict[str, Any] | None,
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
    }
