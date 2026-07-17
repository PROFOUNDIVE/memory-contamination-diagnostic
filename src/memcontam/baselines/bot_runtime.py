from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, Callable

from memcontam.baselines.bot_style import (
    BotStylePolicy,
    _retrieve_top1_template,
    distill_thought_template,
)
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema import VerifierResult
from memcontam.memory.bot_buffer import (
    BotBufferIdentity,
    BotBufferRegistry,
    ThoughtTemplate,
    maybe_update,
)
from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


Verifier = Callable[[str], VerifierResult]


class BotRuntime:
    def __init__(self, policy: BotStylePolicy | None = None) -> None:
        self.policy = policy or BotStylePolicy()

    def run(
        self,
        *,
        identity: BotBufferIdentity,
        task: TaskInstance,
        buffer_snapshot: list[MemoryEntry],
        client: LLMClient,
        model: str,
        config: dict[str, Any],
        verifier: Verifier,
    ) -> dict[str, Any]:
        call_config = {**config, "sample_id": config.get("sample_id", task.sample_id)}
        trial_id = ":".join(
            [identity.run_id, task.task_name, task.sample_id, identity.baseline, identity.arm, model]
        )
        recorder = MethodCallRecorder(
            client,
            event_callback=call_config.get("_logging_event_callback"),
            trial_context={**call_config.get("_logging_trial_context", {}), "trial_id": trial_id},
        )
        memory = MemoryState(entries=list(buffer_snapshot))
        embedding_provider = call_config.get("embedding_provider", FakeEmbeddingProvider())

        distilled = self.policy.problem_distillation(task, recorder, model, call_config)
        retrieved = _retrieve_top1_template(
            str(task.input), memory.entries, provider=embedding_provider
        )
        final_response = self.policy.template_instantiation_solve(
            task, distilled, memory, recorder, model, call_config, retrieved=retrieved
        )
        answer_call_id = recorder.get_records()[-1].call_id
        verifier_result = verifier(final_response)
        memory_before = [_memory_entry_dict(entry) for entry in buffer_snapshot]

        memory_write_event = None
        memory_after = memory_before
        if verifier_result.is_correct:
            registry = BotBufferRegistry()
            for entry in buffer_snapshot:
                registry.insert(identity, _entry_to_template(entry))
            candidate = _candidate_template(task, final_response, verifier_result, retrieved)
            memory_write_event = maybe_update(
                registry,
                identity,
                candidate,
                retrieved["memory_entry"] if retrieved else None,
                recorder,
                model,
                {
                    **call_config,
                    "verifier_result": verifier_result,
                    "embedding_provider": embedding_provider,
                },
            )
            memory_after = [_template_dict(entry) for entry in registry.snapshot(identity)]

        return {
            "final_response": final_response,
            "parsed_answer": verifier_result.parsed_answer,
            "verifier_result": verifier_result,
            "retrieved_template": _retrieved_template_dict(retrieved),
            "method_calls": recorder.get_records(),
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_write_event": memory_write_event,
            "answer_call_id": answer_call_id,
            "metadata": {
                "bot_buffer_identity": asdict(identity),
                "distilled_problem": distilled,
            },
        }


def _entry_to_template(entry: MemoryEntry) -> ThoughtTemplate:
    return ThoughtTemplate(
        entry_id=entry.entry_id,
        content=entry.content,
        source_trial_id=entry.source_trial_id or "unknown",
        metadata=dict(entry.metadata),
    )


def _candidate_template(
    task: TaskInstance,
    final_response: str,
    verifier_result: VerifierResult,
    retrieved: dict[str, Any] | None,
) -> ThoughtTemplate:
    trial_id = f"{task.task_name}:{task.sample_id}"
    content = distill_thought_template(task, final_response, verifier_result, retrieved)
    source_entry_ids = [retrieved["entry_id"]] if retrieved else []
    return ThoughtTemplate(
        entry_id=f"bot_candidate:{hashlib.sha256((trial_id + final_response).encode()).hexdigest()[:12]}",
        content=content,
        source_trial_id=trial_id,
        source_entry_ids=source_entry_ids,
        metadata={"raw_response": final_response, "distillation_source": "bot_runtime"},
    )


def _memory_entry_dict(entry: MemoryEntry) -> dict[str, Any]:
    return entry.model_dump()


def _template_dict(entry: ThoughtTemplate) -> dict[str, Any]:
    data = asdict(entry)
    if entry.accepted_at is not None:
        data["accepted_at"] = entry.accepted_at.isoformat()
    return data


def _retrieved_template_dict(retrieved: dict[str, Any] | None) -> dict[str, Any] | None:
    if retrieved is None:
        return None
    return {
        "entry_id": retrieved["entry_id"],
        "content": retrieved["content"],
        "score": retrieved["score"],
    }
