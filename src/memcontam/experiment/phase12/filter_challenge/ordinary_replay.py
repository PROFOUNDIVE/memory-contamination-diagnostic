from __future__ import annotations

import json
from typing import Final, Mapping, TypeAlias

from memcontam.baselines.bot_phase12 import BoTPhase12Adapter, BoTStateV3, BoTTrialContextV3
from memcontam.baselines.full_history_phase12 import (
    FullHistoryPhase12Adapter,
    FullHistoryStateV3,
    TrialContextV3,
)
from memcontam.baselines.reflexion_phase12 import (
    ReflexionPhase12Adapter,
    ReflexionStateV3,
    ReflexionTrialContextV3,
)
from memcontam.clients.replay import ReplayClient
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, Phase12Checkpoint
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
MODEL: Final = "gpt-4o-2024-11-20"


class OrdinaryAuthorityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _NovelEmbedder:
    def __init__(self, candidate: str) -> None:
        self.candidate = candidate

    def encode_document(self, text: str) -> list[float]:
        return [1.0, 0.0] if text == self.candidate else [0.0, 1.0]

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def _full_history(
    task: TaskInstance, trial_id: str, content: str, checkpoint: Phase12Checkpoint
) -> tuple[NativeEntry, MemoryCardEnvelopeV3, dict[str, JsonValue], dict[str, JsonValue]]:
    records = [_memory_entry(entry) for entry in checkpoint.state.entries]
    client = ReplayClient(responses_by_sample={task.sample_id: {"full_history_generate": f"final: {content}"}})
    result = FullHistoryPhase12Adapter().execute(
        TrialContextV3(
            task, client, MODEL, trial_id, "ordinary", "exact",
            {"_deterministic_memory_entry_id": f"ordinary-{task.task_name}-full-history-position-9"},
            "ordinary-build", 9,
        ),
        FullHistoryStateV3(records),
    )
    envelope = _required_envelope(result.write_envelope)
    native = _native_from_envelope(envelope)
    return native, envelope, _writer_event(result.outcome.memory_write_event, envelope), _interaction(
        task, trial_id, result.outcome, "ordinary_interaction"
    )


def _bot(
    task: TaskInstance, trial_id: str, content: str, checkpoint: Phase12Checkpoint
) -> tuple[NativeEntry, MemoryCardEnvelopeV3, dict[str, JsonValue], dict[str, JsonValue]]:
    responses = {
        "bot_problem_distill": json.dumps(
            {"distilled_task": "realize the registered false family", "key_information": content, "restrictions": "position 9"},
            sort_keys=True, separators=(",", ":"),
        ),
        "bot_instantiate_solve": json.dumps(
            {"final_answer": f"final: {content}", "selected_structure": "procedure-based", "solution_trace": content},
            sort_keys=True, separators=(",", ":"),
        ),
        "bot_thought_distill": json.dumps(
            {"category": "procedure-based", "description": "registered ordinary false family", "explicitly_used_memory_ids": [], "template": content},
            sort_keys=True, separators=(",", ":"),
        ),
    }
    client = ReplayClient(responses_by_sample={task.sample_id: responses})
    initial = _native_entries(checkpoint)
    result = BoTPhase12Adapter().execute(
        BoTTrialContextV3(
            task, client, MODEL, "fv5-ordinary", trial_id, "ordinary", "clean",
            {"embedding_provider": _NovelEmbedder(content), "tool_mode": "text_only"}, 9,
            verifier=lambda _answer: True,
        ),
        BoTStateV3(initial, tuple(_entry_id(entry) for entry in initial)),
    )
    if result.native_entry is None:
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    envelope = _required_envelope(result.write_envelope)
    return result.native_entry, envelope, _writer_event(result.outcome.memory_write_event, envelope), _interaction(
        task, trial_id, result.outcome, "native_distillation_and_novelty"
    )


def _reflexion(
    task: TaskInstance, trial_id: str, content: str, checkpoint: Phase12Checkpoint
) -> tuple[NativeEntry, MemoryCardEnvelopeV3, dict[str, JsonValue], dict[str, JsonValue]]:
    responses = {
        "reflexion_generate": "final: deterministic-wrong-answer",
        "reflexion_reflect": json.dumps(
            {"mode": "corrective", "failure_class": "incorrect_answer", "reflection_text": content, "explicitly_used_memory_ids": []},
            sort_keys=True, separators=(",", ":"),
        ),
    }
    result = ReflexionPhase12Adapter().execute(
        ReflexionTrialContextV3(
            task, ReplayClient(responses_by_sample={task.sample_id: responses}), MODEL,
            "fv5-ordinary", trial_id, "ordinary", "clean",
            {"max_attempts": 1, "_deterministic_memory_entry_id": f"ordinary-{task.task_name}-reflexion-position-9"},
            9, verifier=lambda _answer, _task: False,
        ),
        ReflexionStateV3(_native_entries(checkpoint)),
    )
    if len(result.native_reflections) != 1:
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    envelope = _required_envelope(result.write_envelope)
    interaction = _interaction(task, trial_id, result.outcome, "failed_interaction_then_reflection")
    interaction["failed_actor_call_id"] = result.call_lineage_events[0].failed_actor_call_id
    return result.native_reflections[0], envelope, _writer_event(
        result.outcome.memory_write_event, envelope
    ), interaction


def _rag(
    task: TaskInstance, trial_id: str, content: str, checkpoint: Phase12Checkpoint
) -> tuple[NativeEntry, MemoryCardEnvelopeV3, dict[str, JsonValue], dict[str, JsonValue]]:
    entry_id = f"ordinary-{task.task_name}-rag-frozen-position-9"
    native = NativeEntry(
        entry_id, "rag_document", NATIVE_ENTRY_V1, "corpus", content,
        canonical_content_hash(content), render_id=f"rag-corpus-ingestion-{task.task_name}-position-9",
    )
    event_id = f"{trial_id}:rag-corpus-load"
    envelope = MemoryCardEnvelopeV3(
        entry_id, "rag_frozen", "rag_document", MEMORY_CARD_V3, "rag_corpus_loader",
        event_id, "rag_corpus_load", None, (), None, (), (), (), None, 9, "corpus",
        content, native.content_hash,
    )
    event: dict[str, JsonValue] = {
        "event_id": event_id, "entry_id": entry_id, "writer_stage": "rag_corpus_load",
        "status": "ingested", "corpus_version": "branch-corpus-v3",
    }
    interaction: dict[str, JsonValue] = {
        "interaction_id": f"{trial_id}:corpus-source", "position": 9, "task": task.task_name,
        "source_type": "frozen_corpus_ingestion", "trial_id": None, "verifier_result": None,
        "method_calls": [], "provider_transport": "deterministic_replay",
    }
    return native, envelope, event, interaction


def _interaction(task: TaskInstance, trial_id: str, outcome, source_type: str) -> dict[str, JsonValue]:
    return {
        "interaction_id": trial_id,
        "position": 9,
        "task": task.task_name,
        "source_type": source_type,
        "trial_id": trial_id,
        "verifier_result": outcome.verifier_result,
        "method_calls": [_json_mapping(call.model_dump(mode="json")) for call in outcome.method_calls],
        "provider_transport": "deterministic_replay",
    }


def _writer_event(
    value: Mapping[str, object] | None, envelope: MemoryCardEnvelopeV3
) -> dict[str, JsonValue]:
    if value is None:
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    return {
        **_json_mapping(dict(value)),
        "event_id": envelope.writer_event_id,
        "entry_id": envelope.entry_id,
        "writer_stage": envelope.writer_stage,
    }


def _required_envelope(value: MemoryCardEnvelopeV3 | None) -> MemoryCardEnvelopeV3:
    if value is None:
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    return value


def _native_from_envelope(envelope: MemoryCardEnvelopeV3) -> NativeEntry:
    return NativeEntry(
        envelope.entry_id, envelope.semantic_kind, NATIVE_ENTRY_V1, envelope.native_component,
        envelope.content, envelope.content_hash, envelope.direct_parent_ids,
    )


def _memory_entry(value: str | NativeEntry) -> MemoryEntry:
    if not isinstance(value, NativeEntry):
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    return MemoryEntry(entry_id=value.entry_id, content=value.content, memory_type=value.semantic_kind)


def _native_entries(checkpoint: Phase12Checkpoint) -> list[MemoryEntry | NativeEntry]:
    entries: list[MemoryEntry | NativeEntry] = []
    for value in checkpoint.state.entries:
        if not isinstance(value, NativeEntry):
            raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
        entries.append(value)
    return entries


def _entry_id(value: str | MemoryEntry | NativeEntry) -> str:
    return value.entry_id if isinstance(value, (MemoryEntry, NativeEntry)) else value


def _json_mapping(value: Mapping[str, object]) -> dict[str, JsonValue]:
    normalized = json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":")), parse_float=str
    )
    if not isinstance(normalized, dict):
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    return normalized
