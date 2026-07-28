from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.bot_runtime import BotRuntime, Verifier
from memcontam.clients.base import LLMClient
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import (
    NATIVE_ENTRY_V1,
    NativeEntry,
    Phase12Checkpoint,
    append_native_entry,
    deserialize_checkpoint,
)
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


class BoTChallengeAdapterError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class BoTChallengeExecution:
    checkpoint: Phase12Checkpoint
    task: TaskInstance
    client: LLMClient
    model: str
    identity: BotBufferIdentity
    config: Mapping[str, object]
    verifier: Verifier | None = None


@dataclass(frozen=True, slots=True)
class BoTChallengeResult:
    outcome: BaselineExecutionOutcome
    provisional_checkpoint: Phase12Checkpoint
    provisional_template_ids: tuple[str, ...]
    selected_template_id: str | None
    nonselected_template_ids: tuple[str, ...]
    displaced_template_ids: tuple[str, ...]
    final_context_source_ids: tuple[str, ...]
    candidate_final_context_inclusion: bool


class BoTStyleChallengeAdapter:
    def execute(
        self, execution: BoTChallengeExecution, candidate: ChallengeCandidate
    ) -> BoTChallengeResult:
        if candidate.candidate_native_kind != "thought_template":
            raise BoTChallengeAdapterError("BOT_CANDIDATE_KIND_INVALID")
        provisional_checkpoint, displaced_template_ids = _provisional_checkpoint(
            execution.checkpoint, candidate
        )
        return _run(
            execution,
            provisional_checkpoint,
            candidate.candidate_entry_id,
            displaced_template_ids,
        )

    def execute_control(self, execution: BoTChallengeExecution) -> BoTChallengeResult:
        _native_entries(execution.checkpoint)
        return _run(execution, execution.checkpoint, None, ())


def _provisional_checkpoint(
    checkpoint: Phase12Checkpoint, candidate: ChallengeCandidate
) -> tuple[Phase12Checkpoint, tuple[str, ...]]:
    entries = _native_entries(checkpoint)
    capacity = deserialize_checkpoint(checkpoint).native_state.get("active_capacity")
    if capacity is not None and len(entries) >= capacity:
        return checkpoint, ()
    candidate_entry = NativeEntry(
        entry_id=candidate.candidate_entry_id,
        semantic_kind="thought_template",
        schema_version=NATIVE_ENTRY_V1,
        native_component="buffer",
        content=candidate.candidate_native_content,
        content_hash=canonical_content_hash(candidate.candidate_native_content),
    )
    return append_native_entry(checkpoint, candidate_entry), ()


def _native_entries(checkpoint: Phase12Checkpoint) -> tuple[NativeEntry, ...]:
    state = deserialize_checkpoint(checkpoint)
    if state.baseline != "bot_style":
        raise BoTChallengeAdapterError("BOT_CHECKPOINT_BASELINE_INVALID")
    if not all(isinstance(entry, NativeEntry) for entry in state.entries):
        raise BoTChallengeAdapterError("BOT_CHECKPOINT_TEMPLATE_INVALID")
    return tuple(entry for entry in state.entries if isinstance(entry, NativeEntry))


def _run(
    execution: BoTChallengeExecution,
    provisional_checkpoint: Phase12Checkpoint,
    candidate_entry_id: str | None,
    displaced_template_ids: tuple[str, ...],
) -> BoTChallengeResult:
    entries = _native_entries(provisional_checkpoint)
    buffer_snapshot = [_as_memory_entry(entry) for entry in entries]
    outcome = BotRuntime().run(
        identity=execution.identity,
        task=execution.task,
        buffer_snapshot=buffer_snapshot,
        client=execution.client,
        model=execution.model,
        config={**execution.config, "update_enabled": False},
        verifier=execution.verifier,
    )
    selected_template_id = _selected_template_id(outcome.metadata)
    template_ids = tuple(entry.entry_id for entry in entries)
    nonselected_template_ids = tuple(
        entry_id for entry_id in template_ids if entry_id != selected_template_id
    )
    final_context_source_ids = _final_context_source_ids(outcome)
    return BoTChallengeResult(
        outcome=outcome,
        provisional_checkpoint=provisional_checkpoint,
        provisional_template_ids=template_ids,
        selected_template_id=selected_template_id,
        nonselected_template_ids=nonselected_template_ids,
        displaced_template_ids=displaced_template_ids,
        final_context_source_ids=final_context_source_ids,
        candidate_final_context_inclusion=(
            candidate_entry_id is not None
            and selected_template_id == candidate_entry_id
            and candidate_entry_id in final_context_source_ids
        ),
    )


def _as_memory_entry(entry: NativeEntry) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry.entry_id,
        content=entry.content,
        memory_type="thought_template",
        metadata={"description": entry.content, "category": "procedure-based"},
    )


def _selected_template_id(metadata: Mapping[str, object]) -> str | None:
    decision = metadata.get("retrieval_decision")
    if not isinstance(decision, Mapping):
        return None
    matched_entry_id = decision.get("matched_entry_id")
    return matched_entry_id if isinstance(matched_entry_id, str) else None


def _final_context_source_ids(outcome: BaselineExecutionOutcome) -> tuple[str, ...]:
    if outcome.answer_call_id is None:
        return ()
    for call in outcome.method_calls:
        if call.call_id == outcome.answer_call_id:
            return tuple(span.entry_id for span in call.source_spans if span.entry_id is not None)
    raise BoTChallengeAdapterError("BOT_ANSWER_CALL_MISSING")
