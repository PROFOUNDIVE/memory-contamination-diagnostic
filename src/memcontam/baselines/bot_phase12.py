from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

from memcontam.baselines.bot_runtime import BotRuntime
from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.clients.base import LLMClient
from memcontam.logging.schema_v3 import ContextEvent, RetrievalEvent
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.bot_buffer import BotBufferIdentity, NativeNoveltyDecision
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry
from memcontam.memory.filtered_state import (
    CandidateWrite,
    FilteredCheckpoint,
    FilterTransition,
    route_candidate_write,
)
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


__all__ = [
    "BaselineStepResultV3",
    "BoTContractError",
    "BoTPhase12Adapter",
    "BoTPromptDecision",
    "BoTStateV3",
    "BoTTrialContextV3",
    "resolve_explicit_parents",
]


Branch = Literal["clean", "correct", "irrelevant", "contam", "filter"]
LineageStatus = Literal["exact", "unavailable", "approximate"]


class BoTContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class BoTTrialContextV3:
    task: TaskInstance
    client: LLMClient
    model: str
    run_id: str
    trial_id: str
    condition_id: str
    branch: Branch
    config: Mapping[str, Any]
    order_key: int | str
    verifier: Any = None
    tool_mode: Literal["text_only"] = "text_only"

    def __post_init__(self) -> None:
        if not all((self.run_id, self.trial_id, self.condition_id)):
            raise BoTContractError("INVALID_TRIAL_CONTEXT")
        if not isinstance(self.order_key, (int, str)) or isinstance(self.order_key, bool):
            raise BoTContractError("INVALID_TRIAL_CONTEXT")
        if self.config.get("tool_mode", "text_only") != "text_only":
            raise BoTContractError("PRIMARY_TOOL_FORBIDDEN")


@dataclass
class BoTStateV3:
    entries: list[MemoryEntry | NativeEntry]
    clean_competitor_ids: tuple[str, ...] = ()
    active_capacity: int | None = None
    filter_state: FilteredCheckpoint | None = None
    admission_context: AdmissionContext | None = None

    def __post_init__(self) -> None:
        if len(self.clean_competitor_ids) != len(set(self.clean_competitor_ids)) or any(
            not isinstance(entry_id, str) or not entry_id for entry_id in self.clean_competitor_ids
        ):
            raise BoTContractError("INVALID_CLEAN_COMPETITOR_IDS")
        if self.active_capacity is not None and (
            type(self.active_capacity) is not int or self.active_capacity < 1
        ):
            raise BoTContractError("INVALID_ACTIVE_CAPACITY")
        if self.filter_state is not None and self.admission_context is None:
            raise BoTContractError("FILTER_ADMISSION_CONTEXT_REQUIRED")


@dataclass(frozen=True)
class BoTPromptDecision:
    decision: Literal["matched", "fallback"]
    matched_entry_id: str | None


@dataclass(frozen=True)
class BaselineStepResultV3:
    outcome: BaselineExecutionOutcome
    retrieval_event: RetrievalEvent
    context_event: ContextEvent
    prompt_decision: BoTPromptDecision
    native_novelty_decision: NativeNoveltyDecision
    native_entry: NativeEntry | None
    write_envelope: MemoryCardEnvelopeV3 | None
    filter_transition: FilterTransition | None
    lineage_status: LineageStatus


class BoTPhase12Adapter:
    def execute(self, trial: BoTTrialContextV3, state: BoTStateV3) -> BaselineStepResultV3:
        _validate_branch_state(trial, state)
        active_entries = _active_entries(state)
        _validate_clean_competitors(state, active_entries)
        outcome = BotRuntime().run(
            identity=BotBufferIdentity(
                trial.run_id, trial.task.task_name, "bot_style", trial.branch, trial.model
            ),
            task=trial.task,
            buffer_snapshot=active_entries,
            client=trial.client,
            model=trial.model,
            config={**dict(trial.config), "tool_mode": trial.tool_mode},
            verifier=trial.verifier,
        )
        prompt_decision, retrieval_event, context_event = _prompt_events(trial, outcome)
        novelty = _native_novelty(outcome)
        candidate = _candidate_entry(outcome, novelty)
        if candidate is None:
            return BaselineStepResultV3(
                outcome,
                retrieval_event,
                context_event,
                prompt_decision,
                novelty,
                None,
                None,
                None,
                "unavailable",
            )

        native_entry, envelope, lineage_status, normalized_candidate = _native_write(
            candidate, state, trial
        )
        if _at_active_capacity(state, active_entries):
            return BaselineStepResultV3(
                _replace_candidate(outcome, normalized_candidate),
                retrieval_event,
                context_event,
                prompt_decision,
                novelty,
                None,
                None,
                None,
                lineage_status,
            )

        state.entries.append(normalized_candidate)
        transition = _route_filter_write(state, native_entry, envelope, trial)
        return BaselineStepResultV3(
            _replace_candidate(outcome, normalized_candidate),
            retrieval_event,
            context_event,
            prompt_decision,
            novelty,
            native_entry,
            envelope,
            transition,
            lineage_status,
        )


def resolve_explicit_parents(ids: Sequence[str], state: BoTStateV3) -> tuple[str, ...]:
    if len(ids) != len(set(ids)) or any(not isinstance(entry_id, str) or not entry_id for entry_id in ids):
        raise BoTContractError("INVALID_EXPLICIT_PARENT_IDS")
    known_ids = {entry.entry_id for entry in state.entries}
    return tuple(entry_id for entry_id in ids if entry_id in known_ids)


def _validate_branch_state(trial: BoTTrialContextV3, state: BoTStateV3) -> None:
    if (trial.branch == "filter") != (state.filter_state is not None):
        raise BoTContractError("FILTER_STATE_BRANCH_MISMATCH")


def _active_entries(state: BoTStateV3) -> list[MemoryEntry]:
    allowed_ids = None
    if state.filter_state is not None:
        allowed_ids = {
            entry.entry_id if isinstance(entry, NativeEntry) else entry
            for entry in state.filter_state.reader_entries
        }
    entries = [_as_memory_entry(entry) for entry in state.entries]
    active = entries if allowed_ids is None else [entry for entry in entries if entry.entry_id in allowed_ids]
    if allowed_ids is not None and {entry.entry_id for entry in active} != allowed_ids:
        raise BoTContractError("FILTER_ACTIVE_TEMPLATE_MISSING")
    return active


def _validate_clean_competitors(state: BoTStateV3, active_entries: Sequence[MemoryEntry]) -> None:
    active_ids = {entry.entry_id for entry in active_entries}
    if len(set(state.clean_competitor_ids) & active_ids) < 2:
        raise BoTContractError("BOT_COMPETITORS_UNAVAILABLE")


def _as_memory_entry(entry: MemoryEntry | NativeEntry) -> MemoryEntry:
    if isinstance(entry, MemoryEntry):
        return entry
    if not isinstance(entry, NativeEntry) or (
        entry.semantic_kind,
        entry.native_component,
        entry.schema_version,
    ) != ("thought_template", "buffer", NATIVE_ENTRY_V1):
        raise BoTContractError("INVALID_NATIVE_TEMPLATE")
    return MemoryEntry(
        entry_id=entry.entry_id,
        content=entry.content,
        memory_type="thought_template",
        metadata={"description": entry.content, "category": "procedure-based"},
    )


def _prompt_events(
    trial: BoTTrialContextV3, outcome: BaselineExecutionOutcome
) -> tuple[BoTPromptDecision, RetrievalEvent, ContextEvent]:
    decision_data = outcome.metadata.get("retrieval_decision")
    if not isinstance(decision_data, dict):
        raise BoTContractError("RETRIEVAL_DECISION_MISSING")
    matched_entry_id = decision_data.get("matched_entry_id")
    if not isinstance(matched_entry_id, str):
        matched_entry_id = None
    matched = decision_data.get("decision") == "matched"
    if matched != (matched_entry_id is not None):
        raise BoTContractError("RETRIEVAL_DECISION_MISMATCH")
    prompt_entry_ids = _final_prompt_entry_ids(outcome)
    if tuple(prompt_entry_ids) != (() if matched_entry_id is None else (matched_entry_id,)):
        raise BoTContractError("FINAL_PROMPT_INCLUSION_MISMATCH")
    score = decision_data.get("top_similarity")
    scores = [float(score)] if matched and isinstance(score, (int, float)) else []
    retrieval = RetrievalEvent(
        record_type="retrieval_event",
        event_id=f"{trial.trial_id}:retrieval",
        run_id=trial.run_id,
        trial_id=trial.trial_id,
        event_seq=0,
        retrieval_id=f"{trial.trial_id}:retrieval",
        query_hash="bot-local-retrieval",
        retrieved_entry_ids=[] if matched_entry_id is None else [matched_entry_id],
        retrieved_scores=scores,
    )
    context = ContextEvent(
        record_type="context_event",
        event_id=f"{trial.trial_id}:context",
        run_id=trial.run_id,
        trial_id=trial.trial_id,
        event_seq=1,
        context_id=f"{trial.trial_id}:context",
        final_entry_ids=prompt_entry_ids,
    )
    return BoTPromptDecision("matched" if matched else "fallback", matched_entry_id), retrieval, context


def _final_prompt_entry_ids(outcome: BaselineExecutionOutcome) -> list[str]:
    if outcome.answer_call_id is None:
        return []
    for call in outcome.method_calls:
        if call.call_id == outcome.answer_call_id:
            return [span.entry_id for span in call.source_spans if span.entry_id is not None]
    raise BoTContractError("ANSWER_CALL_MISSING")


def _native_novelty(outcome: BaselineExecutionOutcome) -> NativeNoveltyDecision:
    event = outcome.memory_write_event
    if not isinstance(event, dict) or event.get("status") not in {"accepted", "rejected_novelty"}:
        raise BoTContractError("NATIVE_NOVELTY_DECISION_MISSING")
    return NativeNoveltyDecision(
        admitted=event["status"] == "accepted",
        compared_entry_id=event.get("top_existing_entry_id"),
        top_similarity=event.get("top_similarity"),
    )


def _candidate_entry(
    outcome: BaselineExecutionOutcome, novelty: NativeNoveltyDecision
) -> MemoryEntry | None:
    if not novelty.admitted:
        return None
    if not outcome.memory_after:
        raise BoTContractError("NATIVE_CANDIDATE_MISSING")
    return MemoryEntry.model_validate(outcome.memory_after[-1])


def _native_write(
    candidate: MemoryEntry, state: BoTStateV3, trial: BoTTrialContextV3
) -> tuple[NativeEntry, MemoryCardEnvelopeV3, LineageStatus, MemoryEntry]:
    explicit_ids = tuple(candidate.metadata.get("explicitly_used_memory_ids", ()))
    declared_ids = tuple(candidate.metadata.get("direct_parent_ids", ()))
    if declared_ids != explicit_ids:
        raise BoTContractError("VISIBILITY_ONLY_PARENT")
    parents = resolve_explicit_parents(explicit_ids, state)
    lineage_status: LineageStatus
    if not explicit_ids:
        lineage_status = "unavailable"
    elif parents == explicit_ids:
        lineage_status = "exact"
    else:
        lineage_status = "approximate"
    exact_parents = parents if lineage_status == "exact" else ()
    normalized = candidate.model_copy(
        update={
            "metadata": {
                **candidate.metadata,
                "direct_parent_ids": list(exact_parents),
                "memory_support_ids": list(exact_parents),
                "source_entry_ids": list(exact_parents),
                "lineage_basis": "explicitly_used_memory_ids" if exact_parents else "unavailable",
                "lineage_status": lineage_status,
                "source_outcome": None,
            }
        }
    )
    native = NativeEntry(
        entry_id=normalized.entry_id,
        semantic_kind="thought_template",
        schema_version=NATIVE_ENTRY_V1,
        native_component="buffer",
        content=normalized.content,
        content_hash=canonical_content_hash(normalized.content),
        direct_parent_ids=exact_parents,
    )
    envelope = MemoryCardEnvelopeV3(
        entry_id=native.entry_id,
        baseline="bot_style",
        semantic_kind=native.semantic_kind,
        schema_version=MEMORY_CARD_V3,
        writer_id="bot_buffer_manager",
        writer_event_id=f"{trial.trial_id}:bot-thought-distill",
        writer_stage="bot_thought_distill",
        created_trial_id=trial.trial_id,
        source_trial_ids=(trial.trial_id,),
        source_outcome=None,
        trial_support_ids=(trial.trial_id,),
        memory_support_ids=exact_parents,
        direct_parent_ids=exact_parents,
        version_predecessor_id=None,
        order_key=trial.order_key,
        native_component="buffer",
        content=native.content,
        content_hash=native.content_hash,
    )
    return native, envelope, lineage_status, normalized


def _at_active_capacity(state: BoTStateV3, active_entries: Sequence[MemoryEntry]) -> bool:
    return state.active_capacity is not None and len(active_entries) >= state.active_capacity


def _route_filter_write(
    state: BoTStateV3,
    native_entry: NativeEntry,
    envelope: MemoryCardEnvelopeV3,
    trial: BoTTrialContextV3,
) -> FilterTransition | None:
    if state.filter_state is None:
        return None
    assert state.admission_context is not None
    context = replace(
        state.admission_context,
        writer_event_ids=state.admission_context.writer_event_ids | {envelope.writer_event_id},
        trial_record_ids=state.admission_context.trial_record_ids | {trial.trial_id},
        evidence_envelopes=(*state.admission_context.evidence_envelopes, envelope),
    )
    transition = route_candidate_write(state.filter_state, CandidateWrite(native_entry, envelope), context)
    state.filter_state = transition.state
    state.admission_context = context
    return transition


def _replace_candidate(outcome: BaselineExecutionOutcome, candidate: MemoryEntry) -> BaselineExecutionOutcome:
    return replace(
        outcome,
        memory_after=(*outcome.memory_after[:-1], candidate.model_dump()),
        memory_write_event={
            **(outcome.memory_write_event or {}),
            "direct_parent_ids": list(candidate.metadata["direct_parent_ids"]),
            "memory_support_ids": list(candidate.metadata["memory_support_ids"]),
            "source_outcome": None,
        },
    )
