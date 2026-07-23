from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
from memcontam.clients.base import LLMClient
from memcontam.memory.admission import AdmissionContext, AdmissionError
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.filtered_state import (
    CandidateWrite,
    FilterTransition,
    FilteredCheckpoint,
    route_candidate_write,
)
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


__all__ = [
    "BaselineStepResultV3",
    "ReflectionCallLineageEvent",
    "ReflectionEvictionEvent",
    "ReflexionContractError",
    "ReflexionPhase12Adapter",
    "ReflexionStateV3",
    "ReflexionTrialContextV3",
]


Branch = Literal["clean", "correct", "irrelevant", "contam", "filter"]


class ReflexionContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ReflexionTrialContextV3:
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
            raise ReflexionContractError("INVALID_TRIAL_CONTEXT")
        if not isinstance(self.order_key, (int, str)) or isinstance(self.order_key, bool):
            raise ReflexionContractError("INVALID_TRIAL_CONTEXT")
        if self.tool_mode != "text_only" or self.config.get("tool_mode", "text_only") != "text_only":
            raise ReflexionContractError("PRIMARY_TOOL_FORBIDDEN")
        if self.config.get("tools"):
            raise ReflexionContractError("PRIMARY_TOOL_FORBIDDEN")


@dataclass
class ReflexionStateV3:
    reflections: list[MemoryEntry | NativeEntry]
    injected_root_id: str | None = None
    active_capacity: int | None = None
    filter_state: FilteredCheckpoint | None = None
    admission_context: AdmissionContext | None = None
    evicted_reflections: list[MemoryEntry | NativeEntry] = field(default_factory=list)
    first_injected_eviction_trial_id: str | None = None

    def __post_init__(self) -> None:
        if self.active_capacity is not None and (
            type(self.active_capacity) is not int or self.active_capacity < 1
        ):
            raise ReflexionContractError("INVALID_ACTIVE_CAPACITY")
        if (self.filter_state is None) != (self.admission_context is None):
            raise ReflexionContractError("FILTER_ADMISSION_CONTEXT_REQUIRED")
        _validate_reflection_order(self.reflections)
        if self.injected_root_id is not None:
            reflection_ids = _entry_ids(self.reflections)
            if not reflection_ids or reflection_ids[-1] != self.injected_root_id:
                raise ReflexionContractError("INJECTED_REFLECTION_NOT_NEWEST")
        if self.active_capacity is not None and len(_active_reflections(self)) > self.active_capacity:
            raise ReflexionContractError("ACTIVE_CAPACITY_MISCOUNT")


@dataclass(frozen=True)
class ReflectionCallLineageEvent:
    actor_call_id: str
    reflection_call_id: str
    failed_actor_call_id: str
    reflection_entry_id: str


@dataclass(frozen=True)
class ReflectionEvictionEvent:
    entry_id: str
    trial_id: str
    active_capacity: int


@dataclass(frozen=True)
class BaselineStepResultV3:
    outcome: BaselineExecutionOutcome
    native_reflections: tuple[NativeEntry, ...]
    write_envelope: MemoryCardEnvelopeV3 | None
    filter_transition: FilterTransition | None
    call_lineage_events: tuple[ReflectionCallLineageEvent, ...]
    eviction_events: tuple[ReflectionEvictionEvent, ...]


class ReflexionPhase12Adapter:
    def execute(self, trial: ReflexionTrialContextV3, state: ReflexionStateV3) -> BaselineStepResultV3:
        _validate_branch_state(trial, state)
        active = _active_reflections(state)
        native_reflections: list[NativeEntry] = []
        envelopes: list[MemoryCardEnvelopeV3] = []
        transitions: list[FilterTransition] = []
        lineage_events: list[ReflectionCallLineageEvent] = []
        evictions: list[ReflectionEvictionEvent] = []

        def after_reflection(
            entry: MemoryEntry,
            legacy_state: ReflexionState,
            attempt_index: int,
            max_attempts: int,
        ) -> None:
            native, envelope, lineage = _native_reflection(
                entry, trial, legacy_state.reflections, attempt_index
            )
            native_reflections.append(native)
            envelopes.append(envelope)
            lineage_events.append(lineage)
            state.reflections.append(entry.model_copy())
            transition = _route_filter_write(state, native, envelope, trial)
            if transition is not None:
                transitions.append(transition)
                if not transition.decision.admitted and attempt_index < max_attempts:
                    entry.memory_type = "quarantined_reflection"
                    return
            _enforce_active_capacity(state, legacy_state, trial, evictions)

        outcome = ReflexionAdapter().execute(
            trial.task,
            ReflexionState(reflections=list(active)),
            client=trial.client,
            model=trial.model,
            config={
                **dict(trial.config),
                "run_id": trial.run_id,
                "baseline": "reflexion_style",
                "arm": trial.branch,
                "model": trial.model,
                "tool_mode": trial.tool_mode,
                "_phase12_reflection_hook": after_reflection,
            },
            verifier=trial.verifier,
        )
        return BaselineStepResultV3(
            outcome=_active_filter_outcome(outcome, state, transitions),
            native_reflections=tuple(native_reflections),
            write_envelope=envelopes[-1] if envelopes else None,
            filter_transition=transitions[-1] if transitions else None,
            call_lineage_events=tuple(lineage_events),
            eviction_events=tuple(evictions),
        )


def _validate_branch_state(trial: ReflexionTrialContextV3, state: ReflexionStateV3) -> None:
    if (trial.branch == "filter") != (state.filter_state is not None):
        raise ReflexionContractError("FILTER_STATE_BRANCH_MISMATCH")


def _active_reflections(state: ReflexionStateV3) -> list[MemoryEntry]:
    if state.filter_state is None:
        return [_as_reflection(entry) for entry in state.reflections]
    active_ids = _entry_ids(state.filter_state.reader_entries)
    by_id = {entry.entry_id: _as_reflection(entry) for entry in state.reflections}
    if not set(active_ids).issubset(by_id):
        raise ReflexionContractError("FILTER_ACTIVE_REFLECTION_MISSING")
    return [by_id[entry_id] for entry_id in _entry_ids(state.reflections) if entry_id in active_ids]


def _as_reflection(entry: MemoryEntry | NativeEntry) -> MemoryEntry:
    if isinstance(entry, MemoryEntry):
        if entry.memory_type != "verbal_reflection":
            raise ReflexionContractError("INVALID_REFLECTION_STATE")
        return entry
    if not isinstance(entry, NativeEntry) or (
        entry.semantic_kind,
        entry.schema_version,
        entry.native_component,
    ) != ("verbal_reflection", NATIVE_ENTRY_V1, "reflections"):
        raise ReflexionContractError("INVALID_NATIVE_REFLECTION")
    return MemoryEntry(
        entry_id=entry.entry_id,
        content=entry.content,
        memory_type="verbal_reflection",
        metadata={
            "direct_parent_ids": list(entry.direct_parent_ids),
            "memory_support_ids": list(entry.direct_parent_ids),
            "source_entry_ids": list(entry.direct_parent_ids),
        },
    )


def _validate_reflection_order(reflections: Sequence[MemoryEntry | NativeEntry]) -> None:
    prior_ids: set[str] = set()
    for reflection in reflections:
        entry_id = _entry_id(reflection)
        if not entry_id or entry_id in prior_ids:
            raise ReflexionContractError("INVALID_REFLECTION_STATE")
        if not set(_direct_parent_ids(reflection)).issubset(prior_ids):
            raise ReflexionContractError("FUTURE_REFLECTION_ACCESS")
        prior_ids.add(entry_id)


def _native_reflection(
    entry: MemoryEntry,
    trial: ReflexionTrialContextV3,
    visible_reflections: Sequence[MemoryEntry],
    attempt_index: int,
) -> tuple[NativeEntry, MemoryCardEnvelopeV3, ReflectionCallLineageEvent]:
    lineage_parents = _direct_parent_ids(entry)
    supports = _identifier_metadata(entry, "memory_support_ids")
    sources = _identifier_metadata(entry, "source_entry_ids")
    visible_ids = {reflection.entry_id for reflection in visible_reflections if reflection.entry_id != entry.entry_id}
    if supports != sources or (lineage_parents and lineage_parents != supports):
        raise ReflexionContractError("IMPLICIT_PARENT_UNION")
    parents = supports
    if not set(parents).issubset(visible_ids):
        raise ReflexionContractError("FUTURE_REFLECTION_ACCESS")
    failed_actor_call_id = entry.metadata.get("failed_actor_call_id")
    reflection_call_id = entry.metadata.get("creation_call_id")
    parent_call_ids = entry.metadata.get("parent_call_ids")
    if (
        not isinstance(failed_actor_call_id, str)
        or not isinstance(reflection_call_id, str)
        or parent_call_ids != [failed_actor_call_id]
    ):
        raise ReflexionContractError("FAILED_ACTOR_LINEAGE_MISSING")
    native = NativeEntry(
        entry_id=entry.entry_id,
        semantic_kind="verbal_reflection",
        schema_version=NATIVE_ENTRY_V1,
        native_component="reflections",
        content=entry.content,
        content_hash=canonical_content_hash(entry.content),
        direct_parent_ids=parents,
    )
    envelope = MemoryCardEnvelopeV3(
        entry_id=native.entry_id,
        baseline="reflexion_style",
        semantic_kind=native.semantic_kind,
        schema_version=MEMORY_CARD_V3,
        writer_id="reflexion_reflector",
        writer_event_id=f"{trial.trial_id}:reflexion-reflect:{reflection_call_id}",
        writer_stage="reflexion_reflect",
        created_trial_id=trial.trial_id,
        source_trial_ids=(trial.trial_id,),
        source_outcome=False,
        trial_support_ids=(trial.trial_id,),
        memory_support_ids=parents,
        direct_parent_ids=parents,
        version_predecessor_id=None,
        order_key=_reflection_order_key(trial.order_key, attempt_index),
        native_component=native.native_component,
        content=native.content,
        content_hash=native.content_hash,
    )
    return native, envelope, ReflectionCallLineageEvent(
        actor_call_id=failed_actor_call_id,
        reflection_call_id=reflection_call_id,
        failed_actor_call_id=failed_actor_call_id,
        reflection_entry_id=native.entry_id,
    )


def _active_filter_outcome(
    outcome: BaselineExecutionOutcome,
    state: ReflexionStateV3,
    transitions: Sequence[FilterTransition],
) -> BaselineExecutionOutcome:
    if state.filter_state is None:
        return outcome
    write_event = outcome.memory_write_event
    if write_event is not None and transitions and not transitions[-1].decision.admitted:
        write_event = {**write_event, "status": "quarantined"}
    return replace(
        outcome,
        memory_after=tuple(entry.model_dump() for entry in _active_reflections(state)),
        memory_write_event=write_event,
    )


def _route_filter_write(
    state: ReflexionStateV3,
    native: NativeEntry,
    envelope: MemoryCardEnvelopeV3,
    trial: ReflexionTrialContextV3,
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
    try:
        transition = route_candidate_write(state.filter_state, CandidateWrite(native, envelope), context)
    except AdmissionError as error:
        raise ReflexionContractError(error.code) from error
    state.filter_state = transition.state
    state.admission_context = context
    return transition


def _enforce_active_capacity(
    state: ReflexionStateV3,
    legacy_state: ReflexionState,
    trial: ReflexionTrialContextV3,
    evictions: list[ReflectionEvictionEvent],
) -> None:
    if state.active_capacity is None:
        return
    while len(_active_reflections(state)) > state.active_capacity:
        evicted = _active_reflections(state)[0]
        _remove_reflection(state, legacy_state, evicted.entry_id)
        _remove_from_filter_active_state(state, evicted.entry_id)
        state.evicted_reflections.append(evicted)
        if state.injected_root_id == evicted.entry_id and state.first_injected_eviction_trial_id is None:
            state.first_injected_eviction_trial_id = trial.trial_id
        evictions.append(ReflectionEvictionEvent(evicted.entry_id, trial.trial_id, state.active_capacity))


def _remove_reflection(state: ReflexionStateV3, legacy_state: ReflexionState, entry_id: str) -> None:
    state.reflections[:] = [entry for entry in state.reflections if _entry_id(entry) != entry_id]
    legacy_state.reflections[:] = [entry for entry in legacy_state.reflections if entry.entry_id != entry_id]


def _remove_from_filter_active_state(state: ReflexionStateV3, entry_id: str) -> None:
    if state.filter_state is None:
        return
    active = state.filter_state.active.state
    retained_entries = tuple(entry for entry in active.entries if _entry_id(entry) != entry_id)
    if len(retained_entries) == len(active.entries):
        return
    retained_envelopes = tuple(
        envelope for envelope in state.filter_state.active_envelopes if envelope.entry_id != entry_id
    )
    state.filter_state = replace(
        state.filter_state,
        active=serialize_checkpoint(
            NativeState(
                baseline=active.baseline,
                entries=retained_entries,
                native_state=active.native_state,
                schema_version=active.schema_version,
            )
        ),
        active_envelopes=retained_envelopes,
    )


def _entry_ids(entries: Sequence[MemoryEntry | NativeEntry | str]) -> tuple[str, ...]:
    return tuple(_entry_id(entry) for entry in entries)


def _entry_id(entry: MemoryEntry | NativeEntry | str) -> str:
    if isinstance(entry, (MemoryEntry, NativeEntry)):
        return entry.entry_id
    if isinstance(entry, str):
        return entry
    raise ReflexionContractError("INVALID_REFLECTION_STATE")


def _direct_parent_ids(entry: MemoryEntry | NativeEntry) -> tuple[str, ...]:
    if isinstance(entry, NativeEntry):
        return entry.direct_parent_ids
    return _identifier_metadata(entry, "direct_parent_ids")


def _identifier_metadata(entry: MemoryEntry, key: str) -> tuple[str, ...]:
    value = entry.metadata.get(key, ())
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ReflexionContractError("IMPLICIT_PARENT_UNION")
    if len(value) != len(set(value)):
        raise ReflexionContractError("IMPLICIT_PARENT_UNION")
    return tuple(value)


def _reflection_order_key(order_key: int | str, attempt_index: int) -> int | str:
    if isinstance(order_key, int):
        return order_key * 1_000 + attempt_index
    return f"{order_key}:reflexion:{attempt_index:06d}"
