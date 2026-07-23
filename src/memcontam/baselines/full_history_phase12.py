from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.full_history import FullHistoryState
from memcontam.baselines.full_history_adapter import FullHistoryAdapter
from memcontam.baselines.full_history_context import FullHistoryContextDecision, render_context_bounded_history
from memcontam.clients.base import LLMClient
from memcontam.memory.admission import AdmissionContext
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
    "FullHistoryContractError",
    "FullHistoryFitDecision",
    "FullHistoryPhase12Adapter",
    "FullHistoryRetentionTelemetry",
    "FullHistoryStateV3",
    "TrialContextV3",
    "verify_complete_fit",
]


class FullHistoryContractError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class TrialContextV3:
    task: TaskInstance
    client: LLMClient
    model: str
    trial_id: str
    condition_id: str
    fh_mode: Literal["exact", "bounded"]
    context_config: Mapping[str, Any]
    context_budget_id: str
    order_key: int | str
    verifier: Any = None

    def __post_init__(self) -> None:
        if not self.trial_id or not self.condition_id or not self.context_budget_id:
            raise FullHistoryContractError("INVALID_TRIAL_CONTEXT")
        if not isinstance(self.order_key, (int, str)) or isinstance(self.order_key, bool):
            raise FullHistoryContractError("INVALID_TRIAL_CONTEXT")


@dataclass
class FullHistoryStateV3:
    records: list[MemoryEntry]
    injected_root_id: str | None = None
    injected_root_was_visible: bool = False
    first_eviction_trial_id: str | None = None
    filter_state: FilteredCheckpoint | None = None
    admission_context: AdmissionContext | None = None


@dataclass(frozen=True)
class FullHistoryFitDecision:
    full_fit: bool
    fh_mode: Literal["exact", "bounded"]
    context: FullHistoryContextDecision


@dataclass(frozen=True)
class FullHistoryRetentionTelemetry:
    visible_record_ids: tuple[str, ...]
    pre_record_ids: tuple[str, ...]
    post_record_ids: tuple[str, ...]
    removed_record_ids: tuple[str, ...]
    injected_root_id: str | None
    injected_root_visible: bool
    injected_root_retained: bool
    injected_root_persists_in_store: bool
    first_eviction_trial_id: str | None
    storage_persisted: bool
    context_budget_id: str
    fh_mode: Literal["exact", "bounded"]
    full_fit: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "context_budget_id": self.context_budget_id,
            "fh_mode": self.fh_mode,
            "first_eviction_trial_id": self.first_eviction_trial_id,
            "full_fit": self.full_fit,
            "injected_root_id": self.injected_root_id,
            "injected_root_persists_in_store": self.injected_root_persists_in_store,
            "injected_root_retained": self.injected_root_retained,
            "injected_root_visible": self.injected_root_visible,
            "post_record_ids": list(self.post_record_ids),
            "pre_record_ids": list(self.pre_record_ids),
            "removed_record_ids": list(self.removed_record_ids),
            "storage_persisted": self.storage_persisted,
            "visible_record_ids": list(self.visible_record_ids),
        }


@dataclass(frozen=True)
class BaselineStepResultV3:
    outcome: BaselineExecutionOutcome
    retention: FullHistoryRetentionTelemetry
    write_envelope: MemoryCardEnvelopeV3 | None
    filter_transition: FilterTransition | None


def verify_complete_fit(
    task: TaskInstance,
    records: list[MemoryEntry] | tuple[MemoryEntry, ...],
    context_config: Mapping[str, Any] | None,
    *,
    requested_fh_mode: Literal["exact", "bounded"],
) -> FullHistoryFitDecision:
    config = dict(context_config or {})
    if config.get("eviction_policy", "oldest_first_pair_atomic") != "oldest_first_pair_atomic":
        raise FullHistoryContractError("NON_FIFO_EVICTION_POLICY")
    if requested_fh_mode not in {"exact", "bounded"}:
        raise FullHistoryContractError("INVALID_FH_MODE")
    context = render_context_bounded_history(task, records, config)
    full_fit = not context.removed_record_ids
    if requested_fh_mode == "exact" and not full_fit:
        raise FullHistoryContractError("FALSE_EXACT_LABEL")
    return FullHistoryFitDecision(full_fit=full_fit, fh_mode=requested_fh_mode, context=context)


class FullHistoryPhase12Adapter:
    def execute(self, trial: TrialContextV3, state: FullHistoryStateV3) -> BaselineStepResultV3:
        prompt_records = _prompt_records(state)
        fit = verify_complete_fit(
            trial.task,
            prompt_records,
            trial.context_config,
            requested_fh_mode=trial.fh_mode,
        )
        _record_root_visibility(trial, state, fit.context)

        legacy_state = FullHistoryState(records=list(prompt_records))
        outcome = FullHistoryAdapter().execute(
            trial.task,
            legacy_state,
            client=trial.client,
            model=trial.model,
            config={
                **dict(trial.context_config),
                "arm": "filter" if state.filter_state is not None else "clean",
                "baseline": _baseline_name(trial.fh_mode),
                "model": trial.model,
                "run_id": "phase12",
            },
            verifier=trial.verifier,
        )
        appended = _appended_record(prompt_records, legacy_state.records, trial.trial_id)
        if appended is not None:
            state.records.append(appended)
        envelope = None if appended is None else _write_envelope(appended, trial)
        transition = _route_write(state, envelope, trial)
        retention = _retention_telemetry(trial, state, fit, appended is not None)
        return BaselineStepResultV3(outcome, retention, envelope, transition)


def _prompt_records(state: FullHistoryStateV3) -> list[MemoryEntry]:
    if state.filter_state is None:
        return list(state.records)
    active_ids = {
        entry.entry_id if isinstance(entry, NativeEntry) else entry
        for entry in state.filter_state.reader_entries
    }
    records = [record for record in state.records if record.entry_id in active_ids]
    if {record.entry_id for record in records} != active_ids:
        raise FullHistoryContractError("FILTER_ACTIVE_RECORD_MISSING")
    return records


def _record_root_visibility(
    trial: TrialContextV3,
    state: FullHistoryStateV3,
    context: FullHistoryContextDecision,
) -> None:
    root_id = state.injected_root_id
    if root_id is None:
        return
    root_visible = root_id in context.post_record_ids
    if not state.injected_root_was_visible:
        if not root_visible:
            raise FullHistoryContractError("IMMEDIATE_INJECTED_ROOT_TRUNCATION")
        state.injected_root_was_visible = True
    elif not root_visible and state.first_eviction_trial_id is None:
        state.first_eviction_trial_id = trial.trial_id


def _appended_record(
    prompt_records: list[MemoryEntry],
    records_after_execution: list[MemoryEntry],
    trial_id: str,
) -> MemoryEntry | None:
    if len(records_after_execution) == len(prompt_records):
        return None
    if len(records_after_execution) != len(prompt_records) + 1:
        raise FullHistoryContractError("INVALID_APPEND_ONLY_STORAGE")
    return records_after_execution[-1].model_copy(update={"source_trial_id": trial_id})


def _write_envelope(record: MemoryEntry, trial: TrialContextV3) -> MemoryCardEnvelopeV3:
    return MemoryCardEnvelopeV3(
        entry_id=record.entry_id,
        baseline=_baseline_name(trial.fh_mode),
        semantic_kind="full_history_transcript",
        schema_version=MEMORY_CARD_V3,
        writer_id="fh_appender",
        writer_event_id=f"{trial.trial_id}:full-history-append",
        writer_stage="full_history_generate",
        created_trial_id=trial.trial_id,
        source_trial_ids=(trial.trial_id,),
        source_outcome=None,
        trial_support_ids=(trial.trial_id,),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=trial.order_key,
        native_component="history",
        content=record.content,
        content_hash=canonical_content_hash(record.content),
    )


def _route_write(
    state: FullHistoryStateV3,
    envelope: MemoryCardEnvelopeV3 | None,
    trial: TrialContextV3,
) -> FilterTransition | None:
    if envelope is None or state.filter_state is None:
        return None
    if state.admission_context is None:
        raise FullHistoryContractError("FILTER_ADMISSION_CONTEXT_REQUIRED")
    native_entry = NativeEntry(
        entry_id=envelope.entry_id,
        semantic_kind=envelope.semantic_kind,
        schema_version=NATIVE_ENTRY_V1,
        native_component=envelope.native_component,
        content=envelope.content,
        content_hash=envelope.content_hash,
    )
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


def _retention_telemetry(
    trial: TrialContextV3,
    state: FullHistoryStateV3,
    fit: FullHistoryFitDecision,
    storage_persisted: bool,
) -> FullHistoryRetentionTelemetry:
    root_id = state.injected_root_id
    root_visible = root_id is not None and root_id in fit.context.post_record_ids
    root_in_store = root_id is not None and any(record.entry_id == root_id for record in state.records)
    return FullHistoryRetentionTelemetry(
        visible_record_ids=tuple(fit.context.post_record_ids),
        pre_record_ids=tuple(fit.context.pre_record_ids),
        post_record_ids=tuple(fit.context.post_record_ids),
        removed_record_ids=tuple(fit.context.removed_record_ids),
        injected_root_id=root_id,
        injected_root_visible=root_visible,
        injected_root_retained=root_in_store,
        injected_root_persists_in_store=root_in_store,
        first_eviction_trial_id=state.first_eviction_trial_id,
        storage_persisted=storage_persisted,
        context_budget_id=trial.context_budget_id,
        fh_mode=fit.fh_mode,
        full_fit=fit.full_fit,
    )


def _baseline_name(fh_mode: Literal["exact", "bounded"]) -> Literal["full_history", "fh_bounded"]:
    return "full_history" if fh_mode == "exact" else "fh_bounded"
