from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from memcontam.evaluation.phase13_observability_models import (
    Phase13LineageNode,
    Phase13TargetSetEvidence,
    Phase13TrialEvidence,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun
from memcontam.logging.schema import PromptSourceSpan
from memcontam.logging.schema_v3 import (
    ContextEvent,
    MemoryArmExecutionKey,
    MemoryBranchTrialLog,
    RetrievalEvent,
)
from memcontam.readiness.phase13_production_runtime_memory import production_memory_events
from memcontam.readiness.phase13_production_runtime_models import (
    ProductionOrdinaryRunIdentity,
    ProductionRuntimeJoinError,
)


class _RuntimeEntryMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    source_entry_ids: tuple[str, ...] = ()


class _RuntimeMemoryEntry(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    entry_id: str = Field(min_length=1)
    metadata: _RuntimeEntryMetadata = Field(default_factory=_RuntimeEntryMetadata)


class _FullHistoryContext(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    post_record_ids: tuple[str, ...]
    removed_record_ids: tuple[str, ...] = ()


@runtime_checkable
class _SourceSpanCall(Protocol):
    @property
    def call_id(self) -> str: ...

    @property
    def source_spans(self) -> Sequence[PromptSourceSpan]: ...


def build_production_trial_evidence(
    run: ProspectiveOrdinaryRun,
    result: RuntimeTrialResult,
    identity: ProductionOrdinaryRunIdentity,
    sample_id: str,
    suffix_order: int,
    checkpoint_index: int,
) -> Phase13TrialEvidence:
    assert run.branch is not None
    if run.baseline == "nomem":
        raise ProductionRuntimeJoinError("PRODUCTION_MEMORY_BASELINE_REQUIRED")
    current_trial_id = trial_id(run, sample_id, suffix_order)
    absolute_index = checkpoint_index + suffix_order
    before = _entries(result.outcome.memory_before)
    after = _entries(result.outcome.memory_after)
    before_ids = tuple(entry.entry_id for entry in before)
    after_ids = tuple(entry.entry_id for entry in after)
    new_ids = tuple(entry_id for entry_id in after_ids if entry_id not in before_ids)
    removed_ids = tuple(entry_id for entry_id in before_ids if entry_id not in after_ids)
    target_ids = (
        (run.branch.injected_root_id,)
        if run.arm == "contam" and run.branch.injected_root_id is not None
        else ()
    )
    target_set_id = f"{identity.execution_template_id}:target"
    if result.retrieval_event is not None and not isinstance(
        result.retrieval_event, RetrievalEvent
    ):
        raise ProductionRuntimeJoinError("PRODUCTION_RETRIEVAL_EVENT_INVALID")
    retrievals = () if result.retrieval_event is None else (result.retrieval_event,)
    if result.context_event is not None and not isinstance(result.context_event, ContextEvent):
        raise ProductionRuntimeJoinError("PRODUCTION_CONTEXT_EVENT_INVALID")
    context = _context(
        result.context_event,
        result.outcome.metadata,
        run.run_id,
        current_trial_id,
        retrievals,
    )
    lineage = _lineage((*before, *after), target_ids)
    memory_events = production_memory_events(
        run.run_id,
        current_trial_id,
        absolute_index,
        run.baseline,
        before_ids,
        after_ids,
        new_ids,
        removed_ids,
        lineage,
        context,
    )
    trial = MemoryBranchTrialLog(
        absolute_trial_index=absolute_index,
        event_time=suffix_order - 1,
        parse_status="parsed" if result.outcome.parsed_answer is not None else "unparsed",
        execution_status="completed" if result.outcome.status == "succeeded" else "failed",
        failure_class=result.outcome.error_type,
        analysis_inclusion="included" if result.outcome.status == "succeeded" else "excluded",
        inclusion_reason=(
            "production_runtime_join"
            if result.outcome.status == "succeeded"
            else "terminal_technical_missingness"
        ),
        context_event_id_or_none=None if context is None else context.event_id,
        retrieval_event_ids=[event.event_id for event in retrievals],
        tool_event_ids=[],
        auxiliary_context_inclusion_or_none=None,
        operational_attribution_or_none=None,
        trial_kind="memory_branch",
        execution_key=MemoryArmExecutionKey(kind="memory_arm", arm=run.arm),
        branch_id=run.arm,
        prefix_run_id=run.branch.prefix_identity,
        checkpoint_id=run.branch.checkpoint.identity.checkpoint_id,
        checkpoint_index=checkpoint_index,
        candidate_triplet_id_or_none=run.branch.candidate_triplet_id,
        native_render_id_or_none=run.branch.native_render_id,
        intervention_event_id_or_none=(
            None if run.arm == "clean" else f"{current_trial_id}:intervention"
        ),
        admission_event_ids=[],
        memory_event_ids=[event.memory_id for event in memory_events],
    )
    return Phase13TrialEvidence(
        evidence_scope="production_runtime",
        task=run.task_name,
        baseline=run.baseline,
        trajectory_seed=identity.trajectory_seed,
        concrete_seed_id=identity.concrete_seed_id,
        analysis_window_id=identity.analysis_window_id,
        trial_id=current_trial_id,
        order_key=absolute_index,
        trial=trial,
        retrievals=retrievals,
        context=context,
        target_set=Phase13TargetSetEvidence(
            target_set_id=target_set_id,
            target_entry_ids=target_ids,
            answer_call_id=result.outcome.answer_call_id,
            answer_call_spans=_target_spans(
                result.outcome.method_calls, target_ids, target_set_id
            ),
            source_package_manifest_sha256=identity.source_package_manifest_sha256,
        ),
        verified_outcome=(
            None
            if result.outcome.status == "failed"
            else (1 if result.outcome.verifier_result is True else 0)
        ),
        memory_before_ids=before_ids,
        memory_after_ids=after_ids,
        new_entry_ids=new_ids,
        removed_entry_ids=removed_ids,
        memory_events=memory_events,
        lineage=lineage,
    )


def trial_id(run: ProspectiveOrdinaryRun, sample_id: str, suffix_order: int) -> str:
    arm = "" if run.arm == "clean" else f":{run.arm}"
    return f"{run.run_id}{arm}:trial:{suffix_order}:{sample_id}"


def _entries(rows: Sequence[Mapping[str, JsonValue]]) -> tuple[_RuntimeMemoryEntry, ...]:
    return tuple(_RuntimeMemoryEntry.model_validate(row) for row in rows)


def _context(
    value: ContextEvent | None,
    metadata: Mapping[str, JsonValue],
    run_id: str,
    current_trial_id: str,
    retrievals: tuple[RetrievalEvent, ...],
) -> ContextEvent | None:
    if isinstance(value, ContextEvent):
        return value
    raw = metadata.get("full_history_context")
    if not isinstance(raw, Mapping):
        return None
    recorded = _FullHistoryContext.model_validate(raw)
    return ContextEvent(
        record_type="context_event",
        event_id=f"{current_trial_id}:context",
        context_id=f"{current_trial_id}:context",
        final_entry_ids=list(recorded.post_record_ids),
        removed_entry_ids=list(recorded.removed_record_ids),
        run_id=run_id,
        trial_id=current_trial_id,
        event_seq=max((event.event_seq for event in retrievals), default=-1) + 1,
    )


def _target_spans(
    calls: Sequence[JsonValue], target_ids: tuple[str, ...], target_set_id: str
) -> tuple[PromptSourceSpan, ...]:
    if not target_ids:
        return ()
    target = set(target_ids)
    spans: list[PromptSourceSpan] = []
    for call in calls:
        if not isinstance(call, _SourceSpanCall):
            continue
        for span in call.source_spans:
            if span.entry_id in target:
                spans.append(span.model_copy(update={
                    "parent_call_id": call.call_id,
                    "clean_or_contaminated": "contaminated",
                    "contamination_class": "injected",
                    "injected_root_ids": [span.entry_id],
                    "lineage_status": "exact",
                    "lineage_basis": "seed",
                    "target_set_id": target_set_id,
                    "is_target_contamination": True,
                }))
    return tuple(spans)


def _lineage(
    entries: Sequence[_RuntimeMemoryEntry], target_ids: tuple[str, ...]
) -> tuple[Phase13LineageNode, ...]:
    targets = set(target_ids)
    by_id = {entry.entry_id: entry for entry in entries}
    return tuple(
        Phase13LineageNode(
            entry_id=entry.entry_id,
            lineage_status="exact",
            injected_root_ids=(
                (entry.entry_id,)
                if entry.entry_id in targets
                else tuple(target for target in target_ids if target in entry.metadata.source_entry_ids)
            ),
            direct_parent_ids=entry.metadata.source_entry_ids,
        )
        for entry in by_id.values()
    )


__all__ = ["build_production_trial_evidence", "trial_id"]
