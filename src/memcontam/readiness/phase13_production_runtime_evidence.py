from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memcontam.evaluation.phase13_observability_lineage import recorded_path
from memcontam.evaluation.phase13_observability_models import (
    Phase13LineageNode,
    Phase13TargetSetEvidence,
    Phase13TrialEvidence,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun
from memcontam.logging.schema import MethodCall, PromptSourceSpan, VerifierResult
from memcontam.logging.schema_v3 import (
    ContextEvent,
    MemoryArmExecutionKey,
    MemoryBranchTrialLog,
    NoMemExecutionKey,
    NoMemTrialLog,
    RetrievalEvent,
)
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState
from memcontam.readiness.phase13_production_runtime_memory import production_memory_events
from memcontam.readiness.phase13_production_runtime_models import (
    ProductionNoMemTrialEvidence,
    ProductionOrdinaryRunIdentity,
    ProductionRuntimeJoinError,
)


class _RuntimeEntryMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    source_entry_ids: tuple[str, ...] = ()
    source_lineage_status: Literal["exact", "approximate", "unavailable"] = "exact"


class _RuntimeMemoryEntry(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    entry_id: str = Field(min_length=1)
    metadata: _RuntimeEntryMetadata = Field(default_factory=_RuntimeEntryMetadata)


class _FullHistoryContext(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    post_record_ids: tuple[str, ...]
    removed_record_ids: tuple[str, ...] = ()


def build_production_trial_evidence(
    run: ProspectiveOrdinaryRun,
    result: RuntimeTrialResult,
    identity: ProductionOrdinaryRunIdentity,
    sample_id: str,
    suffix_order: int,
    checkpoint_index: int | None,
) -> Phase13TrialEvidence | ProductionNoMemTrialEvidence:
    if run.baseline == "nomem":
        return _build_nomem_trial_evidence(run, result, identity, sample_id, suffix_order)
    if run.branch is None or checkpoint_index is None:
        raise ProductionRuntimeJoinError("PRODUCTION_CHECKPOINT_REQUIRED")
    current_trial_id = trial_id(run, sample_id, suffix_order)
    absolute_index = checkpoint_index + suffix_order
    before = _snapshot_entries(result.state_before, result.outcome.memory_before)
    after = _snapshot_entries(result.state_after, result.outcome.memory_after)
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
        result.outcome.method_calls,
        result.outcome.answer_call_id,
    )
    lineage = _lineage(
        (
            *(_entry_from_native(entry) for entry in result.provenance_entries),
            *before,
            *after,
            *_entries(result.outcome.memory_before),
            *_entries(result.outcome.memory_after),
        ),
        target_ids,
        result.provenance_envelopes,
        new_ids,
    )
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
                result.outcome.method_calls,
                result.outcome.answer_call_id,
                target_ids,
                target_set_id,
                lineage,
            ),
            source_package_manifest_sha256=identity.source_package_manifest_sha256,
        ),
        verified_outcome=_verified_outcome(result),
        memory_before_ids=before_ids,
        memory_after_ids=after_ids,
        new_entry_ids=new_ids,
        removed_entry_ids=removed_ids,
        memory_events=memory_events,
        lineage=lineage,
    )


def _build_nomem_trial_evidence(
    run: ProspectiveOrdinaryRun,
    result: RuntimeTrialResult,
    identity: ProductionOrdinaryRunIdentity,
    sample_id: str,
    suffix_order: int,
) -> ProductionNoMemTrialEvidence:
    current_trial_id = trial_id(run, sample_id, suffix_order)
    succeeded = result.outcome.status == "succeeded"
    return ProductionNoMemTrialEvidence(
        evidence_scope="production_runtime",
        task=run.task_name,
        baseline="nomem",
        trajectory_seed=identity.trajectory_seed,
        concrete_seed_id=identity.concrete_seed_id,
        analysis_window_id=identity.analysis_window_id,
        trial_id=current_trial_id,
        order_key=suffix_order,
        trial=NoMemTrialLog(
            absolute_trial_index=suffix_order,
            event_time=suffix_order - 1,
            parse_status="parsed" if result.outcome.parsed_answer else "unparsed",
            execution_status="completed" if succeeded else "failed",
            failure_class=result.outcome.error_type,
            analysis_inclusion="included" if succeeded else "excluded",
            inclusion_reason=(
                "production_runtime_join" if succeeded else "terminal_technical_missingness"
            ),
            context_event_id_or_none=None,
            retrieval_event_ids=[],
            tool_event_ids=[],
            auxiliary_context_inclusion_or_none=None,
            operational_attribution_or_none=None,
            trial_kind="nomem_singleton",
            execution_key=NoMemExecutionKey(kind="nomem_singleton", key="*"),
        ),
        verified_outcome=_verified_outcome(result),
    )


def trial_id(run: ProspectiveOrdinaryRun, sample_id: str, suffix_order: int) -> str:
    arm = "" if run.arm == "clean" else f":{run.arm}"
    return f"{run.run_id}{arm}:trial:{suffix_order}:{sample_id}"


def _verified_outcome(result: RuntimeTrialResult) -> Literal[0, 1] | None:
    if result.outcome.status == "failed":
        return None
    match result.outcome.verifier_result:
        case VerifierResult(is_correct=correct):
            return 1 if correct else 0
        case bool() as correct:
            return 1 if correct else 0
        case _:
            raise ProductionRuntimeJoinError("PRODUCTION_VERIFIER_RESULT_MISSING")


def _entries(rows: Sequence[Mapping[str, JsonValue]]) -> tuple[_RuntimeMemoryEntry, ...]:
    return tuple(_RuntimeMemoryEntry.model_validate(row) for row in rows)


def _entry_from_native(entry: NativeEntry) -> _RuntimeMemoryEntry:
    return _RuntimeMemoryEntry(
        entry_id=entry.entry_id,
        metadata=_RuntimeEntryMetadata(source_entry_ids=entry.direct_parent_ids,
                                       source_lineage_status=entry.lineage_status),
    )


def _snapshot_entries(
    snapshot: NativeState | None,
    fallback: Sequence[Mapping[str, JsonValue]],
) -> tuple[_RuntimeMemoryEntry, ...]:
    if snapshot is None:
        return _entries(fallback)
    return tuple(
        _entry_from_native(entry)
        if isinstance(entry, NativeEntry)
        else _RuntimeMemoryEntry(entry_id=entry)
        for entry in snapshot.entries
    )


def _context(
    value: ContextEvent | None,
    metadata: Mapping[str, JsonValue],
    run_id: str,
    current_trial_id: str,
    retrievals: tuple[RetrievalEvent, ...],
    calls: Sequence[JsonValue | MethodCall],
    answer_call_id: str | None,
) -> ContextEvent | None:
    if isinstance(value, ContextEvent):
        return value
    raw = metadata.get("full_history_context")
    if isinstance(raw, Mapping):
        recorded = _FullHistoryContext.model_validate(raw)
        final_entry_ids = list(recorded.post_record_ids)
        removed_entry_ids = list(recorded.removed_record_ids)
    else:
        answer_calls = tuple(
            call
            for call in calls
            if isinstance(call, MethodCall) and call.call_id == answer_call_id
        )
        if not answer_calls:
            return None
        final_entry_ids = list(
            dict.fromkeys(span.entry_id for span in answer_calls[-1].source_spans)
        )
        removed_entry_ids = []
    return ContextEvent(
        record_type="context_event",
        event_id=f"{current_trial_id}:context",
        context_id=f"{current_trial_id}:context",
        final_entry_ids=final_entry_ids,
        removed_entry_ids=removed_entry_ids,
        run_id=run_id,
        trial_id=current_trial_id,
        event_seq=max((event.event_seq for event in retrievals), default=-1) + 1,
    )


def _target_spans(
    calls: Sequence[JsonValue | MethodCall],
    answer_call_id: str | None,
    target_ids: tuple[str, ...],
    target_set_id: str,
    lineage: Sequence[Phase13LineageNode],
) -> tuple[PromptSourceSpan, ...]:
    if not target_ids:
        return ()
    target = set(target_ids)
    nodes = {node.entry_id: node for node in lineage}
    spans: list[PromptSourceSpan] = []
    for call in calls:
        if not isinstance(call, MethodCall) or call.call_id != answer_call_id:
            continue
        for span in call.source_spans:
            node = nodes.get(span.entry_id)
            recorded_roots = () if node is None else node.injected_root_ids
            matched_roots = tuple(
                root_id
                for root_id in target_ids
                if root_id == span.entry_id
                or root_id in recorded_roots
                or root_id in span.injected_root_ids
            )
            if matched_roots:
                direct_root = span.entry_id in target
                if (
                    node is None
                    or node.lineage_status == "approximate"
                    or (direct_root and node.injected_root_ids != (span.entry_id,))
                    or (
                        span.injected_root_ids
                        and set(node.injected_root_ids) != set(span.injected_root_ids)
                    )
                    or any(
                        not recorded_path(node, nodes, {root_id}, {}, set())
                        for root_id in matched_roots
                    )
                ):
                    raise ProductionRuntimeJoinError("PRODUCTION_TARGET_LINEAGE_INVALID")
                spans.append(
                    span.model_copy(
                        update={
                            "parent_call_id": call.call_id,
                            "clean_or_contaminated": "contaminated",
                            "contamination_class": "injected" if direct_root else "derived",
                            "injected_root_ids": list(matched_roots),
                            "lineage_status": "exact",
                            "lineage_basis": (
                                "seed"
                                if direct_root
                                else "recorded_source"
                                if node.direct_parent_ids
                                else "version_edge"
                            ),
                            "direct_parent_ids": list(node.direct_parent_ids),
                            "target_set_id": target_set_id,
                            "is_target_contamination": True,
                        }
                    )
                )
    return tuple(spans)


def _lineage(
    entries: Sequence[_RuntimeMemoryEntry],
    target_ids: tuple[str, ...],
    envelopes: Sequence[MemoryCardEnvelopeV3] = (),
    new_ids: tuple[str, ...] = (),
) -> tuple[Phase13LineageNode, ...]:
    targets = set(target_ids)
    by_id = {entry.entry_id: entry for entry in entries}
    by_envelope = {envelope.entry_id: envelope for envelope in envelopes}
    references: dict[str, tuple[str, ...]] = {}
    statuses: dict[str, Literal["exact", "approximate", "unavailable"]] = {}
    for entry in by_id.values():
        envelope = by_envelope.get(entry.entry_id)
        if envelope is not None and envelope.lineage_status != entry.metadata.source_lineage_status:
            raise ProductionRuntimeJoinError("PRODUCTION_LINEAGE_STATUS_MISMATCH")
        statuses[entry.entry_id] = entry.metadata.source_lineage_status
        parent_ids = (
            entry.metadata.source_entry_ids if envelope is None else envelope.direct_parent_ids
        )
        predecessor_id = None if envelope is None else envelope.version_predecessor_id
        if entry.entry_id in new_ids and (
            envelope is None
            or envelope.created_trial_id is None
            or envelope.created_trial_id not in envelope.source_trial_ids
        ):
            raise ProductionRuntimeJoinError("PRODUCTION_WRITER_ORIGIN_MISSING")
        if any(
            parent_id not in by_id
            for parent_id in (*parent_ids, predecessor_id)
            if parent_id is not None
        ):
            raise ProductionRuntimeJoinError("PRODUCTION_LINEAGE_PARENT_MISSING")
        references[entry.entry_id] = (
            *parent_ids,
            *((predecessor_id,) if predecessor_id is not None else ()),
        )

    memo: dict[str, tuple[str, ...]] = {}

    def injected_roots(entry_id: str, visiting: frozenset[str] = frozenset()) -> tuple[str, ...]:
        if entry_id in targets:
            return (entry_id,)
        if entry_id in memo:
            return memo[entry_id]
        if entry_id in visiting:
            raise ProductionRuntimeJoinError("PRODUCTION_LINEAGE_CYCLE")
        discovered = {
            root_id
            for parent_id in references[entry_id]
            for root_id in injected_roots(parent_id, visiting | {entry_id})
        }
        result = tuple(root_id for root_id in target_ids if root_id in discovered)
        memo[entry_id] = result
        return result

    status_memo: dict[str, Literal["exact", "approximate", "unavailable"]] = {}

    def ancestry_status(entry_id: str) -> Literal["exact", "approximate", "unavailable"]:
        if entry_id not in status_memo:
            inherited = {statuses[entry_id], *(ancestry_status(parent) for parent in references[entry_id])}
            status_memo[entry_id] = (
                "unavailable" if "unavailable" in inherited
                else "approximate" if "approximate" in inherited else "exact"
            )
        return status_memo[entry_id]

    for entry_id in by_id:
        injected_roots(entry_id)

    return tuple(
        Phase13LineageNode(
            entry_id=entry.entry_id,
            lineage_status=ancestry_status(entry.entry_id),
            injected_root_ids=(injected_roots(entry.entry_id)),
            direct_parent_ids=(
                by_envelope[entry.entry_id].direct_parent_ids
                if entry.entry_id in by_envelope
                else entry.metadata.source_entry_ids
            ),
            version_predecessor_id=(
                by_envelope[entry.entry_id].version_predecessor_id
                if entry.entry_id in by_envelope
                else None
            ),
        )
        for entry in by_id.values()
    )


__all__ = ["build_production_trial_evidence", "trial_id"]
