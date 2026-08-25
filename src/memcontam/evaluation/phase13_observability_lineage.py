from __future__ import annotations

from memcontam.evaluation.phase13_observability_models import (
    Phase13LineageNode,
    Phase13ObservabilityError,
    Phase13TrialEvidence,
)
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY


def validate_evidence_joins(evidence: Phase13TrialEvidence) -> None:
    if (evidence.task, evidence.baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells:
        raise Phase13ObservabilityError("EXCLUDED_CURRENT_MAIN_CELL")
    if (
        evidence.trial.branch_id != evidence.trial.execution_key.arm
        or evidence.order_key != evidence.trial.absolute_trial_index
    ):
        raise Phase13ObservabilityError("TRIAL_CELL_IDENTITY_MISMATCH")
    context = evidence.context
    if context is not None and context.trial_id != evidence.trial_id:
        raise Phase13ObservabilityError("TRIAL_EVENT_IDENTITY_MISMATCH")
    if any(event.trial_id != evidence.trial_id for event in evidence.retrievals):
        raise Phase13ObservabilityError("TRIAL_EVENT_IDENTITY_MISMATCH")
    if context is not None and context.event_id != evidence.trial.context_event_id_or_none:
        raise Phase13ObservabilityError("CONTEXT_EVENT_IDENTITY_MISMATCH")
    if tuple(event.event_id for event in evidence.retrievals) != tuple(
        evidence.trial.retrieval_event_ids
    ):
        raise Phase13ObservabilityError("RETRIEVAL_EVENT_IDENTITY_MISMATCH")
    if tuple(event.memory_id for event in evidence.memory_events) != tuple(
        evidence.trial.memory_event_ids
    ):
        raise Phase13ObservabilityError("WRITER_EVENT_IDENTITY_MISMATCH")
    event_sequences = [event.event_seq for event in evidence.retrievals]
    if context is not None:
        if any(event.event_seq >= context.event_seq for event in evidence.retrievals):
            raise Phase13ObservabilityError("EVENT_ORDER_MISMATCH")
        event_sequences.append(context.event_seq)
    event_sequences.extend(event.event_seq for event in evidence.memory_events)
    if len(event_sequences) != len(set(event_sequences)):
        raise Phase13ObservabilityError("EVENT_ORDER_MISMATCH")
    run_ids = {
        *(event.run_id for event in evidence.retrievals),
        *((context.run_id,) if context is not None else ()),
        *(event.run_id for event in evidence.memory_events),
    }
    if len(run_ids) > 1:
        raise Phase13ObservabilityError("RUN_EVENT_IDENTITY_MISMATCH")
    answer_call_id = evidence.target_set.answer_call_id
    if any(span.parent_call_id != answer_call_id for span in evidence.target_set.answer_call_spans):
        raise Phase13ObservabilityError("ANSWER_CALL_IDENTITY_MISMATCH")
    if any(
        span.target_set_id != evidence.target_set.target_set_id
        or span.entry_id not in evidence.target_set.target_entry_ids
        or span.is_target_contamination is not True
        for span in evidence.target_set.answer_call_spans
    ):
        raise Phase13ObservabilityError("TARGET_SET_IDENTITY_MISMATCH")
    target_ids = set(evidence.target_set.target_entry_ids)
    nodes = {node.entry_id: node for node in evidence.lineage}
    if any(target_id not in nodes or nodes[target_id].lineage_status != "exact" for target_id in target_ids):
        raise Phase13ObservabilityError("EXACT_LINEAGE_REQUIRED")
    if any(
        span.entry_id in target_ids and span.lineage_status != "exact"
        for span in evidence.target_set.answer_call_spans
    ):
        raise Phase13ObservabilityError("EXACT_LINEAGE_REQUIRED")
    changed = (*evidence.new_entry_ids, *evidence.updated_entry_ids, *evidence.removed_entry_ids)
    if not changed:
        return
    if context is None:
        raise Phase13ObservabilityError("MUTATION_CONTEXT_REQUIRED")
    if len(evidence.memory_events) != 1:
        raise Phase13ObservabilityError("WRITER_EVENT_REQUIRED")
    event = evidence.memory_events[0]
    if (
        evidence.trial.memory_event_ids != [event.memory_id]
        or event.trial_id != evidence.trial_id
        or event.source_trial_id != evidence.trial_id
        or event.trial_seq != evidence.order_key
        or event.event_type != "memory_write"
        or event.baseline != evidence.baseline
        or event.status != "completed"
        or event.event_seq <= context.event_seq
        or tuple(event.before_entry_ids) != evidence.memory_before_ids
        or tuple(event.after_entry_ids) != evidence.memory_after_ids
        or tuple(event.new_entry_ids) != evidence.new_entry_ids
        or tuple(event.updated_entry_ids) != evidence.updated_entry_ids
        or tuple(event.removed_entry_ids) != evidence.removed_entry_ids
    ):
        raise Phase13ObservabilityError("WRITER_EVENT_IDENTITY_MISMATCH")
    before = set(evidence.memory_before_ids)
    after = set(evidence.memory_after_ids)
    new = set(evidence.new_entry_ids)
    updated = set(evidence.updated_entry_ids)
    removed = set(evidence.removed_entry_ids)
    if (
        new.intersection(before)
        or not new.issubset(after)
        or not updated.issubset(before.intersection(after))
        or not removed.issubset(before.difference(after))
        or after != before.difference(removed).union(new)
    ):
        raise Phase13ObservabilityError("MEMORY_MUTATION_SET_MISMATCH")
    parents = writer_parents(evidence)
    for entry_id in (*evidence.new_entry_ids, *evidence.updated_entry_ids):
        node = nodes.get(entry_id)
        if node is None:
            raise Phase13ObservabilityError("FABRICATED_LINEAGE")
        recorded = parents.get(entry_id, ())
        asserted = (*node.direct_parent_ids, node.version_predecessor_id)
        asserted_ids = tuple(parent for parent in asserted if parent is not None)
        if not recorded or set(recorded) != set(asserted_ids):
            raise Phase13ObservabilityError("EXACT_LINEAGE_REQUIRED")


def writer_parents(evidence: Phase13TrialEvidence) -> dict[str, tuple[str, ...]]:
    parents: dict[str, list[str]] = {}
    for event in evidence.memory_events:
        for edge in event.lineage_edges:
            if (
                edge.lineage_status != "exact"
                or edge.relation != "recorded_parent"
                or edge.lineage_basis
                not in {"recorded_parent", "recorded_source", "version_edge"}
                or edge.child_entry_id not in (*evidence.new_entry_ids, *evidence.updated_entry_ids)
                or edge.child_entry_id == edge.parent_entry_id
            ):
                raise Phase13ObservabilityError("EXACT_LINEAGE_REQUIRED")
            node = next(
                (item for item in evidence.lineage if item.entry_id == edge.child_entry_id), None
            )
            if node is None or set(edge.injected_root_ids) != set(node.injected_root_ids):
                raise Phase13ObservabilityError("EXACT_LINEAGE_REQUIRED")
            parents.setdefault(edge.child_entry_id, []).append(edge.parent_entry_id)
    return {child: tuple(parent_ids) for child, parent_ids in parents.items()}


def recorded_path(
    node: Phase13LineageNode,
    nodes: dict[str, Phase13LineageNode],
    target_ids: set[str],
    writer_parent_ids: dict[str, tuple[str, ...]],
    seen: set[str],
) -> tuple[str, ...]:
    if node.entry_id in seen:
        raise Phase13ObservabilityError("LINEAGE_CYCLE")
    if node.lineage_status != "exact":
        raise Phase13ObservabilityError("EXACT_LINEAGE_REQUIRED")
    if node.entry_id in target_ids:
        return (node.entry_id,)
    for reference in writer_parent_ids.get(node.entry_id, ()):
        parent = nodes.get(reference)
        if parent is None:
            raise Phase13ObservabilityError("FABRICATED_LINEAGE")
        path = recorded_path(
            parent, nodes, target_ids, writer_parent_ids, seen | {node.entry_id}
        )
        if path:
            return (*path, node.entry_id)
    return ()


__all__ = ["recorded_path", "validate_evidence_joins", "writer_parents"]
