from __future__ import annotations

from memcontam.evaluation.phase13_observability_models import Phase13LineageNode
from memcontam.logging.schema import LineageEdge, MemoryEvent
from memcontam.logging.schema_v3 import ContextEvent
from memcontam.readiness.phase13_production_runtime_models import ProductionRuntimeJoinError


def production_memory_events(
    run_id: str,
    trial_id: str,
    order_key: int,
    baseline: str,
    before_ids: tuple[str, ...],
    after_ids: tuple[str, ...],
    new_ids: tuple[str, ...],
    removed_ids: tuple[str, ...],
    lineage: tuple[Phase13LineageNode, ...],
    context: ContextEvent | None,
) -> tuple[MemoryEvent, ...]:
    if not new_ids and not removed_ids:
        return ()
    if context is None:
        raise ProductionRuntimeJoinError("PRODUCTION_MUTATION_CONTEXT_REQUIRED")
    nodes = {node.entry_id: node for node in lineage}
    edges = [
        LineageEdge(
            child_entry_id=entry_id,
            parent_entry_id=parent_id,
            relation="recorded_parent",
            lineage_status="exact",
            lineage_basis="recorded_source",
            injected_root_ids=list(nodes[entry_id].injected_root_ids),
        )
        for entry_id in new_ids
        for parent_id in nodes[entry_id].direct_parent_ids
    ]
    return (
        MemoryEvent(
            run_metadata_id=f"{run_id}:metadata",
            run_id=run_id,
            trial_id=trial_id,
            trial_seq=order_key,
            event_seq=context.event_seq + 1,
            stage="memory_write",
            memory_id=f"{trial_id}:memory-write",
            event_type="memory_write",
            operation="append",
            baseline=baseline,
            source_trial_id=trial_id,
            parent_entry_ids=[edge.parent_entry_id for edge in edges],
            source_entry_ids=[edge.parent_entry_id for edge in edges],
            contaminated_source_ids=[
                root_id for edge in edges for root_id in edge.injected_root_ids
            ],
            before_entry_ids=list(before_ids),
            after_entry_ids=list(after_ids),
            before_snapshot_hash=None,
            after_snapshot_hash=None,
            new_entry_ids=list(new_ids),
            updated_entry_ids=[],
            removed_entry_ids=list(removed_ids),
            creation_origin="ordinary_runtime",
            memory_version="phase13",
            status="completed",
            created_at="production-runtime",
            lineage_edges=edges,
        ),
    )


__all__ = ["production_memory_events"]
