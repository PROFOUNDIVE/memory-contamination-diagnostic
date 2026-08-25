from __future__ import annotations

import importlib
from types import ModuleType
from typing import Literal

from memcontam.logging.schema import LineageEdge, MemoryEvent, PromptSourceSpan
from memcontam.logging.schema_v3 import ContextEvent, MemoryArmExecutionKey, MemoryBranchTrialLog, RetrievalEvent


def module() -> ModuleType:
    return importlib.import_module("memcontam.evaluation.phase13_observability")


def trial(
    *,
    retrieved: bool = False,
    arm: Literal["clean", "correct", "irrelevant", "contam"] = "contam",
    memory_event: bool = False,
) -> MemoryBranchTrialLog:
    clean = arm == "clean"
    return MemoryBranchTrialLog(
        absolute_trial_index=1,
        event_time=1,
        parse_status="parsed",
        execution_status="completed",
        failure_class=None,
        analysis_inclusion="included",
        inclusion_reason="fixture",
        context_event_id_or_none="context-1",
        retrieval_event_ids=["retrieval-1"] if retrieved else [],
        tool_event_ids=[],
        auxiliary_context_inclusion_or_none=None,
        operational_attribution_or_none=None,
        trial_kind="memory_branch",
        execution_key=MemoryArmExecutionKey(kind="memory_arm", arm=arm),
        branch_id=arm,
        prefix_run_id="prefix-1",
        checkpoint_id="checkpoint-1",
        checkpoint_index=1,
        candidate_triplet_id_or_none=None if clean else "triplet-1",
        native_render_id_or_none=None if clean else "render-1",
        intervention_event_id_or_none=None if clean else "intervention-1",
        admission_event_ids=[],
        memory_event_ids=["memory-1"] if memory_event else [],
    )


def retrieval() -> RetrievalEvent:
    return RetrievalEvent(
        record_type="retrieval_event",
        event_id="retrieval-1",
        retrieval_id="retrieval-1",
        query_hash="sha256:query",
        retrieved_entry_ids=["root-b"],
        run_id="run-1",
        trial_id="trial-1",
        event_seq=0,
    )


def context(final_ids: list[str]) -> ContextEvent:
    return ContextEvent(
        record_type="context_event",
        event_id="context-1",
        context_id="context-1",
        final_entry_ids=final_ids,
        run_id="run-1",
        trial_id="trial-1",
        event_seq=1,
    )


def span(entry_id: str) -> PromptSourceSpan:
    return PromptSourceSpan(
        message_index=0,
        start=0,
        end=1,
        rendered_hash="sha256:span",
        entry_id=entry_id,
        parent_call_id="answer-1",
        source_ids=[entry_id],
        parent_ids=[],
        lineage_id=entry_id,
        version="v1",
        origin="fixture",
        clean_or_contaminated="contaminated",
        contamination_class="injected",
        injected_root_ids=[entry_id],
        lineage_status="exact",
        lineage_basis="seed",
        target_set_id="targets-v1",
        is_target_contamination=True,
    )


def memory_event(
    before: tuple[str, ...], after: tuple[str, ...], new: tuple[str, ...]
) -> MemoryEvent:
    return MemoryEvent(
        run_metadata_id="metadata-1",
        run_id="run-1",
        trial_id="trial-1",
        trial_seq=1,
        event_seq=2,
        stage="memory_update",
        memory_id="memory-1",
        event_type="memory_write",
        operation="update",
        baseline="bot_style",
        source_trial_id="trial-1",
        parent_entry_ids=["root-b"],
        source_entry_ids=["root-b"],
        contaminated_source_ids=["root-b"],
        before_entry_ids=list(before),
        after_entry_ids=list(after),
        before_snapshot_hash="sha256:before",
        after_snapshot_hash="sha256:after",
        new_entry_ids=list(new),
        updated_entry_ids=[],
        removed_entry_ids=[],
        creation_origin="fixture",
        memory_version="v1",
        status="completed",
        created_at="2026-08-25T00:00:00Z",
        lineage_edges=[
            LineageEdge(
                child_entry_id=entry_id,
                parent_entry_id="root-b",
                relation="recorded_parent",
                lineage_status="exact",
                lineage_basis="recorded_parent",
                injected_root_ids=["root-b"],
            )
            for entry_id in new
        ],
    )


def evidence(module: ModuleType, *, retrieved: bool, included: bool, verified: int):
    return module.Phase13TrialEvidence(
        evidence_scope="synthetic_contract_fixture",
        task="game24",
        baseline="rag_frozen",
        trajectory_seed=0,
        concrete_seed_id="game24-seed-0",
        analysis_window_id="core_prefix_50",
        trial_id="trial-1",
        order_key=1,
        trial=trial(retrieved=retrieved),
        retrievals=(retrieval(),) if retrieved else (),
        context=context(["root-b"] if included else []),
        target_set=module.Phase13TargetSetEvidence(
            target_set_id="targets-v1",
            target_entry_ids=("root-b",),
            answer_call_id="answer-1",
            answer_call_spans=(span("root-b"),) if included else (),
        ),
        verified_outcome=verified,
        memory_before_ids=("root-b",),
        memory_after_ids=("root-b",),
        lineage=(
            module.Phase13LineageNode(
                entry_id="root-b",
                lineage_status="exact",
                injected_root_ids=("root-b",),
            ),
        ),
    )


__all__ = ["context", "evidence", "memory_event", "module", "retrieval", "span", "trial"]
