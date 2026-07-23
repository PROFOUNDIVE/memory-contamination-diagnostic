from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Literal

import pytest

from memcontam.logging.schema import PromptSourceSpan
from memcontam.logging.schema_v3 import (
    ContextEvent,
    MemoryArmExecutionKey,
    MemoryBranchTrialLog,
    RetrievalEvent,
)


RAG_FIXTURE = Path(__file__).parent / "fixtures" / "phase12" / "FX-RAG-001.json"
SEQUENTIAL_FIXTURE = Path(__file__).parent / "fixtures" / "phase12" / "FX-SEQUENTIAL-001.json"
observables = importlib.import_module("memcontam.evaluation.phase12_observables")
Arm = Literal["clean", "correct", "irrelevant", "contam", "filter"]


def _trial(
    arm: Arm = "contam",
    *,
    context_event_id: str | None = "context-1",
    retrieval_event_ids: list[str] | None = None,
    attribution: dict[str, object] | None = None,
) -> MemoryBranchTrialLog:
    return MemoryBranchTrialLog(
        absolute_trial_index=1,
        event_time=1,
        parse_status="parsed",
        execution_status="completed",
        failure_class=None,
        analysis_inclusion="included",
        inclusion_reason="test",
        context_event_id_or_none=context_event_id,
        retrieval_event_ids=retrieval_event_ids or [],
        tool_event_ids=[],
        auxiliary_context_inclusion_or_none=None,
        operational_attribution_or_none=attribution,
        trial_kind="memory_branch",
        execution_key=MemoryArmExecutionKey(kind="memory_arm", arm=arm),
        branch_id=arm,
        prefix_run_id="prefix-1",
        checkpoint_id="checkpoint-1",
        checkpoint_index=1,
        candidate_triplet_id_or_none=None if arm == "clean" else "triplet-1",
        native_render_id_or_none=None if arm == "clean" else "render-1",
        intervention_event_id_or_none=None if arm == "clean" else "intervention-1",
        admission_event_ids=[],
        memory_event_ids=[],
    )


def _retrieval(entry_ids: list[str]) -> RetrievalEvent:
    return RetrievalEvent(
        record_type="retrieval_event",
        event_id="retrieval-1",
        retrieval_id="retrieval-1",
        query_hash="sha256:query",
        retrieved_entry_ids=entry_ids,
        run_id="run-1",
        trial_id="trial-1",
        event_seq=0,
    )


def _context(entry_ids: list[str]) -> ContextEvent:
    return ContextEvent(
        record_type="context_event",
        event_id="context-1",
        context_id="context-1",
        final_entry_ids=entry_ids,
        run_id="run-1",
        trial_id="trial-1",
        event_seq=1,
    )


def _span(entry_id: str) -> PromptSourceSpan:
    return PromptSourceSpan(
        message_index=0,
        start=0,
        end=1,
        rendered_hash="sha256:span",
        entry_id=entry_id,
        source_ids=[entry_id],
        parent_ids=[],
        lineage_id=entry_id,
        version="v1",
        origin="test",
        clean_or_contaminated="contaminated",
        contamination_class="injected",
        injected_root_ids=[entry_id],
        lineage_status="exact",
        lineage_basis="seed",
        target_set_id="targets-v1",
        is_target_contamination=True,
    )


def test_distinguishes_retrieval_final_context_exposure_and_use() -> None:
    rag = json.loads(RAG_FIXTURE.read_text(encoding="utf-8"))
    false_id = next(
        document["id"]
        for document in rag["branch_documents"]["contam"]
        if "false" in document["id"]
    )
    clean_id = rag["branch_documents"]["clean"][0]["id"]
    target_set = observables.TargetSetEvidence(
        target_set_id="targets-v1",
        target_entry_ids=(false_id,),
        answer_call_id="answer-1",
        answer_call_spans=(_span(clean_id),),
    )

    truncated = observables.compute_observables(
        _trial(retrieval_event_ids=["retrieval-1"]),
        [_retrieval([false_id, clean_id])],
        _context([clean_id]),
        target_set,
    )

    assert truncated.retrieval.is_target_retrieved is True
    assert truncated.final_context.is_target_included is False
    assert truncated.exposure.is_exposed is False
    assert truncated.use.is_used is False

    correct_id = next(
        document["id"]
        for document in rag["branch_documents"]["correct"]
        if "correct" in document["id"]
    )
    auxiliary = observables.compute_observables(
        _trial("correct"),
        [],
        _context([correct_id]),
        observables.TargetSetEvidence(target_set_id="targets-v1"),
    )

    assert auxiliary.exposure.status == "not_applicable"
    assert auxiliary.auxiliary_inclusion.included_entry_ids == (correct_id,)

    included_without_attribution = observables.compute_observables(
        _trial(),
        [],
        _context([false_id]),
        observables.TargetSetEvidence(
            target_set_id="targets-v1",
            target_entry_ids=(false_id,),
            answer_call_id="answer-1",
            answer_call_spans=(_span(false_id),),
        ),
    )

    assert included_without_attribution.final_context.is_target_included is True
    assert included_without_attribution.exposure.is_exposed is True
    assert included_without_attribution.use.is_used is False

    deterministic_full_history = observables.compute_observables(
        _trial(context_event_id=None),
        [],
        None,
        observables.TargetSetEvidence(
            target_set_id="targets-v1",
            target_entry_ids=(false_id,),
            answer_call_id="answer-1",
            answer_call_spans=(_span(false_id),),
            deterministic_full_history=True,
        ),
    )

    assert deterministic_full_history.exposure.is_exposed is True
    assert deterministic_full_history.exposure.exposed_non_exposed_contrast_supported is False


def test_rejects_illegal_use_auxiliary_exposure_and_presence_inference() -> None:
    sequential = json.loads(SEQUENTIAL_FIXTURE.read_text(encoding="utf-8"))
    root_id = sequential["events"][0]["final_sources"][0]
    direct_rule = observables.AttributionRule(
        rule_id="direct-uptake",
        version="v1",
        evaluate=lambda evidence: evidence.get("used") is True,
    )

    with pytest.raises(observables.ObservableError, match="U_GT_Z"):
        observables.compute_observables(
            _trial(attribution={"used": True}),
            [],
            _context([]),
            observables.TargetSetEvidence(target_set_id="targets-v1", target_entry_ids=(root_id,)),
            direct_rule,
        )

    with pytest.raises(observables.ObservableError, match="U_GT_Z"):
        observables.compute_observables(
            _trial("clean", attribution={"used": True}),
            [],
            _context([]),
            observables.TargetSetEvidence(target_set_id="targets-v1"),
            direct_rule,
        )

    with pytest.raises(observables.ObservableError, match="AUXILIARY_THEORY_EXPOSURE_FORBIDDEN"):
        observables.compute_observables(
            _trial("correct"),
            [],
            _context([root_id]),
            observables.TargetSetEvidence(
                target_set_id="targets-v1",
                target_entry_ids=(root_id,),
                answer_call_id="answer-1",
                answer_call_spans=(_span(root_id),),
            ),
        )

    with pytest.raises(observables.ObservableError, match="PRESENCE_ONLY_EXPOSURE_INFERENCE"):
        observables.compute_observables(
            _trial(context_event_id=None, retrieval_event_ids=["retrieval-1"]),
            [_retrieval([root_id])],
            None,
            observables.TargetSetEvidence(target_set_id="targets-v1", target_entry_ids=(root_id,)),
        )
