from __future__ import annotations

from typing import assert_never

from memcontam.evaluation.phase12_observables import (
    TargetSetEvidence,
    compute_observables,
)
from memcontam.evaluation.phase13_aggregate import aggregate_phase13
from memcontam.evaluation.phase13_observability_models import (
    MetricValue,
    Phase13Aggregate,
    Phase13AggregateCell,
    Phase13AggregateTrial,
    Phase13LineageNode,
    Phase13ObservabilityError,
    Phase13TargetSetEvidence,
    Phase13TrialAnalysis,
    Phase13TrialEvidence,
)
from .phase13_observability_lineage import (
    recorded_path as _recorded_path,
    validate_evidence_joins as _validate_evidence_joins,
    writer_parents as _writer_parents,
)
from .phase13_observability_registration import classify_registered_failure
from .phase13_observability_sequence import (
    first_continuous_episode,
    has_registered_recurrence,
    is_exact_root_eviction,
    reconstruct_registered_sequence,
)


def _blocked(reason: str) -> MetricValue:
    return MetricValue(status="unavailable", reason=reason)


def reconstruct_phase13_trial(evidence: Phase13TrialEvidence) -> Phase13TrialAnalysis:
    _validate_evidence_joins(evidence)
    target = evidence.target_set
    observables = compute_observables(
        evidence.trial,
        evidence.retrievals,
        evidence.context,
        TargetSetEvidence(
            target_set_id=target.target_set_id,
            target_entry_ids=target.target_entry_ids,
            answer_call_id=target.answer_call_id,
            answer_call_spans=target.answer_call_spans,
        ),
    )
    arm = evidence.trial.execution_key.arm
    match arm:
        case "clean" | "correct" | "irrelevant" | "contam":
            current_arm = arm
        case "filter":
            raise Phase13ObservabilityError("FILTER_NOT_CURRENT_MAIN_ARM")
        case unreachable:
            assert_never(unreachable)
    target_ids = set(target.target_entry_ids)
    nodes = {node.entry_id: node for node in evidence.lineage}
    roots = tuple(
        node.entry_id
        for node in evidence.lineage
        if node.entry_id in target_ids
        and node.entry_id in node.injected_root_ids
        and not node.direct_parent_ids
        and node.version_predecessor_id is None
    )
    root_ids = set(roots)
    writer_parents = _writer_parents(evidence)
    descendants = tuple(
        node.entry_id
        for node in evidence.lineage
        if node.entry_id not in root_ids
        and root_ids.intersection(node.injected_root_ids)
        and _recorded_path(node, nodes, root_ids, writer_parents, set())
    )
    final_ids = set(observables.final_context.final_entry_ids)
    applicable = current_arm == "contam"
    if not applicable and (target_ids or target.answer_call_spans):
        raise Phase13ObservabilityError("NONCONTAM_TARGET_EVIDENCE")
    if applicable and (
        observables.final_context.is_target_included is None
        or observables.exposure.is_exposed is None
    ):
        raise Phase13ObservabilityError("FINAL_CONTEXT_EVIDENCE_REQUIRED")
    presence = bool(target_ids & set(evidence.memory_before_ids))
    retrieval = observables.retrieval.is_target_retrieved
    inclusion = observables.final_context.is_target_included
    exposure = observables.exposure.is_exposed
    return Phase13TrialAnalysis(
        evidence_scope=evidence.evidence_scope,
        task=evidence.task,
        baseline=evidence.baseline,
        arm=current_arm,
        trajectory_seed=evidence.trajectory_seed,
        concrete_seed_id=evidence.concrete_seed_id,
        analysis_window_id=evidence.analysis_window_id,
        trial_id=evidence.trial_id,
        order_key=evidence.order_key,
        target_set_id=target.target_set_id,
        target_present_in_store_before_answer=_arm_metric(applicable, presence),
        target_retrieved=_arm_metric(applicable, retrieval),
        target_final_context_included=_arm_metric(applicable, inclusion),
        theory_exposure=_arm_metric(applicable, exposure),
        operational_use=MetricValue(
            status="not_registered",
            reason="NOT_REGISTERED_FOR_CURRENT_MAIN",
        ),
        verified_outcome=evidence.verified_outcome,
        failure_class=_blocked("FAILURE_CLASSIFIER_REGISTRY_NOT_REGISTERED"),
        root_entry_ids=roots,
        descendant_entry_ids=descendants,
        memory_before_ids=evidence.memory_before_ids,
        memory_after_ids=evidence.memory_after_ids,
        new_entry_ids=evidence.new_entry_ids,
        updated_entry_ids=evidence.updated_entry_ids,
        removed_entry_ids=evidence.removed_entry_ids,
        generic_recurrence=_blocked("RECURRENCE_LOOKBACK_NOT_REGISTERED"),
        exact_lineage_recurrence=_blocked("RECURRENCE_LOOKBACK_NOT_REGISTERED"),
        exposure_conditioned_recurrence=_blocked("EXPOSURE_CONDITIONING_RULE_NOT_REGISTERED"),
        post_eviction_recurrence=_blocked("POST_EVICTION_TIMING_NOT_REGISTERED"),
        root_storage_persistence=_applicable_presence_metric(
            applicable, roots, evidence.memory_after_ids
        ),
        descendant_storage_persistence=_applicable_descendant_metric(
            applicable, descendants, evidence.memory_after_ids
        ),
        root_prompt_visibility=_applicable_presence_metric(applicable, roots, final_ids),
        descendant_prompt_visibility=_applicable_descendant_metric(
            applicable, descendants, final_ids
        ),
        root_retention_duration=_blocked("RETENTION_DURATION_ENDPOINT_NOT_REGISTERED"),
        prompt_retention_duration=_blocked("RETENTION_DURATION_ENDPOINT_NOT_REGISTERED"),
        descendant_retention_duration=_blocked("RETENTION_DURATION_ENDPOINT_NOT_REGISTERED"),
        propagation=_propagation(
            evidence,
            nodes,
            writer_parents,
            {span.entry_id for span in target.answer_call_spans if span.entry_id in final_ids},
            applicable,
        ),
    )


def _arm_metric(applicable: bool, value: bool | None) -> MetricValue:
    if not applicable:
        return MetricValue(status="not_applicable", reason="ARM_HAS_NO_TARGET_CONTAMINATION")
    if value is None:
        return MetricValue(status="unavailable", reason="TRIAL_EVIDENCE_UNAVAILABLE")
    return MetricValue(
        status="supported", value=value, reason="SYNTHETIC_CONTRACT_FIXTURE_ONLY"
    )


def _presence_metric(entry_ids: tuple[str, ...], container: tuple[str, ...] | set[str]) -> MetricValue:
    return MetricValue(
        status="supported",
        value=bool(set(entry_ids) & set(container)),
        reason="RECORDED_IDENTITY_INTERSECTION",
    )


def _descendant_metric(
    descendants: tuple[str, ...], container: tuple[str, ...] | set[str]
) -> MetricValue:
    if not descendants:
        return MetricValue(status="not_applicable", reason="NO_RECORDED_DESCENDANT")
    return _presence_metric(descendants, container)


def _applicable_presence_metric(
    applicable: bool, entry_ids: tuple[str, ...], container: tuple[str, ...] | set[str]
) -> MetricValue:
    if not applicable:
        return MetricValue(status="not_applicable", reason="ARM_HAS_NO_TARGET_CONTAMINATION")
    return _presence_metric(entry_ids, container)


def _applicable_descendant_metric(
    applicable: bool, descendants: tuple[str, ...], container: tuple[str, ...] | set[str]
) -> MetricValue:
    if not applicable:
        return MetricValue(status="not_applicable", reason="ARM_HAS_NO_TARGET_CONTAMINATION")
    return _descendant_metric(descendants, container)


def _propagation(
    evidence: Phase13TrialEvidence,
    nodes: dict[str, Phase13LineageNode],
    writer_parents: dict[str, tuple[str, ...]],
    exposed_target_ids: set[str],
    applicable: bool,
) -> MetricValue:
    if not applicable:
        return MetricValue(status="not_applicable", reason="ARM_HAS_NO_TARGET_CONTAMINATION")
    match evidence.baseline:
        case "rag_frozen" | "fh_bounded":
            return MetricValue(
                status="not_applicable",
                reason="BASELINE_HAS_NO_QUALIFYING_DESCENDANT_WRITE",
            )
        case "bot_style" | "reflexion_style" | "dc_rs":
            changed = (*evidence.new_entry_ids, *evidence.updated_entry_ids)
            if not changed:
                return MetricValue(status="not_applicable", reason="NO_RECORDED_DESCENDANT_WRITE")
            for entry_id in changed:
                if entry_id in evidence.target_set.target_entry_ids:
                    continue
                node = nodes.get(entry_id)
                if node is None:
                    raise Phase13ObservabilityError("FABRICATED_LINEAGE")
                candidate_roots = exposed_target_ids.intersection(node.injected_root_ids)
                if set(node.injected_root_ids).intersection(evidence.target_set.target_entry_ids):
                    if not exposed_target_ids:
                        raise Phase13ObservabilityError("PROPAGATION_REQUIRES_EXPOSURE")
                    if node.lineage_status != "exact":
                        raise Phase13ObservabilityError("EXACT_LINEAGE_REQUIRED")
                    path = _recorded_path(
                        node, nodes, candidate_roots, writer_parents, set()
                    )
                    if len(path) >= 2:
                        return MetricValue(
                            status="supported",
                            value=True,
                            reason="RECORDED_EXACT_LINEAGE_DESCENDANT",
                            path=path,
                        )
            if exposed_target_ids:
                return MetricValue(
                    status="supported",
                    value=False,
                    reason="NO_EXPOSED_TARGET_ANCESTRY_IN_EXACT_DESCENDANT_WRITE",
                )
            return MetricValue(status="not_applicable", reason="NO_QUALIFYING_RECORDED_LINEAGE")
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "MetricValue",
    "Phase13Aggregate",
    "Phase13AggregateCell",
    "Phase13AggregateTrial",
    "Phase13LineageNode",
    "Phase13ObservabilityError",
    "Phase13TargetSetEvidence",
    "Phase13TrialAnalysis",
    "Phase13TrialEvidence",
    "aggregate_phase13",
    "classify_registered_failure",
    "first_continuous_episode",
    "has_registered_recurrence",
    "is_exact_root_eviction",
    "reconstruct_phase13_trial",
    "reconstruct_registered_sequence",
]
