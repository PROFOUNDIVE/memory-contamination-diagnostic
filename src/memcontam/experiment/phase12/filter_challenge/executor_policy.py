from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

from memcontam.experiment.phase12.filter_challenge.contracts import (
    ChallengeCandidate,
    ChallengeRoutingDecision,
)
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    PairAuditEvidence,
    PairExecutorError,
    SharedAssessmentKey,
)


@dataclass(frozen=True, slots=True)
class ActivationContext:
    policy_activation_checkpoint_id: str
    evolved_branch_checkpoint_id: str | None
    grandfathered_entry_ids: tuple[str, ...]
    controlled_root_entry_id: str
    arm: Literal["contam", "filter"]


@dataclass(frozen=True, slots=True)
class ActivationDecision:
    status: Literal["grandfathered", "assess", "not_evaluable", "not_assessed"]
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class RoutingConsumption:
    effect: Literal["shadow", "apply"]
    route_target: Literal["active", "quarantine"] | None
    shared_assessment_key: SharedAssessmentKey


@dataclass(frozen=True, slots=True)
class PairedRoutingConsumption:
    contam: RoutingConsumption
    filter: RoutingConsumption


def activation_decision(
    context: ActivationContext, candidate: ChallengeCandidate, *, later_native_write: bool
) -> ActivationDecision:
    if candidate.candidate_entry_id in context.grandfathered_entry_ids:
        return ActivationDecision("grandfathered", None)
    if later_native_write:
        checkpoint_id = context.evolved_branch_checkpoint_id
        if checkpoint_id is None or checkpoint_id == context.policy_activation_checkpoint_id:
            raise PairExecutorError("EVOLVED_BRANCH_CHECKPOINT_REQUIRED")
    else:
        checkpoint_id = context.policy_activation_checkpoint_id
    if candidate.source_checkpoint_id != checkpoint_id:
        raise PairExecutorError("CANDIDATE_CHECKPOINT_MISMATCH")
    match candidate.routability.routability:
        case "unsupported":
            return ActivationDecision("not_evaluable", "PROBE_MAPPING_UNSUPPORTED")
        case "challenge_routable_v1":
            if candidate.candidate_entry_id == context.controlled_root_entry_id:
                return ActivationDecision("assess", None)
            if candidate.baseline_family == "rag_frozen" and later_native_write:
                raise PairExecutorError("RAG_FROZEN_LATER_WRITE")
            if later_native_write and context.arm == "filter":
                return ActivationDecision("assess", None)
            return ActivationDecision("not_assessed", None)
        case unreachable:
            assert_never(unreachable)


def consume_routing(
    assessment_id: str,
    routing: ChallengeRoutingDecision,
    shared_assessment_key: SharedAssessmentKey,
    evidence: PairAuditEvidence,
) -> PairedRoutingConsumption:
    if (
        evidence.assessment_id != assessment_id
        or evidence.shared_assessment_key != shared_assessment_key
    ):
        raise PairExecutorError("ASSESSMENT_IDENTITY_MISMATCH")
    return PairedRoutingConsumption(
        RoutingConsumption("shadow", None, shared_assessment_key),
        RoutingConsumption("apply", routing.route_target, shared_assessment_key),
    )
