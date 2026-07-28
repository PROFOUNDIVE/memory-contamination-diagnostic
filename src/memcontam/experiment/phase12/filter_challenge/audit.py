from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeRoutingDecision


DOMAIN_SCHEMA_VERSION: Final = "filter_challenge_domain_v1"


class _StrictAuditModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ChallengeAuditLabels(_StrictAuditModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    candidate_role: str
    correctness_label: str
    irrelevance_label: str
    B_star_membership: bool
    is_injected: bool
    origin_class: str
    injection_event_id: str
    treatment_arm: str
    future_main_outcomes: str
    future_suffix_outcomes: str


class PostRouteAuditJoin(_StrictAuditModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    candidate_entry_id: str
    routing_decision: ChallengeRoutingDecision
    audit_labels: ChallengeAuditLabels
