from __future__ import annotations

from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


DOMAIN_SCHEMA_VERSION: Final = "filter_challenge_domain_v1"


class _StrictDomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FilterPolicyIdentity(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    filter_policy_version: Literal["verifier-paired-challenge-v1"] = "verifier-paired-challenge-v1"
    policy_display_name: Literal["Filter-Challenge-v1"] = "Filter-Challenge-v1"
    policy_family: Literal["verifier_backed_paired_challenge"] = "verifier_backed_paired_challenge"
    decision_rule_id: str
    failure_mode_id: Literal["fail_open"]


class _ChallengeRoutable(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    routability: Literal["challenge_routable_v1"]
    challenge_suite_key: str


class _UnsupportedChallenge(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    routability: Literal["unsupported"]
    reason_code: Literal["PROBE_MAPPING_UNSUPPORTED"]


ChallengeRoutability: TypeAlias = Annotated[
    _ChallengeRoutable | _UnsupportedChallenge, Field(discriminator="routability")
]


class ChallengeCandidate(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    candidate_entry_id: str
    candidate_native_content: str
    candidate_native_kind: str
    baseline_family: Literal["full_history", "rag_frozen", "bot_style", "reflexion_style"]
    rag_mode: Literal["frozen", "not_applicable"]
    source_checkpoint_id: str
    source_active_state_hash: str
    routability: ChallengeRoutability


class ProbeInventoryManifest(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str
    probe_map_version: Literal["phase12-filter-probe-map-v1"]


class OperationalProbeSuite(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    operational_probe_suite_id: str
    operational_probe_suite_manifest_hash: str


class _StrictPrimaryEligible(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    probe_eligibility_state: Literal["strict_primary_eligible"]


class _CanonicalizationSensitivityEligible(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    probe_eligibility_state: Literal["canonicalization_sensitivity_eligible"]
    canonicalizer_version: str


class _Ineligible(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    probe_eligibility_state: Literal["ineligible"]
    reason_code: str


ProbeEligibilityState: TypeAlias = Annotated[
    _StrictPrimaryEligible | _CanonicalizationSensitivityEligible | _Ineligible,
    Field(discriminator="probe_eligibility_state"),
]


class _MatchedPair(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    paired_execution_identity_status: Literal["matched"]
    pair_id: str


class _MismatchedPair(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    paired_execution_identity_status: Literal["mismatched"]
    pair_id: str
    reason_code: Literal["PAIRED_EXECUTION_IDENTITY_MISMATCH"]


PairedExecutionIdentity: TypeAlias = Annotated[
    _MatchedPair | _MismatchedPair, Field(discriminator="paired_execution_identity_status")
]


class _ExplicitMatchedRelation(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    answer_call_provenance_status: Literal["explicit_matched"]
    answer_call_id: str
    parsed_response_source_call_id: str
    parser_result_id: str
    verifier_result_id: str

    @model_validator(mode="after")
    def _validate_call_identity(self) -> Self:
        if self.answer_call_id != self.parsed_response_source_call_id:
            raise ValueError("ANSWER_CALL_ID_MISMATCH")
        return self


class _MissingRelation(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    answer_call_provenance_status: Literal["missing"]
    answer_call_id: str


class _AmbiguousRelation(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    answer_call_provenance_status: Literal["ambiguous"]
    answer_call_id: str


class _HistoricalRelation(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    answer_call_provenance_status: Literal["historically_reconstructed"]
    answer_call_id: str


class _MismatchedRelation(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    answer_call_provenance_status: Literal["mismatched"]
    answer_call_id: str


AnswerCallRelation: TypeAlias = Annotated[
    _ExplicitMatchedRelation
    | _MissingRelation
    | _AmbiguousRelation
    | _HistoricalRelation
    | _MismatchedRelation,
    Field(discriminator="answer_call_provenance_status"),
]


class CandidateExposureRecord(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    candidate_entry_id: str
    candidate_final_context_inclusion: bool
    candidate_final_context_source_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_final_context_membership(self) -> Self:
        if self.candidate_final_context_inclusion != (
            self.candidate_entry_id in self.candidate_final_context_source_ids
        ):
            raise ValueError("CANDIDATE_EXPOSURE_MISMATCH")
        return self


class _WitnessDisposition(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    probe_disposition: Literal["witness"]
    reason_code: Literal["VERIFIER_HARM_WITNESS"]


class _NoWitnessDisposition(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    probe_disposition: Literal["evaluated_no_witness"]
    reason_code: Literal["NO_HARM_WITNESS", "OUTPUT_DIVERGENCE_WITHOUT_VERIFIED_HARM"]


class _NotEvaluableDisposition(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    probe_disposition: Literal["not_evaluable"]
    reason_code: Literal[
        "CONTROL_PROVIDER_FAILURE",
        "CONTROL_PARSE_FAILURE",
        "CONTROL_VERIFIER_FAILURE",
        "CONTROL_NOT_CLEAN_SOLVABLE",
        "CHALLENGE_PROVIDER_FAILURE",
        "CHALLENGE_PARSE_FAILURE",
        "CHALLENGE_VERIFIER_FAILURE",
        "CANDIDATE_NOT_EXPOSED",
        "PROBE_MAPPING_UNSUPPORTED",
        "ANSWER_CALL_PROVENANCE_UNRESOLVED",
        "PAIRED_EXECUTION_IDENTITY_MISMATCH",
        "CONTROL_NOT_CLEAN_SOLVABLE_CHALLENGE_BENEFIT",
    ]


ProbeDisposition: TypeAlias = Annotated[
    _WitnessDisposition | _NoWitnessDisposition | _NotEvaluableDisposition,
    Field(discriminator="probe_disposition"),
]


class _ContradictedAssessment(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    assessment_state: Literal["contradicted"]
    coverage_satisfied: Literal[True]
    contradiction_satisfied: Literal[True]


class _NotContradictedAssessment(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    assessment_state: Literal["not_contradicted"]
    coverage_satisfied: Literal[True]
    contradiction_satisfied: Literal[False]


class _NotEvaluableAssessment(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    assessment_state: Literal["not_evaluable"]
    coverage_satisfied: Literal[False]
    contradiction_satisfied: Literal[False]


ChallengeAssessmentState: TypeAlias = Annotated[
    _ContradictedAssessment | _NotContradictedAssessment | _NotEvaluableAssessment,
    Field(discriminator="assessment_state"),
]


class _ContradictedRouting(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    assessment_state: Literal["contradicted"]
    route_target: Literal["quarantine"]
    audit_flag: Literal[False]
    routing_reason_code: Literal["CONTRADICTED"]


class _NotContradictedRouting(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    assessment_state: Literal["not_contradicted"]
    route_target: Literal["active"]
    audit_flag: Literal[False]
    routing_reason_code: Literal["NOT_CONTRADICTED"]


class _FailOpenRouting(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    assessment_state: Literal["not_evaluable"]
    route_target: Literal["active"]
    audit_flag: Literal[True]
    routing_reason_code: Literal["FAIL_OPEN_NOT_EVALUABLE"]


ChallengeRoutingDecision: TypeAlias = Annotated[
    _ContradictedRouting | _NotContradictedRouting | _FailOpenRouting,
    Field(discriminator="assessment_state"),
]


class CandidateAssessmentAggregate(_StrictDomainModel):
    schema_version: Literal["filter_challenge_domain_v1"] = DOMAIN_SCHEMA_VERSION
    candidate_entry_id: str
    assessment_state: ChallengeAssessmentState
    routing_decision: ChallengeRoutingDecision

    @model_validator(mode="after")
    def _validate_routing_state(self) -> Self:
        if self.assessment_state.assessment_state != self.routing_decision.assessment_state:
            raise ValueError("ASSESSMENT_ROUTING_MISMATCH")
        return self
