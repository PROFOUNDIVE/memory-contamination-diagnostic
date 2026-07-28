from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge import (
    AnswerCallRelation,
    CandidateAssessmentAggregate,
    CandidateExposureRecord,
    ChallengeAssessmentState,
    ChallengeCandidate,
    ChallengeRoutability,
    ChallengeRoutingDecision,
    FilterPolicyIdentity,
    OperationalProbeSuite,
    PairedExecutionIdentity,
    ProbeDisposition,
    ProbeEligibilityState,
    ProbeInventoryManifest,
)


DOMAIN_SCHEMA_VERSION = "filter_challenge_domain_v1"
EXPECTED_PUBLIC_DOMAIN_MODEL_NAMES = (
    "FilterPolicyIdentity",
    "ChallengeCandidate",
    "ChallengeRoutability",
    "ProbeInventoryManifest",
    "OperationalProbeSuite",
    "ProbeEligibilityState",
    "PairedExecutionIdentity",
    "AnswerCallRelation",
    "CandidateExposureRecord",
    "ProbeDisposition",
    "ChallengeAssessmentState",
    "ChallengeRoutingDecision",
    "CandidateAssessmentAggregate",
)
FORBIDDEN_PROPERTIES = (
    "candidate_role",
    "correctness_label",
    "irrelevance_label",
    "B_star_membership",
    "is_injected",
    "origin_class",
    "injection_event_id",
    "treatment_arm",
    "future_main_outcomes",
    "future_suffix_outcomes",
)
ROUTABLE_CANDIDATE = {
    "candidate_entry_id": "candidate-opaque-1",
    "candidate_native_content": "candidate native content",
    "candidate_native_kind": "full_history_transcript",
    "baseline_family": "full_history",
    "rag_mode": "not_applicable",
    "source_checkpoint_id": "checkpoint-1",
    "source_active_state_hash": "active-state-hash-1",
    "routability": {"routability": "challenge_routable_v1", "challenge_suite_key": "suite-key-1"},
}
NOT_EVALUABLE_ASSESSMENT = {
    "assessment_state": "not_evaluable",
    "coverage_satisfied": False,
    "contradiction_satisfied": False,
}
NOT_EVALUABLE_ROUTING = {
    "assessment_state": "not_evaluable",
    "route_target": "active",
    "audit_flag": True,
    "routing_reason_code": "FAIL_OPEN_NOT_EVALUABLE",
}
PUBLIC_SCHEMA_SUBJECTS = (
    FilterPolicyIdentity,
    ChallengeCandidate,
    ChallengeRoutability,
    ProbeInventoryManifest,
    OperationalProbeSuite,
    ProbeEligibilityState,
    PairedExecutionIdentity,
    AnswerCallRelation,
    CandidateExposureRecord,
    ProbeDisposition,
    ChallengeAssessmentState,
    ChallengeRoutingDecision,
    CandidateAssessmentAggregate,
)
PUBLIC_SCHEMAS = (
    FilterPolicyIdentity.model_json_schema(),
    ChallengeCandidate.model_json_schema(),
    TypeAdapter(ChallengeRoutability).json_schema(),
    ProbeInventoryManifest.model_json_schema(),
    OperationalProbeSuite.model_json_schema(),
    TypeAdapter(ProbeEligibilityState).json_schema(),
    TypeAdapter(PairedExecutionIdentity).json_schema(),
    TypeAdapter(AnswerCallRelation).json_schema(),
    CandidateExposureRecord.model_json_schema(),
    TypeAdapter(ProbeDisposition).json_schema(),
    TypeAdapter(ChallengeAssessmentState).json_schema(),
    TypeAdapter(ChallengeRoutingDecision).json_schema(),
    CandidateAssessmentAggregate.model_json_schema(),
)
PUBLIC_CONTRACT_PAYLOADS = (
    (FilterPolicyIdentity, {"decision_rule_id": "rule-1", "failure_mode_id": "fail_open"}),
    (ChallengeCandidate, ROUTABLE_CANDIDATE),
    (ChallengeRoutability, ROUTABLE_CANDIDATE["routability"]),
    (
        ProbeInventoryManifest,
        {
            "calibration_probe_inventory_id": "inventory-1",
            "calibration_probe_inventory_manifest_hash": "hash-1",
            "probe_map_version": "phase12-filter-probe-map-v1",
        },
    ),
    (
        OperationalProbeSuite,
        {"operational_probe_suite_id": "suite-1", "operational_probe_suite_manifest_hash": "hash-1"},
    ),
    (ProbeEligibilityState, {"probe_eligibility_state": "strict_primary_eligible"}),
    (PairedExecutionIdentity, {"paired_execution_identity_status": "matched", "pair_id": "pair-1"}),
    (
        AnswerCallRelation,
        {
            "answer_call_provenance_status": "explicit_matched",
            "answer_call_id": "call-1",
            "parsed_response_source_call_id": "call-1",
            "parser_result_id": "parser-1",
            "verifier_result_id": "verifier-1",
        },
    ),
    (
        CandidateExposureRecord,
        {
            "candidate_entry_id": "candidate-opaque-1",
            "candidate_final_context_inclusion": True,
            "candidate_final_context_source_ids": ("candidate-opaque-1",),
        },
    ),
    (ProbeDisposition, {"probe_disposition": "witness", "reason_code": "VERIFIER_HARM_WITNESS"}),
    (
        ChallengeAssessmentState,
        {"assessment_state": "contradicted", "coverage_satisfied": True, "contradiction_satisfied": True},
    ),
    (
        ChallengeRoutingDecision,
        {
            "assessment_state": "contradicted",
            "route_target": "quarantine",
            "audit_flag": False,
            "routing_reason_code": "CONTRADICTED",
        },
    ),
    (
        CandidateAssessmentAggregate,
        {
            "candidate_entry_id": "candidate-opaque-1",
            "assessment_state": NOT_EVALUABLE_ASSESSMENT,
            "routing_decision": NOT_EVALUABLE_ROUTING,
        },
    ),
)
LEGAL_VARIANT_PAYLOADS = (
    (ChallengeRoutability, {"routability": "unsupported", "reason_code": "PROBE_MAPPING_UNSUPPORTED"}),
    (
        ProbeEligibilityState,
        {"probe_eligibility_state": "canonicalization_sensitivity_eligible", "canonicalizer_version": "canon-1"},
    ),
    (ProbeEligibilityState, {"probe_eligibility_state": "ineligible", "reason_code": "CONTROL_PARSE_FAILURE"}),
    (
        PairedExecutionIdentity,
        {
            "paired_execution_identity_status": "mismatched",
            "pair_id": "pair-1",
            "reason_code": "PAIRED_EXECUTION_IDENTITY_MISMATCH",
        },
    ),
    (AnswerCallRelation, {"answer_call_provenance_status": "missing", "answer_call_id": "call-1"}),
    (AnswerCallRelation, {"answer_call_provenance_status": "ambiguous", "answer_call_id": "call-1"}),
    (
        AnswerCallRelation,
        {"answer_call_provenance_status": "historically_reconstructed", "answer_call_id": "call-1"},
    ),
    (AnswerCallRelation, {"answer_call_provenance_status": "mismatched", "answer_call_id": "call-1"}),
    (
        ProbeDisposition,
        {"probe_disposition": "evaluated_no_witness", "reason_code": "NO_HARM_WITNESS"},
    ),
    (ProbeDisposition, {"probe_disposition": "not_evaluable", "reason_code": "CANDIDATE_NOT_EXPOSED"}),
    (ChallengeAssessmentState, {"assessment_state": "not_contradicted", "coverage_satisfied": True, "contradiction_satisfied": False}),
    (ChallengeAssessmentState, NOT_EVALUABLE_ASSESSMENT),
    (
        ChallengeRoutingDecision,
        {
            "assessment_state": "not_contradicted",
            "route_target": "active",
            "audit_flag": False,
            "routing_reason_code": "NOT_CONTRADICTED",
        },
    ),
    (ChallengeRoutingDecision, NOT_EVALUABLE_ROUTING),
)
INVALID_CONTRACT_PAYLOADS = (
    (ChallengeRoutability, {"routability": "unsupported", "challenge_suite_key": "suite-key-1"}),
    (PairedExecutionIdentity, {"paired_execution_identity_status": "matched", "reason_code": "mismatch"}),
    (AnswerCallRelation, {"answer_call_provenance_status": "explicit_matched", "answer_call_id": "call-1"}),
    (
        CandidateExposureRecord,
        {
            "candidate_entry_id": "candidate-opaque-1",
            "candidate_final_context_inclusion": True,
            "candidate_final_context_source_ids": ("existing-entry",),
        },
    ),
    (ProbeDisposition, {"probe_disposition": "witness", "reason_code": "NO_HARM_WITNESS"}),
    (
        ChallengeAssessmentState,
        {"assessment_state": "not_evaluable", "coverage_satisfied": True, "contradiction_satisfied": False},
    ),
    (
        ChallengeRoutingDecision,
        {"assessment_state": "contradicted", "route_target": "active", "audit_flag": False, "routing_reason_code": "CONTRADICTED"},
    ),
    (
        ChallengeRoutingDecision,
        {"assessment_state": "not_evaluable", "route_target": "active", "audit_flag": False, "routing_reason_code": "FAIL_OPEN_NOT_EVALUABLE"},
    ),
    (
        CandidateAssessmentAggregate,
        {
            "candidate_entry_id": "candidate-opaque-1",
            "assessment_state": {"assessment_state": "contradicted", "coverage_satisfied": True, "contradiction_satisfied": True},
            "routing_decision": NOT_EVALUABLE_ROUTING,
        },
    ),
)
AUDIT_LABELS = {
    "candidate_role": "false",
    "correctness_label": "incorrect",
    "irrelevance_label": "not_irrelevant",
    "B_star_membership": False,
    "is_injected": True,
    "origin_class": "protocol_injected",
    "injection_event_id": "injection-1",
    "treatment_arm": "filter",
    "future_main_outcomes": "unobserved",
    "future_suffix_outcomes": "unobserved",
}
