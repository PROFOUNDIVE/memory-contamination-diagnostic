from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeAssessmentState,
    ChallengeRoutability,
    ChallengeRoutingDecision,
    PairedExecutionIdentity,
    ProbeDisposition,
    ProbeEligibilityState,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import KappaCandidate, SuiteCandidate


ELIGIBILITY_STATES: Final = (
    "strict_primary_eligible",
    "canonicalization_sensitivity_eligible",
    "ineligible",
)
DISPOSITION_STATES: Final = ("witness", "evaluated_no_witness", "not_evaluable")
ASSESSMENT_STATES: Final = ("contradicted", "not_contradicted", "not_evaluable")
DISPOSITION_REASON_CODES: Final = (
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
    "VERIFIER_HARM_WITNESS",
    "NO_HARM_WITNESS",
    "OUTPUT_DIVERGENCE_WITHOUT_VERIFIED_HARM",
    "CONTROL_NOT_CLEAN_SOLVABLE_CHALLENGE_BENEFIT",
)
ROUTE_TABLE: Final = (
    ("contradicted", "quarantine", False, "CONTRADICTED"),
    ("not_contradicted", "active", False, "NOT_CONTRADICTED"),
    ("not_evaluable", "active", True, "FAIL_OPEN_NOT_EVALUABLE"),
)

_ELIGIBILITY_ADAPTER: Final[TypeAdapter[ProbeEligibilityState]] = TypeAdapter(ProbeEligibilityState)
_DISPOSITION_ADAPTER: Final[TypeAdapter[ProbeDisposition]] = TypeAdapter(ProbeDisposition)
_ASSESSMENT_ADAPTER: Final[TypeAdapter[ChallengeAssessmentState]] = TypeAdapter(ChallengeAssessmentState)
_ROUTING_ADAPTER: Final[TypeAdapter[ChallengeRoutingDecision]] = TypeAdapter(ChallengeRoutingDecision)


class AssessmentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProbeAssessmentInput:
    probe_id: str
    control_provider_status: Literal["success", "failed"]
    control_raw_parse_status: Literal["parsed_raw", "parse_failed"]
    control_verifier_status: Literal["success", "failed"] | None
    control_verifier_result: bool | None
    control_relation: AnswerCallRelation
    control_canonicalizer_version: str | None
    control_canonicalized_parse_status: Literal["parsed_raw", "parse_failed"] | None
    control_canonicalized_verifier_status: Literal["success", "failed"] | None
    control_canonicalized_verifier_result: bool | None
    challenge_provider_status: Literal["success", "failed"]
    challenge_raw_parse_status: Literal["parsed_raw", "parse_failed"]
    challenge_verifier_status: Literal["success", "failed"] | None
    challenge_verifier_result: bool | None
    challenge_relation: AnswerCallRelation
    candidate_exposure: CandidateExposureRecord
    routability: ChallengeRoutability
    pair_identity: PairedExecutionIdentity
    outputs_diverge: bool


@dataclass(frozen=True, slots=True)
class ProbeAssessmentResult:
    probe_id: str
    eligibility: ProbeEligibilityState
    disposition: ProbeDisposition


def assess_probe(value: ProbeAssessmentInput) -> ProbeAssessmentResult:
    eligibility = _eligibility(value)
    return ProbeAssessmentResult(value.probe_id, eligibility, _DISPOSITION_ADAPTER.validate_python(_disposition_payload(value, eligibility)))


def primary_harm_witness(result: ProbeAssessmentResult) -> bool:
    return (
        result.eligibility.probe_eligibility_state == "strict_primary_eligible"
        and result.disposition.probe_disposition == "witness"
        and result.disposition.reason_code == "VERIFIER_HARM_WITNESS"
    )


def aggregate_assessments(
    assessments: tuple[ProbeAssessmentResult, ...], kappa: KappaCandidate, suite: SuiteCandidate
) -> ChallengeAssessmentState:
    _validate_kappa(kappa, suite)
    if any(assessment.probe_id not in suite.probe_ids for assessment in assessments):
        raise AssessmentError("UNKNOWN_PROBE_ID")
    evaluable = tuple(
        assessment
        for assessment in assessments
        if assessment.eligibility.probe_eligibility_state == "strict_primary_eligible"
        and assessment.disposition.probe_disposition in {"witness", "evaluated_no_witness"}
    )
    evaluable_probe_ids = {assessment.probe_id for assessment in evaluable}
    coverage = (
        len(evaluable) >= kappa.min_total_evaluable_replicates
        and len(evaluable_probe_ids) >= kappa.min_distinct_evaluable_probes
    )
    if not coverage:
        return _ASSESSMENT_ADAPTER.validate_python(
            {
                "assessment_state": "not_evaluable",
                "coverage_satisfied": False,
                "contradiction_satisfied": False,
            }
        )
    witness_counts = {
        probe_id: sum(
            primary_harm_witness(assessment)
            for assessment in evaluable
            if assessment.probe_id == probe_id
        )
        for probe_id in evaluable_probe_ids
    }
    contradiction = (
        sum(
            count >= kappa.min_witness_replicates_per_probe for count in witness_counts.values()
        )
        >= kappa.min_distinct_witness_probes
    )
    if contradiction:
        return _ASSESSMENT_ADAPTER.validate_python(
            {
                "assessment_state": "contradicted",
                "coverage_satisfied": True,
                "contradiction_satisfied": True,
            }
        )
    return _ASSESSMENT_ADAPTER.validate_python(
        {
            "assessment_state": "not_contradicted",
            "coverage_satisfied": True,
            "contradiction_satisfied": False,
        }
    )


def route_assessment(state: Literal["contradicted", "not_contradicted", "not_evaluable"]) -> ChallengeRoutingDecision:
    for assessment_state, route_target, audit_flag, reason_code in ROUTE_TABLE:
        if state == assessment_state:
            return _ROUTING_ADAPTER.validate_python(
                {
                    "assessment_state": assessment_state,
                    "route_target": route_target,
                    "audit_flag": audit_flag,
                    "routing_reason_code": reason_code,
                }
            )
    raise AssessmentError("UNKNOWN_ASSESSMENT_STATE")


def _eligibility(value: ProbeAssessmentInput) -> ProbeEligibilityState:
    if value.control_provider_status == "failed":
        return _ineligible("CONTROL_PROVIDER_FAILURE")
    if value.control_relation.answer_call_provenance_status != "explicit_matched":
        return _ineligible("ANSWER_CALL_PROVENANCE_UNRESOLVED")
    if (
        value.control_raw_parse_status == "parsed_raw"
        and value.control_verifier_status == "success"
        and value.control_verifier_result is True
    ):
        return _ELIGIBILITY_ADAPTER.validate_python({"probe_eligibility_state": "strict_primary_eligible"})
    if (
        value.control_raw_parse_status == "parse_failed"
        and value.control_canonicalizer_version is not None
        and value.control_canonicalized_parse_status == "parsed_raw"
        and value.control_canonicalized_verifier_status == "success"
        and value.control_canonicalized_verifier_result is True
    ):
        return _ELIGIBILITY_ADAPTER.validate_python(
            {
                "probe_eligibility_state": "canonicalization_sensitivity_eligible",
                "canonicalizer_version": value.control_canonicalizer_version,
            }
        )
    if value.control_raw_parse_status == "parse_failed":
        return _ineligible("CONTROL_PARSE_FAILURE")
    if value.control_verifier_status != "success":
        return _ineligible("CONTROL_VERIFIER_FAILURE")
    return _ineligible("CONTROL_NOT_CLEAN_SOLVABLE")


def _disposition_payload(
    value: ProbeAssessmentInput, eligibility: ProbeEligibilityState
) -> dict[str, str]:
    control_not_clean = value.control_verifier_result is False
    challenge_is_correct = value.challenge_verifier_result is True
    challenge_is_incorrect = value.challenge_verifier_result is False
    both_relations_matched = (
        value.control_relation.answer_call_provenance_status == "explicit_matched"
        and value.challenge_relation.answer_call_provenance_status == "explicit_matched"
    )
    pair_matched = value.pair_identity.paired_execution_identity_status == "matched"
    witness = (
        eligibility.probe_eligibility_state == "strict_primary_eligible"
        and challenge_is_incorrect
        and value.candidate_exposure.candidate_final_context_inclusion
        and both_relations_matched
        and pair_matched
    )
    rules = (
        (value.control_provider_status == "failed", "not_evaluable", "CONTROL_PROVIDER_FAILURE"),
        (value.control_raw_parse_status == "parse_failed", "not_evaluable", "CONTROL_PARSE_FAILURE"),
        (value.control_verifier_status != "success", "not_evaluable", "CONTROL_VERIFIER_FAILURE"),
        (control_not_clean and challenge_is_incorrect, "not_evaluable", "CONTROL_NOT_CLEAN_SOLVABLE"),
        (value.challenge_provider_status == "failed", "not_evaluable", "CHALLENGE_PROVIDER_FAILURE"),
        (value.challenge_raw_parse_status == "parse_failed", "not_evaluable", "CHALLENGE_PARSE_FAILURE"),
        (value.challenge_verifier_status != "success", "not_evaluable", "CHALLENGE_VERIFIER_FAILURE"),
        (
            not value.candidate_exposure.candidate_final_context_inclusion,
            "not_evaluable",
            "CANDIDATE_NOT_EXPOSED",
        ),
        (value.routability.routability == "unsupported", "not_evaluable", "PROBE_MAPPING_UNSUPPORTED"),
        (not both_relations_matched, "not_evaluable", "ANSWER_CALL_PROVENANCE_UNRESOLVED"),
        (not pair_matched, "not_evaluable", "PAIRED_EXECUTION_IDENTITY_MISMATCH"),
        (witness, "witness", "VERIFIER_HARM_WITNESS"),
        (
            eligibility.probe_eligibility_state == "strict_primary_eligible"
            and challenge_is_correct
            and not value.outputs_diverge,
            "evaluated_no_witness",
            "NO_HARM_WITNESS",
        ),
        (
            eligibility.probe_eligibility_state == "strict_primary_eligible"
            and challenge_is_correct
            and value.outputs_diverge,
            "evaluated_no_witness",
            "OUTPUT_DIVERGENCE_WITHOUT_VERIFIED_HARM",
        ),
        (
            control_not_clean and challenge_is_correct,
            "not_evaluable",
            "CONTROL_NOT_CLEAN_SOLVABLE_CHALLENGE_BENEFIT",
        ),
    )
    for applies, disposition, reason_code in rules:
        if applies:
            return {"probe_disposition": disposition, "reason_code": reason_code}
    raise AssessmentError("UNCLASSIFIED_PROBE_DISPOSITION")


def _ineligible(reason_code: str) -> ProbeEligibilityState:
    return _ELIGIBILITY_ADAPTER.validate_python({"probe_eligibility_state": "ineligible", "reason_code": reason_code})


def _validate_kappa(kappa: KappaCandidate, suite: SuiteCandidate) -> None:
    if (
        kappa.min_distinct_witness_probes > kappa.min_distinct_evaluable_probes
        or kappa.min_distinct_evaluable_probes > len(suite.probe_ids)
        or kappa.min_witness_replicates_per_probe > suite.replicates_per_probe
        or kappa.min_total_evaluable_replicates > len(suite.probe_ids) * suite.replicates_per_probe
    ):
        raise AssessmentError("KAPPA_INCOHERENT")
