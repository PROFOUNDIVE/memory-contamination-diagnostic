from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeRoutability,
    PairedExecutionIdentity,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import KappaCandidate, SuiteCandidate

from memcontam.experiment.phase12.filter_challenge.assessment import (
    ASSESSMENT_STATES,
    DISPOSITION_REASON_CODES,
    DISPOSITION_STATES,
    ELIGIBILITY_STATES,
    ROUTE_TABLE,
    AssessmentError,
    ProbeAssessmentInput,
    aggregate_assessments,
    assess_probe,
    primary_harm_witness,
    route_assessment,
)


def _relation(status: str = "explicit_matched") -> AnswerCallRelation:
    payload: dict[str, str] = {
        "answer_call_provenance_status": status,
        "answer_call_id": f"{status}-answer",
    }
    if status == "explicit_matched":
        payload |= {
            "parsed_response_source_call_id": payload["answer_call_id"],
            "parser_result_id": "parser-1",
            "verifier_result_id": "verifier-1",
        }
    return TypeAdapter(AnswerCallRelation).validate_python(payload)


def _pair(status: str = "matched") -> PairedExecutionIdentity:
    payload = {"paired_execution_identity_status": status, "pair_id": "pair-1"}
    if status == "mismatched":
        payload["reason_code"] = "PAIRED_EXECUTION_IDENTITY_MISMATCH"
    return TypeAdapter(PairedExecutionIdentity).validate_python(payload)


def _routability(status: str = "challenge_routable_v1") -> ChallengeRoutability:
    payload = {"routability": status}
    if status == "challenge_routable_v1":
        payload["challenge_suite_key"] = "suite-1"
    else:
        payload["reason_code"] = "PROBE_MAPPING_UNSUPPORTED"
    return TypeAdapter(ChallengeRoutability).validate_python(payload)


def _exposure(included: bool = True) -> CandidateExposureRecord:
    return CandidateExposureRecord(
        candidate_entry_id="candidate-1",
        candidate_final_context_inclusion=included,
        candidate_final_context_source_ids=("candidate-1",) if included else (),
    )


def _input(**changes: object) -> ProbeAssessmentInput:
    values = {
        "probe_id": "probe-1",
        "control_provider_status": "success",
        "control_raw_parse_status": "parsed_raw",
        "control_verifier_status": "success",
        "control_verifier_result": True,
        "control_relation": _relation(),
        "control_canonicalizer_version": None,
        "control_canonicalized_parse_status": None,
        "control_canonicalized_verifier_status": None,
        "control_canonicalized_verifier_result": None,
        "challenge_provider_status": "success",
        "challenge_raw_parse_status": "parsed_raw",
        "challenge_verifier_status": "success",
        "challenge_verifier_result": False,
        "challenge_relation": _relation(),
        "candidate_exposure": _exposure(),
        "routability": _routability(),
        "pair_identity": _pair(),
        "outputs_diverge": False,
    }
    return ProbeAssessmentInput(**(values | changes))


def _kappa(**changes: int) -> KappaCandidate:
    values = {
        "kappa_id": "kappa-1",
        "min_total_evaluable_replicates": 2,
        "min_distinct_evaluable_probes": 2,
        "min_witness_replicates_per_probe": 1,
        "min_distinct_witness_probes": 1,
    }
    return KappaCandidate(**(values | changes))


def _suite() -> SuiteCandidate:
    return SuiteCandidate(
        operational_probe_suite_id="suite-1",
        probe_ids=("probe-1", "probe-2"),
        replicates_per_probe=2,
    )


def test_locked_domain_tuples_and_route_table_are_literal() -> None:
    # Given: the Task 2 closed unions.
    # When: the assessment-facing constants are inspected.
    # Then: their order and fail-open route table remain exact.
    assert ELIGIBILITY_STATES == (
        "strict_primary_eligible",
        "canonicalization_sensitivity_eligible",
        "ineligible",
    )
    assert DISPOSITION_STATES == ("witness", "evaluated_no_witness", "not_evaluable")
    assert ASSESSMENT_STATES == ("contradicted", "not_contradicted", "not_evaluable")
    assert DISPOSITION_REASON_CODES == (
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
    assert ROUTE_TABLE == (
        ("contradicted", "quarantine", False, "CONTRADICTED"),
        ("not_contradicted", "active", False, "NOT_CONTRADICTED"),
        ("not_evaluable", "active", True, "FAIL_OPEN_NOT_EVALUABLE"),
    )


def test_control_eligibility_is_frozen_before_pair_validation() -> None:
    # Given: an otherwise strict control with a mismatched pair identity.
    # When: the pair is assessed.
    # Then: frozen control eligibility remains strict while the pair is not evaluable.
    result = assess_probe(_input(pair_identity=_pair("mismatched")))
    assert result.eligibility.probe_eligibility_state == "strict_primary_eligible"
    assert (result.disposition.probe_disposition, result.disposition.reason_code) == (
        "not_evaluable",
        "PAIRED_EXECUTION_IDENTITY_MISMATCH",
    )


def test_canonicalization_is_a_separate_control_sensitivity() -> None:
    # Given: a raw control parse failure repaired only by the registered canonicalizer.
    # When: control eligibility is assigned.
    # Then: the record is sensitivity-only and primary disposition preserves raw parse failure.
    result = assess_probe(
        _input(
            control_raw_parse_status="parse_failed",
            control_verifier_status=None,
            control_verifier_result=None,
            control_canonicalizer_version="canonicalizer-v1",
            control_canonicalized_parse_status="parsed_raw",
            control_canonicalized_verifier_status="success",
            control_canonicalized_verifier_result=True,
        )
    )
    assert result.eligibility.probe_eligibility_state == "canonicalization_sensitivity_eligible"
    assert (result.disposition.probe_disposition, result.disposition.reason_code) == (
        "not_evaluable",
        "CONTROL_PARSE_FAILURE",
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"control_provider_status": "failed"}, "CONTROL_PROVIDER_FAILURE"),
        ({"control_raw_parse_status": "parse_failed"}, "CONTROL_PARSE_FAILURE"),
        ({"control_verifier_status": "failed"}, "CONTROL_VERIFIER_FAILURE"),
        ({"control_verifier_result": False}, "CONTROL_NOT_CLEAN_SOLVABLE"),
        ({"challenge_provider_status": "failed"}, "CHALLENGE_PROVIDER_FAILURE"),
        ({"challenge_raw_parse_status": "parse_failed"}, "CHALLENGE_PARSE_FAILURE"),
        ({"challenge_verifier_status": "failed"}, "CHALLENGE_VERIFIER_FAILURE"),
        ({"candidate_exposure": _exposure(False)}, "CANDIDATE_NOT_EXPOSED"),
        ({"routability": _routability("unsupported")}, "PROBE_MAPPING_UNSUPPORTED"),
        ({"challenge_relation": _relation("missing")}, "ANSWER_CALL_PROVENANCE_UNRESOLVED"),
        ({"pair_identity": _pair("mismatched")}, "PAIRED_EXECUTION_IDENTITY_MISMATCH"),
        ({}, "VERIFIER_HARM_WITNESS"),
        ({"challenge_verifier_result": True}, "NO_HARM_WITNESS"),
        (
            {"challenge_verifier_result": True, "outputs_diverge": True},
            "OUTPUT_DIVERGENCE_WITHOUT_VERIFIED_HARM",
        ),
        (
            {"control_verifier_result": False, "challenge_verifier_result": True},
            "CONTROL_NOT_CLEAN_SOLVABLE_CHALLENGE_BENEFIT",
        ),
    ),
)
def test_disposition_uses_the_exact_first_matching_reason(
    changes: dict[str, object], expected: str
) -> None:
    # Given: one ordered disposition condition.
    # When: the pair is assessed.
    # Then: the exact first applicable reason is retained.
    result = assess_probe(_input(**changes))
    assert result.disposition.reason_code == expected


@pytest.mark.parametrize(
    "changes",
    (
        {"control_verifier_result": False},
        {"control_verifier_result": False, "control_raw_parse_status": "parse_failed"},
        {"challenge_verifier_result": True},
        {"candidate_exposure": _exposure(False)},
        {"control_relation": _relation("missing")},
        {"pair_identity": _pair("mismatched")},
    ),
)
def test_primary_witness_requires_each_of_its_six_conjuncts(changes: dict[str, object]) -> None:
    # Given: one independently false primary-witness conjunct.
    # When: the strict witness predicate is evaluated.
    # Then: no mutation can be a primary harm witness.
    result = assess_probe(_input(**changes))
    assert primary_harm_witness(result) is False


def test_aggregate_uses_coverage_before_contradiction() -> None:
    # Given: one witness without the two-probe registered coverage requirement.
    # When: strict primary assessments are aggregated.
    # Then: the result is not evaluable rather than contradicted.
    witness = assess_probe(_input())
    assert aggregate_assessments((witness,), _kappa(), _suite()).assessment_state == "not_evaluable"


def test_aggregate_has_all_three_states_without_sensitivity_pooling() -> None:
    # Given: two strict records covering two probes plus a canonicalization-sensitivity witness.
    witness = assess_probe(_input())
    no_witness = assess_probe(_input(probe_id="probe-2", challenge_verifier_result=True))
    sensitivity = assess_probe(
        _input(
            probe_id="probe-2",
            control_raw_parse_status="parse_failed",
            control_verifier_status=None,
            control_verifier_result=None,
            control_canonicalizer_version="canonicalizer-v1",
            control_canonicalized_parse_status="parsed_raw",
            control_canonicalized_verifier_status="success",
            control_canonicalized_verifier_result=True,
        )
    )

    # When: each registered κ rule is applied without reweighting the survivors.
    contradicted = aggregate_assessments((witness, replace(witness, probe_id="probe-2")), _kappa(), _suite())
    not_contradicted = aggregate_assessments(
        (replace(no_witness, probe_id="probe-1"), no_witness), _kappa(), _suite()
    )
    not_evaluable = aggregate_assessments((witness, sensitivity), _kappa(), _suite())

    # Then: the states are distinct and sensitivity never fills primary coverage.
    assert contradicted.assessment_state == "contradicted"
    assert not_contradicted.assessment_state == "not_contradicted"
    assert not_evaluable.assessment_state == "not_evaluable"


def test_aggregate_rejects_incoherent_kappa_without_a_default() -> None:
    # Given: a κ whose witness-probe minimum exceeds evaluable-probe coverage.
    # When: aggregation is requested.
    # Then: no majority or implicit fallback κ is used.
    with pytest.raises(AssessmentError, match="KAPPA_INCOHERENT"):
        aggregate_assessments(
            (assess_probe(_input()),),
            _kappa(min_distinct_evaluable_probes=1, min_distinct_witness_probes=2),
            _suite(),
        )


@pytest.mark.parametrize(
    ("state", "expected"),
    (
        ("contradicted", ("quarantine", False, "CONTRADICTED")),
        ("not_contradicted", ("active", False, "NOT_CONTRADICTED")),
        ("not_evaluable", ("active", True, "FAIL_OPEN_NOT_EVALUABLE")),
    ),
)
def test_routing_is_exact_and_audit_free(state: str, expected: tuple[str, bool, str]) -> None:
    # Given: one candidate assessment state.
    # When: it is routed.
    # Then: only the locked fail-open table determines the route and audit flag.
    decision = route_assessment(state)
    assert (decision.route_target, decision.audit_flag, decision.routing_reason_code) == expected
