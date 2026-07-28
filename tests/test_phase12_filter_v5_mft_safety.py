from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import memcontam.experiment.phase12.filter_challenge.mft_safety_assessment as safety_assessment
from memcontam.experiment.phase12.filter_challenge.assessment import ProbeAssessmentInput
from memcontam.experiment.phase12.filter_challenge.contracts import CandidateExposureRecord
from memcontam.experiment.phase12.filter_challenge.mft_safety import (
    MFT_SAFETY_FAILURE_REASONS,
    MFT_SAFETY_IDS,
    MftSafetyError,
    build_mft_safety_report,
    write_mft_safety_report,
)
from memcontam.experiment.phase12.filter_challenge.mft_safety_types import (
    MftAssertion,
    MftSafetyCase,
)
from memcontam.experiment.phase12.filter_challenge.mft_safety_executor import gate_shadow_share


EXPECTED_IDS = (
    "MFT-FV5-09-CONTAM-SHADOW-SHARE",
    "MFT-FV5-10-PARSER-BOUNDARY",
    "MFT-FV5-11-CONTROL-CACHE",
    "MFT-FV5-12-PROBE-KEY-INVARIANCE",
    "MFT-FV5-13-ANSWER-CALL-PROVENANCE",
    "MFT-FV5-14-ACTIVATION-DOMAIN",
    "MFT-FV5-15-ELIGIBILITY-STATES",
    "MFT-FV5-16-COVERAGE-NOT-ESTIMABLE",
)
EXPECTED_FAILURE_REASONS = (
    "CONTAM_SHADOW_SHARE_IMPLEMENTATION_FAILURE",
    "PARSER_BOUNDARY_IMPLEMENTATION_FAILURE",
    "CONTROL_CACHE_IMPLEMENTATION_FAILURE",
    "PROBE_KEY_INVARIANCE_IMPLEMENTATION_FAILURE",
    "ANSWER_CALL_PROVENANCE_IMPLEMENTATION_FAILURE",
    "ACTIVATION_DOMAIN_IMPLEMENTATION_FAILURE",
    "ELIGIBILITY_STATES_IMPLEMENTATION_FAILURE",
    "COVERAGE_ESTIMABILITY_IMPLEMENTATION_FAILURE",
)
TASK_12_IDS = (
    "MFT-FV5-01-PAIR-MATCH",
    "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE",
    "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE",
    "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT",
    "MFT-FV5-08-NO-WRITEBACK",
)
EXACT_16_IDS = (
    "MFT-FV5-01-PAIR-MATCH",
    "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE",
    "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE",
    "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT",
    "MFT-FV5-08-NO-WRITEBACK",
    "MFT-FV5-09-CONTAM-SHADOW-SHARE",
    "MFT-FV5-10-PARSER-BOUNDARY",
    "MFT-FV5-11-CONTROL-CACHE",
    "MFT-FV5-12-PROBE-KEY-INVARIANCE",
    "MFT-FV5-13-ANSWER-CALL-PROVENANCE",
    "MFT-FV5-14-ACTIVATION-DOMAIN",
    "MFT-FV5-15-ELIGIBILITY-STATES",
    "MFT-FV5-16-COVERAGE-NOT-ESTIMABLE",
)


def _assertion(case: MftSafetyCase, field: str) -> MftAssertion:
    return next(assertion for assertion in case.assertions if assertion.field == field)


def test_registry_and_report_lock_exact_order_and_single_execution() -> None:
    # Given: the independent Task 12 authority tuple and the Task 13 runner.
    report = build_mft_safety_report()

    # When: the registry is executed without mutations.
    # Then: every literal ID runs exactly once and the future merged tuple is exact.
    assert MFT_SAFETY_IDS == EXPECTED_IDS
    assert MFT_SAFETY_FAILURE_REASONS == EXPECTED_FAILURE_REASONS
    assert TASK_12_IDS + MFT_SAFETY_IDS == EXACT_16_IDS
    assert report.test_ids == EXPECTED_IDS
    assert tuple(case.test_id for case in report.cases) == EXPECTED_IDS
    assert tuple((item.test_id, item.count) for item in report.execution_counts) == tuple(
        (test_id, 1) for test_id in EXPECTED_IDS
    )
    assert report.all_passed and report.provider_calls_issued == 0
    assert report.decision_inputs == ()
    assert all(case.status == "pass" and case.reason_code is None for case in report.cases)


def test_safety_gates_exercise_exact_machine_laws() -> None:
    # Given: all eight deterministic safety cases.
    cases = {case.test_id: case for case in build_mft_safety_report().cases}

    # When: their structured observations are inspected.
    # Then: shadowing, parser, cache, provenance, activation, eligibility, and coverage stay exact.
    assert _assertion(cases[EXPECTED_IDS[0]], "routing_consumption").actual == (
        "shadow:none", "apply:quarantine",
    )
    assert _assertion(cases[EXPECTED_IDS[1]], "primary_raw_parser_boundary").actual == (
        "not_evaluable:CHALLENGE_PARSE_FAILURE",
    )
    assert _assertion(cases[EXPECTED_IDS[1]], "canonicalization_sensitivity").actual == (
        "canonicalization_sensitivity_eligible:CONTROL_PARSE_FAILURE",
    )
    cache = cases[EXPECTED_IDS[2]]
    assert _assertion(cache, "equal_identity_key_equality").actual == ("true",)
    assert _assertion(cache, "all_field_mutation_sensitivity").actual == ("22",)
    provenance = cases[EXPECTED_IDS[4]]
    assert _assertion(provenance, "relation_statuses").actual == (
        "explicit_matched", "missing", "ambiguous", "historically_reconstructed", "mismatched",
    )
    assert _assertion(provenance, "primary_admissibility").actual == (
        "true", "false", "false", "false", "false",
    )
    assert _assertion(provenance, "known_valid_batch_health").actual == ("false",)
    assert _assertion(cases[EXPECTED_IDS[5]], "activation_paths").actual == (
        "grandfathered", "assess", "assess", "not_evaluable",
    )
    assert _assertion(cases[EXPECTED_IDS[6]], "eligibility_states").actual == (
        "strict_primary_eligible", "canonicalization_sensitivity_eligible", "ineligible",
    )
    coverage = cases[EXPECTED_IDS[7]]
    assert _assertion(coverage, "missing_stratum_results").actual == (
        "FILTER_V5_PILOT_B_NOT_ESTIMABLE", "FILTER_V5_PILOT_B_NOT_ESTIMABLE",
    )
    assert _assertion(coverage, "sensitivity_substitution").actual == (
        "FILTER_V5_PILOT_B_NOT_ESTIMABLE",
    )
    assert _assertion(coverage, "retained_required_baselines").actual == (
        "full_history", "rag_frozen", "bot_style", "reflexion_style",
    )
    assert _assertion(coverage, "retained_required_strata").actual == (
        "game24:full_history", "game24:rag_frozen",
        "math_equation_balancer:bot_style", "word_sorting:reflexion_style",
    )
    assert _assertion(coverage, "weights_after_rejection").actual == ("1", "1", "1", "1")


def test_shadow_share_observes_each_consumed_key_and_rejects_filter_substitution() -> None:
    # Given: healthy and distinct-valid-Filter-key executions of the shadow/share gate.
    healthy = gate_shadow_share(False)
    attacked = gate_shadow_share(True)

    # When: the independently consumed key hashes are inspected.
    healthy_keys = next(item for item in healthy.assertions if item.field == "shared_assessment_keys")
    attacked_keys = next(item for item in attacked.assertions if item.field == "shared_assessment_keys")

    # Then: healthy Contam/Filter keys match and substituting Filter's key fails the gate.
    assert healthy_keys.actual[0] == healthy_keys.actual[1]
    assert attacked_keys.actual[0] != attacked_keys.actual[1]
    assert not attacked_keys.matched
    assert "assessment_identity" not in {item.field for item in healthy.assertions}


def test_probe_invariance_fails_when_actual_candidate_inclusion_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: production probe inputs whose candidate is absent from final context.
    original = safety_assessment.probe_input

    def without_inclusion(
        candidate_id: str, probe_id: str, suite_key: str
    ) -> ProbeAssessmentInput:
        value = original(candidate_id, probe_id, suite_key)
        return replace(
            value,
            candidate_exposure=CandidateExposureRecord(
                candidate_entry_id=candidate_id,
                candidate_final_context_inclusion=False,
                candidate_final_context_source_ids=(),
            ),
        )

    monkeypatch.setattr(safety_assessment, "probe_input", without_inclusion)

    # When: MFT-12 derives its observations from those production inputs.
    evidence = safety_assessment.gate_probe_invariance(False)

    # Then: false inclusion is observed and fails the gate.
    inclusion = next(item for item in evidence.assertions if item.field == "candidate_inclusion")
    assert inclusion.actual == ("false", "false", "false", "false")
    assert not inclusion.matched


def test_probe_invariance_fails_when_actual_challenge_verifier_is_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: production probe inputs whose challenge verifier result is correct.
    original = safety_assessment.probe_input

    def with_correct_challenge(
        candidate_id: str, probe_id: str, suite_key: str
    ) -> ProbeAssessmentInput:
        return replace(
            original(candidate_id, probe_id, suite_key),
            challenge_verifier_result=True,
        )

    monkeypatch.setattr(safety_assessment, "probe_input", with_correct_challenge)

    # When: MFT-12 derives its observations from those production inputs and results.
    evidence = safety_assessment.gate_probe_invariance(False)

    # Then: the correct verifier result is observed and fails the false-harm fixture gate.
    verifier = next(item for item in evidence.assertions if item.field == "verifier_results")
    assert verifier.actual == ("true", "true", "true", "true")
    assert not verifier.matched


def test_probe_mapping_is_role_route_and_audit_blind_across_families() -> None:
    # Given: four native candidate families and four excluded-label variants per family.
    case = build_mft_safety_report().cases[3]

    # When: only role, route, and audit labels differ.
    # Then: every machine-consumed projection stays invariant without prose decisions.
    assert _assertion(case, "candidate_families").actual == (
        "full_history", "rag_frozen", "bot_style", "reflexion_style",
    )
    assert _assertion(case, "variants_per_family").actual == ("4", "4", "4", "4")
    assert _assertion(case, "audit_label_mutations").actual == (
        "ordinary", "false", "correct", "irrelevant",
    )
    assert _assertion(case, "audit_flag_mutations").actual == (
        "false", "true", "false", "true",
    )
    for field in (
        "opaque_suite_keys", "probe_mapping", "control_prompt_payloads",
        "challenge_prompt_payloads", "prompt_hashes", "candidate_inclusion_invariance",
        "verifier_result_invariance", "assessment_states", "routing_decisions",
        "route_audit_flags", "routing_reason_codes",
    ):
        assert _assertion(case, field).actual == ("true", "true", "true", "true")
    assert _assertion(case, "candidate_inclusion").actual == ("true",) * 4
    assert _assertion(case, "verifier_results").actual == ("false",) * 4
    prompt_examples = _assertion(case, "machine_prompt_examples").actual
    assert prompt_examples == (
        '{"arm":"control","baseline_family":"full_history","native_kind":"full_history_transcript","probe_id":"probe-ec8f5922","suite_key":"k9m2x7","task_family":"game24"}',
        '{"arm":"control","baseline_family":"rag_frozen","native_kind":"rag_document","probe_id":"probe-cd976fa5","suite_key":"q4v8n1","task_family":"game24"}',
        '{"arm":"control","baseline_family":"bot_style","native_kind":"thought_template","probe_id":"probe-675fe56d","suite_key":"t7c3w5","task_family":"math_equation_balancer"}',
        '{"arm":"control","baseline_family":"reflexion_style","native_kind":"verbal_reflection","probe_id":"probe-b586aa6c","suite_key":"h2s6j9","task_family":"word_sorting"}',
    )
    for payload in prompt_examples:
        assert isinstance(json.loads(payload), dict)


@pytest.mark.parametrize(
    ("test_id", "reason_code"), tuple(zip(EXPECTED_IDS, EXPECTED_FAILURE_REASONS, strict=True))
)
def test_negative_mutations_return_exact_implementation_failure(
    test_id: str, reason_code: str
) -> None:
    # Given: one gate-specific deliberate implementation mutation.
    report = build_mft_safety_report((test_id,))

    # When: all eight IDs still execute once.
    failed = tuple(case for case in report.cases if case.status == "implementation_failure")

    # Then: only the mutated gate fails with its exact stable reason.
    assert tuple((case.test_id, case.reason_code) for case in failed) == ((test_id, reason_code),)
    assert not report.all_passed
    assert all(item.count == 1 for item in report.execution_counts)


@pytest.mark.parametrize("mutations", (("UNKNOWN",), (EXPECTED_IDS[0], EXPECTED_IDS[0])))
def test_mutation_boundary_rejects_unknown_or_duplicate_ids(mutations: tuple[str, ...]) -> None:
    # Given: a mutation selector that cannot name exactly one registered gate.
    # When / Then: boundary parsing rejects it with one typed machine reason.
    with pytest.raises(MftSafetyError, match="INVALID_MFT_SAFETY_MUTATION"):
        build_mft_safety_report(mutations)


def test_report_serializes_as_canonical_machine_json_with_bound_evidence_hashes(
    tmp_path: Path,
) -> None:
    # Given: an output path for the Task 13 machine report.
    output = tmp_path / "task-13-mft-safety.json"

    # When: the report is serialized.
    report = write_mft_safety_report(output)
    parsed = json.loads(output.read_text(encoding="utf-8"))

    # Then: JSON round-trips and each case hash binds all non-hash evidence fields.
    assert parsed == report.model_dump(mode="json")
    assert output.read_text(encoding="utf-8").endswith("\n")
    for case in parsed["cases"]:
        evidence_hash = case.pop("evidence_hash")
        canonical = json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert evidence_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert parsed["decision_inputs"] == []
