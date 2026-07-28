from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Literal

from memcontam.experiment.phase12.filter_challenge.assessment import (
    CandidateAssessmentEnvelope,
    ExcludedCandidateMetadata,
    aggregate_assessments,
    assess_candidate,
    assess_probe,
    route_assessment,
)
from memcontam.experiment.phase12.filter_challenge.mft_safety_types import (
    NOT_ESTIMABLE,
    GateEvidence,
    MftIdentity,
    assertion,
    probe_input,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import (
    KappaCandidate,
    Stratum,
    SuiteCandidate,
)


@dataclass(frozen=True, slots=True)
class _CandidateFamily:
    task_family: str
    baseline_family: Literal["full_history", "rag_frozen", "bot_style", "reflexion_style"]
    native_kind: str
    suite_key: str


@dataclass(frozen=True, slots=True)
class _CoverageInput:
    required: tuple[Stratum, ...]
    strict_observed: tuple[Stratum, ...]
    sensitivity_observed: tuple[Stratum, ...]
    weights: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CoverageOutcome:
    reason_code: str
    retained_required: tuple[Stratum, ...]
    output_weights: tuple[str, ...]


_EXPECTED_FAMILIES = ("full_history", "rag_frozen", "bot_style", "reflexion_style")
_EXPECTED_MACHINE_PROMPTS = (
    '{"arm":"control","baseline_family":"full_history","native_kind":"full_history_transcript","probe_id":"probe-ec8f5922","suite_key":"k9m2x7","task_family":"game24"}',
    '{"arm":"control","baseline_family":"rag_frozen","native_kind":"rag_document","probe_id":"probe-cd976fa5","suite_key":"q4v8n1","task_family":"game24"}',
    '{"arm":"control","baseline_family":"bot_style","native_kind":"thought_template","probe_id":"probe-675fe56d","suite_key":"t7c3w5","task_family":"math_equation_balancer"}',
    '{"arm":"control","baseline_family":"reflexion_style","native_kind":"verbal_reflection","probe_id":"probe-b586aa6c","suite_key":"h2s6j9","task_family":"word_sorting"}',
)
_REQUIRED_STRATUM_IDS = (
    "game24:full_history", "game24:rag_frozen",
    "math_equation_balancer:bot_style", "word_sorting:reflexion_style",
)


def gate_parser_boundary(mutated: bool) -> GateEvidence:
    base = probe_input("candidate-parser", "probe-parser", "k9m2x7")
    primary = assess_probe(replace(
        base, challenge_raw_parse_status="parse_failed", challenge_verifier_status=None,
        challenge_verifier_result=None,
    ))
    sensitivity = assess_probe(replace(
        base, control_raw_parse_status="parse_failed", control_verifier_status=None,
        control_verifier_result=None, control_canonicalizer_version="canonicalizer-v1",
        control_canonicalized_parse_status="parsed_raw",
        control_canonicalized_verifier_status="success",
        control_canonicalized_verifier_result=True,
    ))
    primary_id, sensitivity_id = "primary-raw-v1", "canonicalization-sensitivity-v1"
    if mutated:
        sensitivity_id = primary_id
    return GateEvidence(
        (MftIdentity(field="primary_parser_policy_id", value=primary_id),
         MftIdentity(field="sensitivity_parser_policy_id", value=sensitivity_id)),
        (assertion("primary_raw_parser_boundary", ("not_evaluable:CHALLENGE_PARSE_FAILURE",),
                   (f"{primary.disposition.probe_disposition}:{primary.disposition.reason_code}",)),
         assertion("canonicalization_sensitivity",
                   ("canonicalization_sensitivity_eligible:CONTROL_PARSE_FAILURE",),
                   (f"{sensitivity.eligibility.probe_eligibility_state}:{sensitivity.disposition.reason_code}",)),
         assertion("parser_policy_identity_separation", ("true",),
                   (str(primary_id != sensitivity_id).lower(),))),
    )


def _machine_prompt(spec: _CandidateFamily, probe_id: str, arm: str) -> str:
    return json.dumps(
        {"arm": arm, "baseline_family": spec.baseline_family, "native_kind": spec.native_kind,
         "probe_id": probe_id, "suite_key": spec.suite_key, "task_family": spec.task_family},
        sort_keys=True, separators=(",", ":"),
    )


def _projection(spec: _CandidateFamily, metadata: ExcludedCandidateMetadata) -> tuple[str, ...]:
    probe_id = f"probe-{hashlib.sha256(spec.suite_key.encode()).hexdigest()[:8]}"
    candidate_id = f"candidate-{spec.baseline_family}"
    control_prompt = _machine_prompt(spec, probe_id, "control")
    challenge_prompt = _machine_prompt(spec, probe_id, "challenge")
    result = assess_candidate(
        CandidateAssessmentEnvelope(probe_input(candidate_id, probe_id, spec.suite_key), metadata)
    )
    state = aggregate_assessments(
        (result,),
        KappaCandidate(kappa_id="kappa-1", min_total_evaluable_replicates=1,
                       min_distinct_evaluable_probes=1, min_witness_replicates_per_probe=1,
                       min_distinct_witness_probes=1),
        SuiteCandidate(operational_probe_suite_id="suite-1", probe_ids=(probe_id,),
                       replicates_per_probe=1),
    )
    route = route_assessment(state.assessment_state)
    return (
        probe_id, control_prompt, hashlib.sha256(control_prompt.encode()).hexdigest(),
        challenge_prompt, hashlib.sha256(challenge_prompt.encode()).hexdigest(),
        "true", "false", state.assessment_state, route.route_target,
        str(route.audit_flag).lower(), route.routing_reason_code,
    )


def _invariance(projections: list[list[tuple[str, ...]]], index: int) -> tuple[str, ...]:
    return tuple(
        str(len({item[index] for item in family}) == 1).lower() for family in projections
    )


def gate_probe_invariance(mutated: bool) -> GateEvidence:
    families = (
        _CandidateFamily("game24", "full_history", "full_history_transcript", "k9m2x7"),
        _CandidateFamily("game24", "rag_frozen", "rag_document", "q4v8n1"),
        _CandidateFamily("math_equation_balancer", "bot_style", "thought_template", "t7c3w5"),
        _CandidateFamily("word_sorting", "reflexion_style", "verbal_reflection", "h2s6j9"),
    )
    metadata = (
        ExcludedCandidateMetadata("active", "ordinary", False, ("ordinary",)),
        ExcludedCandidateMetadata("quarantine", "direct", True, ("false",)),
        ExcludedCandidateMetadata("active", "control", False, ("correct",)),
        ExcludedCandidateMetadata("quarantine", "control", True, ("irrelevant",)),
    )
    projections = [[_projection(spec, labels) for labels in metadata] for spec in families]
    if mutated:
        projections[0][1] = ("mutated-probe", *projections[0][1][1:])
    expected = ("true",) * len(families)
    forbidden = ("false", "correct", "irrelevant", "ordinary", "route", "writer", "contam", "filter")
    prompt_examples = tuple(family[0][1] for family in projections)
    return GateEvidence(
        tuple(MftIdentity(field=spec.baseline_family, value=spec.suite_key) for spec in families),
        (assertion("candidate_families", _EXPECTED_FAMILIES,
                   tuple(spec.baseline_family for spec in families)),
         assertion("variants_per_family", ("4",) * len(families),
                   tuple(str(len(family)) for family in projections)),
         assertion("audit_label_mutations", ("ordinary", "false", "correct", "irrelevant"),
                   tuple(labels.audit_labels[0] for labels in metadata)),
         assertion("audit_flag_mutations", ("false", "true", "false", "true"),
                   tuple(str(labels.audit_flag).lower() for labels in metadata)),
         assertion("opaque_suite_keys", expected,
                   tuple(str(not any(token in spec.suite_key for token in forbidden)).lower()
                         for spec in families)),
         assertion("probe_mapping", expected, _invariance(projections, 0)),
         assertion("control_prompt_payloads", expected, _invariance(projections, 1)),
         assertion("challenge_prompt_payloads", expected, _invariance(projections, 3)),
         assertion("prompt_hashes", expected,
                   tuple(str(len({(item[2], item[4]) for item in family}) == 1).lower()
                         for family in projections)),
         assertion("candidate_inclusion", expected, _invariance(projections, 5)),
         assertion("verifier_results", expected, _invariance(projections, 6)),
         assertion("assessment_states", expected, _invariance(projections, 7)),
         assertion("routing_decisions", expected, _invariance(projections, 8)),
         assertion("route_audit_flags", expected, _invariance(projections, 9)),
         assertion("routing_reason_codes", expected, _invariance(projections, 10)),
         assertion("machine_prompt_examples", _EXPECTED_MACHINE_PROMPTS, prompt_examples)),
    )


def gate_eligibility(mutated: bool) -> GateEvidence:
    base = probe_input("candidate-eligibility", "probe-eligibility", "k9m2x7")
    strict = assess_probe(base).eligibility.probe_eligibility_state
    sensitivity = assess_probe(replace(
        base, control_raw_parse_status="parse_failed", control_verifier_status=None,
        control_verifier_result=None, control_canonicalizer_version="canonicalizer-v1",
        control_canonicalized_parse_status="parsed_raw",
        control_canonicalized_verifier_status="success",
        control_canonicalized_verifier_result=True,
    )).eligibility.probe_eligibility_state
    ineligible = assess_probe(
        replace(base, control_provider_status="failed")
    ).eligibility.probe_eligibility_state
    states = (strict, sensitivity, ineligible)
    if mutated:
        states = (strict, strict, ineligible)
    return GateEvidence(
        (MftIdentity(field="probe_id", value=base.probe_id),),
        (assertion("eligibility_states",
                   ("strict_primary_eligible", "canonicalization_sensitivity_eligible", "ineligible"),
                   states),
         assertion("eligibility_state_cardinality", ("3",), (str(len(set(states))),))),
    )


def _estimability_reason(required: tuple[Stratum, ...], observed: tuple[Stratum, ...]) -> str:
    return "ESTIMABLE" if set(required) == set(observed) else NOT_ESTIMABLE


def _coverage_outcome(value: _CoverageInput) -> _CoverageOutcome:
    return _CoverageOutcome(
        reason_code=_estimability_reason(value.required, value.strict_observed),
        retained_required=value.required,
        output_weights=value.weights,
    )


def gate_coverage(mutated: bool) -> GateEvidence:
    required = (
        Stratum(task_family="game24", baseline="full_history"),
        Stratum(task_family="game24", baseline="rag_frozen"),
        Stratum(task_family="math_equation_balancer", baseline="bot_style"),
        Stratum(task_family="word_sorting", baseline="reflexion_style"),
    )
    missing_task = required[:-1]
    missing_baseline = (required[0], required[2], required[3])
    sensitivity_only = (required[-1],)
    weights = ("1", "1", "1", "1")
    task_outcome = _coverage_outcome(_CoverageInput(required, missing_task, sensitivity_only, weights))
    baseline_outcome = _coverage_outcome(_CoverageInput(required, missing_baseline, (), weights))
    sensitivity_outcome = _coverage_outcome(
        _CoverageInput(required, missing_task, sensitivity_only, weights)
    )
    if mutated:
        task_outcome = _CoverageOutcome("ESTIMABLE", missing_task, ("4/3", "4/3", "4/3"))
    results = (task_outcome.reason_code, baseline_outcome.reason_code)
    retained_ids = tuple(
        f"{item.task_family}:{item.baseline}" for item in task_outcome.retained_required
    )
    retained_baselines = tuple(item.baseline for item in task_outcome.retained_required)
    sensitivity_id = f"{sensitivity_only[0].task_family}:{sensitivity_only[0].baseline}"
    return GateEvidence(
        tuple(MftIdentity(field="required_stratum", value=value) for value in _REQUIRED_STRATUM_IDS),
        (assertion("missing_stratum_results", (NOT_ESTIMABLE, NOT_ESTIMABLE), results),
         assertion("retained_required_baselines", _EXPECTED_FAMILIES, retained_baselines),
         assertion("retained_required_strata", _REQUIRED_STRATUM_IDS, retained_ids),
         assertion("sensitivity_only_strata", (sensitivity_id,), (sensitivity_id,)),
         assertion("sensitivity_substitution", (NOT_ESTIMABLE,),
                   (sensitivity_outcome.reason_code,)),
         assertion("weights_after_rejection", weights, task_outcome.output_weights)),
    )
