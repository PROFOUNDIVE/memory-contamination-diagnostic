from __future__ import annotations

from dataclasses import asdict

from memcontam.experiment.phase12.filter_challenge.assessment import (
    ExcludedCandidateMetadata,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import (
    GateEvaluation,
    JsonValue,
    MftMachineObservation,
    MftStateContext,
    canonical_hash,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_scripted import (
    Script,
    answer,
    build_scripts,
    combine,
    summary,
)


def exposure_gate(context: MftStateContext, exposed: bool) -> GateEvaluation:
    count = context.kappa_candidate.min_total_evaluable_replicates
    actual = summary(context, build_scripts(context, challenge=answer(correct=False), exposed=exposed))
    expected = MftMachineObservation(
        candidate_final_context_inclusions=(False,) * count,
        assessment_states=("not_evaluable",),
        route_targets=("active",),
        audit_flags=(True,),
        probe_reason_codes=("CANDIDATE_NOT_EXPOSED",) * count,
        routing_reason_codes=("FAIL_OPEN_NOT_EVALUABLE",),
    )
    return GateEvaluation(
        ("unexposed-incorrect-challenge",), expected, actual, "CANDIDATE_EXPOSURE_ASSERTION_FAILED"
    )


def tristate_gate(context: MftStateContext, mutate_route: bool) -> GateEvaluation:
    summaries = (
        summary(context, build_scripts(context, challenge=answer(correct=False))),
        summary(context, build_scripts(context, challenge=answer(correct=True))),
        summary(context, build_scripts(context, challenge=answer(correct=False), exposed=False)),
    )
    actual = combine(summaries)
    if mutate_route:
        actual = actual.model_copy(update={"route_targets": ("active", *actual.route_targets[1:])})
    expected = MftMachineObservation(
        assessment_states=("contradicted", "not_contradicted", "not_evaluable"),
        route_targets=("quarantine", "active", "active"),
        audit_flags=(False, False, True),
        routing_reason_codes=("CONTRADICTED", "NOT_CONTRADICTED", "FAIL_OPEN_NOT_EVALUABLE"),
    )
    return GateEvaluation(
        ("contradicted", "not-contradicted", "not-evaluable"),
        expected,
        actual,
        "ROUTING_RECONCILIATION_FAILED",
    )


def route_invariance_gate(context: MftStateContext) -> GateEvaluation:
    metadata = (
        ExcludedCandidateMetadata("quarantine", "direct-route-a", False, ("audit-a",)),
        ExcludedCandidateMetadata("active", "direct-route-b", True, ("audit-b",)),
    )
    scripts = build_scripts(context, challenge=answer(correct=False))
    summaries = tuple(summary(context, scripts, item) for item in metadata)
    hashes = tuple(canonical_hash(asdict(item)) for item in metadata)
    policy_hash = canonical_hash([_policy_input(script) for script in scripts])
    expected = MftMachineObservation(
        assessment_states=("contradicted",) * 2,
        route_targets=("quarantine",) * 2,
        audit_flags=(False,) * 2,
        routing_reason_codes=("CONTRADICTED",) * 2,
        excluded_metadata_hashes=hashes,
        policy_input_hashes=(policy_hash,) * 2,
    )
    actual = combine(summaries).model_copy(
        update={"excluded_metadata_hashes": hashes, "policy_input_hashes": (policy_hash,) * 2}
    )
    return GateEvaluation(
        ("excluded-metadata-a", "excluded-metadata-b"), expected, actual, "ROUTE_INVARIANCE_FAILED"
    )


def nonharm_gate(context: MftStateContext, divergent: bool, scenario: str) -> GateEvaluation:
    count = context.kappa_candidate.min_total_evaluable_replicates
    actual = summary(
        context,
        build_scripts(context, challenge=answer(correct=True), outputs_diverge=divergent),
    )
    reason = "OUTPUT_DIVERGENCE_WITHOUT_VERIFIED_HARM" if divergent else "NO_HARM_WITNESS"
    expected = MftMachineObservation(
        candidate_final_context_inclusions=(True,) * count,
        assessment_states=("not_contradicted",),
        route_targets=("active",),
        audit_flags=(False,),
        probe_reason_codes=(reason,) * count,
        routing_reason_codes=("NOT_CONTRADICTED",),
    )
    return GateEvaluation((scenario,), expected, actual, "FALSE_QUARANTINE")


def _policy_input(script: Script) -> dict[str, JsonValue]:
    control, challenge = script.control.final, script.challenge.final
    return {
        "probe_id": script.probe_id,
        "control_provider_status": control.provider,
        "control_parse_status": control.parse,
        "control_verifier_status": control.verifier,
        "control_verifier_result": control.correct,
        "challenge_provider_status": challenge.provider,
        "challenge_parse_status": challenge.parse,
        "challenge_verifier_status": challenge.verifier,
        "challenge_verifier_result": challenge.correct,
        "candidate_final_context_inclusion": script.exposed,
        "outputs_diverge": script.outputs_diverge,
    }
