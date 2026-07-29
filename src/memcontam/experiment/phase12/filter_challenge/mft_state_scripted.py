from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge.assessment import (
    CandidateAssessmentEnvelope,
    ExcludedCandidateMetadata,
    ProbeAssessmentInput,
    aggregate_assessments,
    assess_candidate,
    assess_probe,
    route_assessment,
)
from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeRoutability,
    PairedExecutionIdentity,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import (
    JsonValue,
    MftMachineObservation,
    MftStateContext,
    StrictMftModel,
    canonical_hash,
)

ProviderStatus: TypeAlias = Literal["success", "failed"]
ParseStatus: TypeAlias = Literal["parsed_raw", "parse_failed"]
VerifierStatus: TypeAlias = Literal["success", "failed"] | None

_RELATION: Final[TypeAdapter[AnswerCallRelation]] = TypeAdapter(AnswerCallRelation)
_PAIR: Final[TypeAdapter[PairedExecutionIdentity]] = TypeAdapter(PairedExecutionIdentity)
_ROUTABILITY: Final[TypeAdapter[ChallengeRoutability]] = TypeAdapter(ChallengeRoutability)


class Answer(StrictMftModel):
    provider: ProviderStatus
    parse: ParseStatus
    verifier: VerifierStatus
    correct: bool | None
    artifact_hash: str


class _FakeClient(StrictMftModel):
    attempts: tuple[Answer, ...]

    @property
    def final(self) -> Answer:
        return self.attempts[-1]

    @property
    def persistent(self) -> Answer:
        final = self.final
        provider: ProviderStatus = (
            "failed" if all(item.provider == "failed" for item in self.attempts) else "success"
        )
        parse: ParseStatus = (
            "parse_failed"
            if all(item.parse == "parse_failed" for item in self.attempts)
            else "parsed_raw"
        )
        verifier: VerifierStatus = (
            "failed" if all(item.verifier == "failed" for item in self.attempts) else "success"
        )
        correct = final.correct if (provider, parse, verifier) == (
            "success",
            "parsed_raw",
            "success",
        ) else None
        return final.model_copy(
            update={"provider": provider, "parse": parse, "verifier": verifier, "correct": correct}
        )


@dataclass(frozen=True, slots=True)
class Script:
    probe_id: str
    control: _FakeClient
    challenge: _FakeClient
    exposed: bool
    outputs_diverge: bool


def answer(
    *,
    provider: ProviderStatus = "success",
    parse: ParseStatus = "parsed_raw",
    verifier: VerifierStatus = "success",
    correct: bool | None = True,
) -> Answer:
    result = correct if verifier == "success" and parse == "parsed_raw" else None
    payload: dict[str, JsonValue] = {
        "provider": provider,
        "parse": parse,
        "verifier": verifier,
        "correct": result,
    }
    return Answer(
        provider=provider,
        parse=parse,
        verifier=verifier,
        correct=result,
        artifact_hash=canonical_hash(payload),
    )


def build_scripts(
    context: MftStateContext,
    *,
    control: Answer | None = None,
    challenge: Answer | None = None,
    exposed: bool = True,
    outputs_diverge: bool = False,
    attempts: int = 1,
) -> tuple[Script, ...]:
    control, challenge = control or answer(), challenge or answer(correct=False)
    probe_ids = context.suite_candidate.probe_ids[
        : context.kappa_candidate.min_distinct_evaluable_probes
    ]
    return tuple(
        Script(
            probe_ids[index % len(probe_ids)],
            _FakeClient(attempts=(control,) * attempts),
            _FakeClient(attempts=(challenge,) * attempts),
            exposed,
            outputs_diverge,
        )
        for index in range(context.kappa_candidate.min_total_evaluable_replicates)
    )


def summary(
    context: MftStateContext,
    scripts: tuple[Script, ...],
    excluded: ExcludedCandidateMetadata | None = None,
) -> MftMachineObservation:
    results = tuple(
        assess_probe(_probe(script, persistent=False))
        if excluded is None
        else assess_candidate(
            CandidateAssessmentEnvelope(_probe(script, persistent=False), excluded)
        )
        for script in scripts
    )
    state = aggregate_assessments(results, context.kappa_candidate, context.suite_candidate)
    route = route_assessment(state.assessment_state)
    return MftMachineObservation(
        candidate_final_context_inclusions=tuple(script.exposed for script in scripts),
        assessment_states=(state.assessment_state,),
        route_targets=(route.route_target,),
        audit_flags=(route.audit_flag,),
        probe_reason_codes=tuple(result.disposition.reason_code for result in results),
        routing_reason_codes=(route.routing_reason_code,),
    )


def persistent_summary(
    context: MftStateContext, scripts: tuple[Script, ...]
) -> MftMachineObservation:
    results = tuple(assess_probe(_probe(script, persistent=True)) for script in scripts)
    state = aggregate_assessments(results, context.kappa_candidate, context.suite_candidate)
    route = route_assessment(state.assessment_state)
    return MftMachineObservation(
        candidate_final_context_inclusions=tuple(script.exposed for script in scripts),
        assessment_states=(state.assessment_state,),
        route_targets=(route.route_target,),
        audit_flags=(route.audit_flag,),
        probe_reason_codes=tuple(result.disposition.reason_code for result in results),
        routing_reason_codes=(route.routing_reason_code,),
    )


def failure_attempt_counts(scripts: tuple[Script, ...]) -> tuple[int, ...]:
    counts = []
    for script in scripts:
        control = script.control.persistent
        failing = script.control if (
            control.provider == "failed"
            or control.parse == "parse_failed"
            or control.verifier == "failed"
        ) else script.challenge
        counts.append(len(failing.attempts))
    return tuple(counts)


def combine(summaries: tuple[MftMachineObservation, ...]) -> MftMachineObservation:
    return MftMachineObservation(
        assessment_states=tuple(item.assessment_states[0] for item in summaries),
        route_targets=tuple(item.route_targets[0] for item in summaries),
        audit_flags=tuple(item.audit_flags[0] for item in summaries),
        routing_reason_codes=tuple(item.routing_reason_codes[0] for item in summaries),
    )


def _probe(script: Script, *, persistent: bool) -> ProbeAssessmentInput:
    control = script.control.persistent if persistent else script.control.final
    challenge = script.challenge.persistent if persistent else script.challenge.final
    return ProbeAssessmentInput(
        probe_id=script.probe_id,
        control_provider_status=control.provider,
        control_raw_parse_status=control.parse,
        control_verifier_status=control.verifier,
        control_verifier_result=control.correct,
        control_relation=_relation("control"),
        control_canonicalizer_version=None,
        control_canonicalized_parse_status=None,
        control_canonicalized_verifier_status=None,
        control_canonicalized_verifier_result=None,
        challenge_provider_status=challenge.provider,
        challenge_raw_parse_status=challenge.parse,
        challenge_verifier_status=challenge.verifier,
        challenge_verifier_result=challenge.correct,
        challenge_relation=_relation("challenge"),
        candidate_exposure=CandidateExposureRecord(
            candidate_entry_id="synthetic-build-candidate",
            candidate_final_context_inclusion=script.exposed,
            candidate_final_context_source_ids=("synthetic-build-candidate",) if script.exposed else (),
        ),
        routability=_ROUTABILITY.validate_python(
            {"routability": "challenge_routable_v1", "challenge_suite_key": "synthetic-build-suite-key"}
        ),
        pair_identity=_PAIR.validate_python(
            {"paired_execution_identity_status": "matched", "pair_id": "synthetic-build-pair"}
        ),
        outputs_diverge=script.outputs_diverge,
    )


def _relation(side: Literal["control", "challenge"]) -> AnswerCallRelation:
    call_id = f"synthetic-build-{side}-answer-call"
    return _RELATION.validate_python(
        {
            "answer_call_provenance_status": "explicit_matched",
            "answer_call_id": call_id,
            "parsed_response_source_call_id": call_id,
            "parser_result_id": f"synthetic-build-{side}-parser",
            "verifier_result_id": f"synthetic-build-{side}-verifier",
        }
    )
