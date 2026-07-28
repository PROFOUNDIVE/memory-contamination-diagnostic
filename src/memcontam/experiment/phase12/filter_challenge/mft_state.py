# allow: SIZE_OK - Task 12 requires one exact eight-gate state-machine registry module.
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final, Literal, TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

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
from memcontam.experiment.phase12.filter_challenge.registry_search import (
    KappaCandidate,
    SuiteCandidate,
)
from memcontam.experiment.phase12.filter_challenge.executor_types import PairingIdentity


MFT_STATE_SCHEMA_VERSION: Final = "filter_challenge_mft_state_v1"
MftStateId: TypeAlias = Literal[
    "MFT-FV5-01-PAIR-MATCH",
    "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE",
    "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE",
    "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT",
    "MFT-FV5-08-NO-WRITEBACK",
]
MFT_STATE_IDS: Final[tuple[MftStateId, ...]] = (
    "MFT-FV5-01-PAIR-MATCH",
    "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE",
    "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE",
    "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT",
    "MFT-FV5-08-NO-WRITEBACK",
)
MftStateMutation: TypeAlias = Literal[
    "none", "pair_identity", "exposure", "route", "source_state"
]
ProviderStatus: TypeAlias = Literal["success", "failed"]
ParseStatus: TypeAlias = Literal["parsed_raw", "parse_failed"]
VerifierStatus: TypeAlias = Literal["success", "failed"] | None
JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | tuple["JsonValue", ...]
    | dict[str, "JsonValue"]
)

_RELATION: Final[TypeAdapter[AnswerCallRelation]] = TypeAdapter(AnswerCallRelation)
_PAIR: Final[TypeAdapter[PairedExecutionIdentity]] = TypeAdapter(PairedExecutionIdentity)
_ROUTABILITY: Final[TypeAdapter[ChallengeRoutability]] = TypeAdapter(ChallengeRoutability)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MftStateContext(_StrictModel):
    evidence_layer: Literal["build"] = "build"
    scientific_result: Literal[False] = False
    fixture_only: Literal[True] = True
    search_config_id: str
    search_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    operational_probe_suite_manifest_id: str
    operational_probe_suite_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_candidate: SuiteCandidate
    kappa_candidate: KappaCandidate

    @model_validator(mode="after")
    def _validate_synthetic_contract(self) -> MftStateContext:
        identities = (
            self.search_config_id,
            self.calibration_probe_inventory_id,
            self.operational_probe_suite_manifest_id,
            self.suite_candidate.operational_probe_suite_id,
            self.kappa_candidate.kappa_id,
            *self.suite_candidate.probe_ids,
        )
        if any("synthetic-build" not in identity for identity in identities):
            raise ValueError("MFT_SYNTHETIC_FIXTURE_REQUIRED")
        kappa, suite = self.kappa_candidate, self.suite_candidate
        if (
            kappa.min_distinct_witness_probes > kappa.min_distinct_evaluable_probes
            or kappa.min_distinct_evaluable_probes > len(suite.probe_ids)
            or kappa.min_witness_replicates_per_probe > suite.replicates_per_probe
            or kappa.min_total_evaluable_replicates
            > len(suite.probe_ids) * suite.replicates_per_probe
        ):
            raise ValueError("KAPPA_INCOHERENT")
        return self


class MftGateInputs(_StrictModel):
    registry_context: MftStateContext
    candidate_entry_id: str
    source_checkpoint_id: str
    source_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_ids: tuple[str, ...]
    provider_calls_issued: Literal[0] = 0


class MftMachineObservation(_StrictModel):
    paired_execution_identity_status: Literal["matched", "mismatched"] | None = None
    paired_identity_fields: tuple[str, ...] = ()
    config_diff_fields: tuple[str, ...] = ()
    control_config_hash: str | None = None
    challenge_config_hash: str | None = None
    candidate_final_context_inclusions: tuple[bool, ...] = ()
    assessment_states: tuple[str, ...] = ()
    route_targets: tuple[str, ...] = ()
    audit_flags: tuple[bool, ...] = ()
    probe_reason_codes: tuple[str, ...] = ()
    routing_reason_codes: tuple[str, ...] = ()
    scripted_attempt_counts: tuple[int, ...] = ()
    excluded_metadata_hashes: tuple[str, ...] = ()
    policy_input_hashes: tuple[str, ...] = ()
    source_state_before_hash: str | None = None
    source_state_after_hash: str | None = None
    challenge_output_artifact_count: int = 0
    challenge_failure_artifact_count: int = 0
    challenge_record_artifact_count: int = 0
    active_memory_write_count: int = 0
    ordinary_trial_write_count: int = 0
    updater_write_count: int = 0


class MftGateResult(_StrictModel):
    schema_version: Literal["filter_challenge_mft_state_v1"] = MFT_STATE_SCHEMA_VERSION
    test_id: MftStateId
    execution_index: int = Field(ge=1, le=8)
    inputs: MftGateInputs
    expected: MftMachineObservation
    actual: MftMachineObservation
    reason: str
    status: Literal["pass", "fail"]
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class MftStateReport(_StrictModel):
    schema_version: Literal["filter_challenge_mft_state_v1"] = MFT_STATE_SCHEMA_VERSION
    evidence_layer: Literal["build"] = "build"
    scientific_result: Literal[False] = False
    fixture_only: Literal[True] = True
    decision_input_kind: Literal["machine_structure"] = "machine_structure"
    ordered_test_ids: tuple[MftStateId, ...]
    results: tuple[MftGateResult, ...]
    provider_calls_issued: Literal[0] = 0

    @model_validator(mode="after")
    def _validate_registry(self) -> MftStateReport:
        if self.ordered_test_ids != MFT_STATE_IDS or tuple(
            result.test_id for result in self.results
        ) != MFT_STATE_IDS:
            raise ValueError("MFT_STATE_REGISTRY_MISMATCH")
        if tuple(result.execution_index for result in self.results) != tuple(range(1, 9)):
            raise ValueError("MFT_STATE_EXECUTION_COUNT_MISMATCH")
        return self


class _Answer(_StrictModel):
    provider: ProviderStatus
    parse: ParseStatus
    verifier: VerifierStatus
    correct: bool | None
    artifact_hash: str


class _FakeClient(_StrictModel):
    attempts: tuple[_Answer, ...]

    @property
    def final(self) -> _Answer:
        return self.attempts[-1]


@dataclass(frozen=True, slots=True)
class _Script:
    probe_id: str
    control: _FakeClient
    challenge: _FakeClient
    exposed: bool
    outputs_diverge: bool


@dataclass(frozen=True, slots=True)
class _Evaluation:
    scenarios: tuple[str, ...]
    expected: MftMachineObservation
    actual: MftMachineObservation
    failure_reason: str


def mft_state_evidence_hash(result: MftGateResult) -> str:
    return _hash(result.model_dump(mode="json", exclude={"evidence_hash"}))


def run_mft_state_gates(
    context: MftStateContext, *, mutation: MftStateMutation = "none"
) -> MftStateReport:
    source_hash = _hash({"active_entry_ids": ["synthetic-source-entry"]})
    common = MftGateInputs(
        registry_context=context,
        candidate_entry_id="synthetic-build-candidate",
        source_checkpoint_id="synthetic-build-checkpoint",
        source_state_hash=source_hash,
        scenario_ids=(),
    )
    results = []
    for index, test_id in enumerate(MFT_STATE_IDS, 1):
        evaluation = _evaluate(test_id, context, mutation, source_hash)
        inputs = common.model_copy(update={"scenario_ids": evaluation.scenarios})
        passed = evaluation.expected == evaluation.actual
        draft = MftGateResult(
            test_id=test_id,
            execution_index=index,
            inputs=inputs,
            expected=evaluation.expected,
            actual=evaluation.actual,
            reason="MFT_GATE_PASSED" if passed else evaluation.failure_reason,
            status="pass" if passed else "fail",
            evidence_hash="0" * 64,
        )
        results.append(draft.model_copy(update={"evidence_hash": mft_state_evidence_hash(draft)}))
    return MftStateReport(ordered_test_ids=MFT_STATE_IDS, results=tuple(results))


def write_mft_state_report(
    output: Path, context: MftStateContext, *, mutation: MftStateMutation = "none"
) -> MftStateReport:
    report = run_mft_state_gates(context, mutation=mutation)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate(
    test_id: MftStateId,
    context: MftStateContext,
    mutation: MftStateMutation,
    source_hash: str,
) -> _Evaluation:
    match test_id:
        case "MFT-FV5-01-PAIR-MATCH":
            return _pair_gate(mutation, source_hash)
        case "MFT-FV5-02-EXPOSURE-REQUIRED":
            count = context.kappa_candidate.min_total_evaluable_replicates
            actual = _summary(
                context,
                _scripts(context, challenge=_answer(correct=False), exposed=mutation == "exposure"),
            )
            expected = MftMachineObservation(
                candidate_final_context_inclusions=(False,) * count,
                assessment_states=("not_evaluable",),
                route_targets=("active",),
                audit_flags=(True,),
                probe_reason_codes=("CANDIDATE_NOT_EXPOSED",) * count,
                routing_reason_codes=("FAIL_OPEN_NOT_EVALUABLE",),
            )
            return _Evaluation(("unexposed-incorrect-challenge",), expected, actual, "CANDIDATE_EXPOSURE_ASSERTION_FAILED")
        case "MFT-FV5-03-TRISTATE":
            summaries = (
                _summary(context, _scripts(context, challenge=_answer(correct=False))),
                _summary(context, _scripts(context, challenge=_answer(correct=True))),
                _summary(context, _scripts(context, challenge=_answer(correct=False), exposed=False)),
            )
            actual = _combine(summaries)
            if mutation == "route":
                actual = actual.model_copy(update={"route_targets": ("active", *actual.route_targets[1:])})
            expected = MftMachineObservation(
                assessment_states=("contradicted", "not_contradicted", "not_evaluable"),
                route_targets=("quarantine", "active", "active"),
                audit_flags=(False, False, True),
                routing_reason_codes=("CONTRADICTED", "NOT_CONTRADICTED", "FAIL_OPEN_NOT_EVALUABLE"),
            )
            return _Evaluation(("contradicted", "not-contradicted", "not-evaluable"), expected, actual, "ROUTING_RECONCILIATION_FAILED")
        case "MFT-FV5-04-FAIL-OPEN":
            return _fail_open_gate(context)
        case "MFT-FV5-05-ROUTE-INVARIANCE":
            return _route_invariance_gate(context)
        case "MFT-FV5-06-SCRIPTED-CORRECT":
            return _nonharm_gate(context, False, "correct-candidate")
        case "MFT-FV5-07-SCRIPTED-IRRELEVANT":
            return _nonharm_gate(context, True, "irrelevant-candidate")
        case "MFT-FV5-08-NO-WRITEBACK":
            artifacts = _FakeClient(
                attempts=(_answer(correct=True), _answer(provider="failed"))
            ).attempts
            after = (
                _hash({"active_entry_ids": ["synthetic-source-entry", "forbidden-output"]})
                if mutation == "source_state"
                else source_hash
            )
            expected = MftMachineObservation(
                source_state_before_hash=source_hash,
                source_state_after_hash=source_hash,
                challenge_output_artifact_count=sum(item.provider == "success" for item in artifacts),
                challenge_failure_artifact_count=sum(item.provider == "failed" for item in artifacts),
                challenge_record_artifact_count=len(artifacts),
            )
            return _Evaluation(
                ("challenge-artifact-1", "challenge-artifact-2"),
                expected,
                expected.model_copy(update={"source_state_after_hash": after}),
                "SOURCE_DRIFT",
            )
        case unreachable:
            assert_never(unreachable)


def _pair_gate(mutation: MftStateMutation, source_hash: str) -> _Evaluation:
    identity: dict[str, JsonValue] = asdict(_pairing_identity(source_hash))
    control: dict[str, JsonValue] = {
        "candidate_entry_id": None,
        "pairing_identity": identity,
        "updater_enabled": False,
    }
    challenge = {**control, "candidate_entry_id": "synthetic-build-candidate"}
    if mutation == "pair_identity":
        challenge["pairing_identity"] = {**identity, "model_snapshot": "mutated-model"}
    diff = tuple(key for key in control if control[key] != challenge[key] and key != "pairing_identity")
    challenge_identity = challenge["pairing_identity"]
    if isinstance(challenge_identity, dict):
        diff += tuple(
            f"pairing_identity.{key}"
            for key in identity
            if identity[key] != challenge_identity[key]
        )
    identity_fields = tuple(field.name for field in fields(PairingIdentity))
    expected = MftMachineObservation(
        paired_execution_identity_status="matched",
        paired_identity_fields=identity_fields,
        config_diff_fields=("candidate_entry_id",),
        control_config_hash=_hash(control),
        challenge_config_hash=_hash({**control, "candidate_entry_id": "synthetic-build-candidate"}),
        source_state_before_hash=source_hash,
        source_state_after_hash=source_hash,
    )
    actual = expected.model_copy(
        update={
            "paired_execution_identity_status": "matched" if diff == ("candidate_entry_id",) else "mismatched",
            "config_diff_fields": diff,
            "challenge_config_hash": _hash(challenge),
        }
    )
    return _Evaluation(("candidate-only-native-diff",), expected, actual, "PAIRED_EXECUTION_IDENTITY_MISMATCH")


def _fail_open_gate(context: MftStateContext) -> _Evaluation:
    failures = (
        ("control-provider", _answer(provider="failed"), _answer(correct=False)),
        ("control-parser", _answer(parse="parse_failed"), _answer(correct=False)),
        ("control-verifier", _answer(verifier="failed"), _answer(correct=False)),
        ("challenge-provider", _answer(), _answer(provider="failed")),
        ("challenge-parser", _answer(), _answer(parse="parse_failed")),
        ("challenge-verifier", _answer(), _answer(verifier="failed")),
    )
    summaries = tuple(
        _summary(context, _scripts(context, control=control, challenge=challenge, attempts=2))
        for _, control, challenge in failures
    )
    reasons = (
        "CONTROL_PROVIDER_FAILURE",
        "CONTROL_PARSE_FAILURE",
        "CONTROL_VERIFIER_FAILURE",
        "CHALLENGE_PROVIDER_FAILURE",
        "CHALLENGE_PARSE_FAILURE",
        "CHALLENGE_VERIFIER_FAILURE",
    )
    expected = MftMachineObservation(
        assessment_states=("not_evaluable",) * 6,
        route_targets=("active",) * 6,
        audit_flags=(True,) * 6,
        probe_reason_codes=reasons,
        routing_reason_codes=("FAIL_OPEN_NOT_EVALUABLE",) * 6,
        scripted_attempt_counts=(2,) * 6,
    )
    actual = _combine(summaries).model_copy(
        update={
            "probe_reason_codes": tuple(summary.probe_reason_codes[0] for summary in summaries),
            "scripted_attempt_counts": (2,) * 6,
        }
    )
    return _Evaluation(tuple(item[0] for item in failures), expected, actual, "FAIL_OPEN_ASSERTION_FAILED")


def _route_invariance_gate(context: MftStateContext) -> _Evaluation:
    metadata = (
        ExcludedCandidateMetadata("quarantine", "direct-route-a", False, ("audit-a",)),
        ExcludedCandidateMetadata("active", "direct-route-b", True, ("audit-b",)),
    )
    scripts = _scripts(context, challenge=_answer(correct=False))
    summaries = tuple(
        _summary(
            context,
            scripts,
            excluded=excluded,
        )
        for excluded in metadata
    )
    hashes = tuple(_hash(asdict(item)) for item in metadata)
    policy_hash = _hash([_policy_input(script) for script in scripts])
    policy_hashes = (policy_hash,) * 2
    expected = MftMachineObservation(
        assessment_states=("contradicted",) * 2,
        route_targets=("quarantine",) * 2,
        audit_flags=(False,) * 2,
        routing_reason_codes=("CONTRADICTED",) * 2,
        excluded_metadata_hashes=hashes,
        policy_input_hashes=policy_hashes,
    )
    actual = _combine(summaries).model_copy(
        update={"excluded_metadata_hashes": hashes, "policy_input_hashes": policy_hashes}
    )
    return _Evaluation(("excluded-metadata-a", "excluded-metadata-b"), expected, actual, "ROUTE_INVARIANCE_FAILED")


def _nonharm_gate(context: MftStateContext, divergent: bool, scenario: str) -> _Evaluation:
    count = context.kappa_candidate.min_total_evaluable_replicates
    actual = _summary(
        context,
        _scripts(context, challenge=_answer(correct=True), outputs_diverge=divergent),
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
    return _Evaluation((scenario,), expected, actual, "FALSE_QUARANTINE")


def _summary(
    context: MftStateContext,
    scripts: tuple[_Script, ...],
    *,
    excluded: ExcludedCandidateMetadata | None = None,
) -> MftMachineObservation:
    results = tuple(
        assess_probe(_probe(script))
        if excluded is None
        else assess_candidate(CandidateAssessmentEnvelope(_probe(script), excluded))
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


def _combine(summaries: tuple[MftMachineObservation, ...]) -> MftMachineObservation:
    return MftMachineObservation(
        assessment_states=tuple(item.assessment_states[0] for item in summaries),
        route_targets=tuple(item.route_targets[0] for item in summaries),
        audit_flags=tuple(item.audit_flags[0] for item in summaries),
        routing_reason_codes=tuple(item.routing_reason_codes[0] for item in summaries),
    )


def _scripts(
    context: MftStateContext,
    *,
    control: _Answer | None = None,
    challenge: _Answer | None = None,
    exposed: bool = True,
    outputs_diverge: bool = False,
    attempts: int = 1,
) -> tuple[_Script, ...]:
    control, challenge = control or _answer(), challenge or _answer(correct=False)
    probe_ids = context.suite_candidate.probe_ids[: context.kappa_candidate.min_distinct_evaluable_probes]
    return tuple(
        _Script(
            probe_ids[index % len(probe_ids)],
            _FakeClient(attempts=(control,) * attempts),
            _FakeClient(attempts=(challenge,) * attempts),
            exposed,
            outputs_diverge,
        )
        for index in range(context.kappa_candidate.min_total_evaluable_replicates)
    )


def _answer(
    *,
    provider: ProviderStatus = "success",
    parse: ParseStatus = "parsed_raw",
    verifier: VerifierStatus = "success",
    correct: bool | None = True,
) -> _Answer:
    result = correct if verifier == "success" and parse == "parsed_raw" else None
    payload: dict[str, JsonValue] = {
        "provider": provider,
        "parse": parse,
        "verifier": verifier,
        "correct": result,
    }
    return _Answer(
        provider=provider,
        parse=parse,
        verifier=verifier,
        correct=result,
        artifact_hash=_hash(payload),
    )


def _pairing_identity(source_hash: str) -> PairingIdentity:
    machine_hash = _hash({"fixture": "synthetic-build-pair-identity"})
    return PairingIdentity(
        source_checkpoint_id="synthetic-build-checkpoint",
        source_checkpoint_hash=source_hash,
        baseline_family="full_history",
        rag_mode="not_applicable",
        candidate_native_kind="full_history_transcript",
        probe_id="synthetic-build-game24-probe-1",
        prompt_payload_hash=machine_hash,
        replicate_seed_contract="deterministic",
        replicate_id=0,
        paired_seed_replay_id="synthetic-build-seed-replay-0",
        model_snapshot="synthetic-model",
        decoding_contract_hash=machine_hash,
        fidelity_label="synthetic-build-fidelity",
        tool_mode="text_only",
        tool_permissions_hash=machine_hash,
        raw_parser_version="synthetic-build-parser-v1",
        canonicalizer_version="synthetic-build-canonicalizer-v1",
        verifier_version="synthetic-build-verifier-v1",
        base_prompt_hash=machine_hash,
        formatter_version="synthetic-build-formatter-v1",
        context_budget_capacity_hash=machine_hash,
        retriever_index_capacity_hash=machine_hash,
        noncandidate_memory_hash=source_hash,
        resource_retry_limit_hash=machine_hash,
    )


def _policy_input(script: _Script) -> dict[str, JsonValue]:
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


def _probe(script: _Script) -> ProbeAssessmentInput:
    control, challenge = script.control.final, script.challenge.final
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


def _hash(value: JsonValue) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = (
    "MFT_STATE_IDS",
    "MFT_STATE_SCHEMA_VERSION",
    "MftGateInputs",
    "MftGateResult",
    "MftMachineObservation",
    "MftStateContext",
    "MftStateMutation",
    "MftStateReport",
    "mft_state_evidence_hash",
    "run_mft_state_gates",
    "write_mft_state_report",
)
