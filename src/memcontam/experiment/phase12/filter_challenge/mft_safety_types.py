from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter

from memcontam.experiment.phase12.filter_challenge.assessment import ProbeAssessmentInput
from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeCandidate,
    ChallengeRoutability,
    PairedExecutionIdentity,
)
from memcontam.experiment.phase12.filter_challenge.executor_types import PairingIdentity


MFT_SAFETY_SCHEMA_VERSION: Final = "filter_challenge_mft_safety_v1"
MFT_SAFETY_IDS: Final = (
    "MFT-FV5-09-CONTAM-SHADOW-SHARE",
    "MFT-FV5-10-PARSER-BOUNDARY",
    "MFT-FV5-11-CONTROL-CACHE",
    "MFT-FV5-12-PROBE-KEY-INVARIANCE",
    "MFT-FV5-13-ANSWER-CALL-PROVENANCE",
    "MFT-FV5-14-ACTIVATION-DOMAIN",
    "MFT-FV5-15-ELIGIBILITY-STATES",
    "MFT-FV5-16-COVERAGE-NOT-ESTIMABLE",
)
MFT_SAFETY_FAILURE_REASONS: Final = (
    "CONTAM_SHADOW_SHARE_IMPLEMENTATION_FAILURE",
    "PARSER_BOUNDARY_IMPLEMENTATION_FAILURE",
    "CONTROL_CACHE_IMPLEMENTATION_FAILURE",
    "PROBE_KEY_INVARIANCE_IMPLEMENTATION_FAILURE",
    "ANSWER_CALL_PROVENANCE_IMPLEMENTATION_FAILURE",
    "ACTIVATION_DOMAIN_IMPLEMENTATION_FAILURE",
    "ELIGIBILITY_STATES_IMPLEMENTATION_FAILURE",
    "COVERAGE_ESTIMABILITY_IMPLEMENTATION_FAILURE",
)
CONTROL_CACHE_FIELDS: Final = (
    "source_checkpoint_hash", "baseline_family", "rag_mode", "probe_id",
    "prompt_payload_hash", "replicate_seed_contract", "replicate_id",
    "paired_seed_replay_id", "model_snapshot", "decoding_contract_hash",
    "fidelity_label", "tool_mode", "tool_permissions_hash", "raw_parser_version",
    "canonicalizer_version", "verifier_version", "base_prompt_hash",
    "formatter_version", "context_budget_capacity_hash", "retriever_index_capacity_hash",
    "noncandidate_memory_hash", "resource_retry_limit_hash",
)
NOT_ESTIMABLE: Final = "FILTER_V5_PILOT_B_NOT_ESTIMABLE"


class _StrictMftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MftIdentity(_StrictMftModel):
    field: str
    value: str


class MftAssertion(_StrictMftModel):
    field: str
    expected: tuple[str, ...]
    actual: tuple[str, ...]
    matched: bool


class MftSafetyCase(_StrictMftModel):
    test_id: str
    input_identities: tuple[MftIdentity, ...]
    assertions: tuple[MftAssertion, ...]
    status: Literal["pass", "implementation_failure"]
    reason_code: str | None
    evidence_hash: str


class MftExecutionCount(_StrictMftModel):
    test_id: str
    count: int


class MftSafetyReport(_StrictMftModel):
    schema_version: Literal["filter_challenge_mft_safety_v1"] = MFT_SAFETY_SCHEMA_VERSION
    test_ids: tuple[str, ...]
    cases: tuple[MftSafetyCase, ...]
    execution_counts: tuple[MftExecutionCount, ...]
    all_passed: bool
    provider_calls_issued: Literal[0] = 0
    decision_inputs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GateEvidence:
    identities: tuple[MftIdentity, ...]
    assertions: tuple[MftAssertion, ...]


Gate: TypeAlias = Callable[[bool], GateEvidence]
RELATION_ADAPTER: Final[TypeAdapter[AnswerCallRelation]] = TypeAdapter(AnswerCallRelation)
PAIR_ADAPTER: Final[TypeAdapter[PairedExecutionIdentity]] = TypeAdapter(PairedExecutionIdentity)
ROUTABILITY_ADAPTER: Final[TypeAdapter[ChallengeRoutability]] = TypeAdapter(ChallengeRoutability)


@dataclass(frozen=True, slots=True)
class MftSafetyError(ValueError):
    code: Literal["INVALID_MFT_SAFETY_MUTATION"]

    def __str__(self) -> str:
        return self.code


def assertion(field: str, expected: tuple[str, ...], actual: tuple[str, ...]) -> MftAssertion:
    return MftAssertion(field=field, expected=expected, actual=actual, matched=expected == actual)


def pairing_identity() -> PairingIdentity:
    return PairingIdentity(
        source_checkpoint_id="checkpoint-1", source_checkpoint_hash="checkpoint-hash",
        baseline_family="full_history", rag_mode="not_applicable",
        candidate_native_kind="full_history_transcript", probe_id="probe-1",
        prompt_payload_hash="prompt-payload-hash", replicate_seed_contract="deterministic",
        replicate_id=0, paired_seed_replay_id="seed-replay-1", model_snapshot="model-1",
        decoding_contract_hash="decoding-hash", fidelity_label="fidelity-1",
        tool_mode="none", tool_permissions_hash="tool-hash", raw_parser_version="parser-v1",
        canonicalizer_version="canonicalizer-v1", verifier_version="verifier-v1",
        base_prompt_hash="base-prompt-hash", formatter_version="formatter-v1",
        context_budget_capacity_hash="context-hash", retriever_index_capacity_hash="index-hash",
        noncandidate_memory_hash="memory-hash", resource_retry_limit_hash="retry-hash",
    )


def candidate() -> ChallengeCandidate:
    return ChallengeCandidate(
        candidate_entry_id="candidate-1", candidate_native_content="candidate-fixture",
        candidate_native_kind="full_history_transcript", baseline_family="full_history",
        rag_mode="not_applicable", source_checkpoint_id="checkpoint-1",
        source_active_state_hash="checkpoint-hash",
        routability=ROUTABILITY_ADAPTER.validate_python(
            {"routability": "challenge_routable_v1", "challenge_suite_key": "k9m2x7"}
        ),
    )


def relation(status: str, answer_call_id: str) -> AnswerCallRelation:
    payload = {"answer_call_provenance_status": status, "answer_call_id": answer_call_id}
    if status == "explicit_matched":
        payload |= {
            "parsed_response_source_call_id": answer_call_id,
            "parser_result_id": f"{answer_call_id}:parser",
            "verifier_result_id": f"{answer_call_id}:verifier",
        }
    return RELATION_ADAPTER.validate_python(payload)


def probe_input(candidate_id: str, probe_id: str, suite_key: str) -> ProbeAssessmentInput:
    return ProbeAssessmentInput(
        probe_id=probe_id, control_provider_status="success", control_raw_parse_status="parsed_raw",
        control_verifier_status="success", control_verifier_result=True,
        control_relation=relation("explicit_matched", f"{candidate_id}:control"),
        control_canonicalizer_version=None, control_canonicalized_parse_status=None,
        control_canonicalized_verifier_status=None, control_canonicalized_verifier_result=None,
        challenge_provider_status="success", challenge_raw_parse_status="parsed_raw",
        challenge_verifier_status="success", challenge_verifier_result=False,
        challenge_relation=relation("explicit_matched", f"{candidate_id}:challenge"),
        candidate_exposure=CandidateExposureRecord(
            candidate_entry_id=candidate_id, candidate_final_context_inclusion=True,
            candidate_final_context_source_ids=(candidate_id,),
        ),
        routability=ROUTABILITY_ADAPTER.validate_python(
            {"routability": "challenge_routable_v1", "challenge_suite_key": suite_key}
        ),
        pair_identity=PAIR_ADAPTER.validate_python(
            {"paired_execution_identity_status": "matched", "pair_id": candidate_id}
        ),
        outputs_diverge=True,
    )
