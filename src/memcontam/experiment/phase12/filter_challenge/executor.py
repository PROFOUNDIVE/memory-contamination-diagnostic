from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, assert_never

from pydantic import TypeAdapter

from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    ChallengeCandidate,
    ChallengeRoutingDecision,
    PairedExecutionIdentity,
)
from memcontam.experiment.phase12.filter_challenge.executor_source import source_snapshot
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    BoTExecutionRequest,
    ControlCacheKey,
    ControlResultCache,
    FullHistoryExecutionRequest,
    IsolatedPairRequest,
    PairArmResult,
    PairAuditEvidence,
    PairExecutionSinks,
    PairExecutorError,
    PairingIdentity,
    PairIsolation,
    RagFrozenExecutionRequest,
    ReflexionExecutionRequest,
    SharedAssessmentKey,
    SourceSnapshot,
    execution_clients,
)
from memcontam.experiment.phase12.filter_challenge.native_execution import execute_native_pair

_PAIRED_IDENTITY: TypeAdapter[PairedExecutionIdentity] = TypeAdapter(PairedExecutionIdentity)
__all__ = (
    "ActivationContext",
    "ActivationDecision",
    "BoTExecutionRequest",
    "ControlResultCache",
    "FullHistoryExecutionRequest",
    "IsolatedPairRequest",
    "PairAuditEvidence",
    "PairExecutionSinks",
    "PairExecutorError",
    "PairingIdentity",
    "PairIsolation",
    "RagFrozenExecutionRequest",
    "ReflexionExecutionRequest",
    "RoutingConsumption",
    "activation_decision",
    "build_control_cache_key",
    "build_shared_assessment_key",
    "consume_routing",
    "evaluability_rate",
    "execute_isolated_pair",
)


@dataclass(frozen=True, slots=True)
class ActivationContext:
    policy_activation_checkpoint_id: str
    evolved_branch_checkpoint_id: str | None
    grandfathered_entry_ids: tuple[str, ...]
    controlled_root_entry_id: str
    arm: Literal["contam", "filter"]


@dataclass(frozen=True, slots=True)
class ActivationDecision:
    status: Literal["grandfathered", "assess", "not_evaluable", "not_assessed"]
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class RoutingConsumption:
    effect: Literal["shadow", "apply"]
    route_target: Literal["active", "quarantine"] | None
    shared_assessment_key: SharedAssessmentKey


def build_control_cache_key(identity: PairingIdentity) -> ControlCacheKey:
    return ControlCacheKey(
        source_checkpoint_hash=identity.source_checkpoint_hash,
        baseline_family=identity.baseline_family,
        rag_mode=identity.rag_mode,
        probe_id=identity.probe_id,
        prompt_payload_hash=identity.prompt_payload_hash,
        replicate_seed_contract=identity.replicate_seed_contract,
        model_snapshot=identity.model_snapshot,
        decoding_contract_hash=identity.decoding_contract_hash,
        fidelity_label=identity.fidelity_label,
        tool_mode=identity.tool_mode,
        tool_permissions_hash=identity.tool_permissions_hash,
        raw_parser_version=identity.raw_parser_version,
        canonicalizer_version=identity.canonicalizer_version,
        verifier_version=identity.verifier_version,
        base_prompt_hash=identity.base_prompt_hash,
        formatter_version=identity.formatter_version,
        context_budget_capacity_hash=identity.context_budget_capacity_hash,
        retriever_index_capacity_hash=identity.retriever_index_capacity_hash,
        noncandidate_memory_hash=identity.noncandidate_memory_hash,
        resource_retry_limit_hash=identity.resource_retry_limit_hash,
    )


def build_shared_assessment_key(
    identity: PairingIdentity,
    candidate: ChallengeCandidate,
    candidate_version: str,
    probe_configuration_hash: str,
) -> SharedAssessmentKey:
    return SharedAssessmentKey(
        candidate_entry_id=candidate.candidate_entry_id,
        candidate_content_hash=sha256(candidate.candidate_native_content.encode()).hexdigest(),
        candidate_native_kind=candidate.candidate_native_kind,
        candidate_version=candidate_version,
        source_checkpoint_id=identity.source_checkpoint_id,
        source_checkpoint_hash=identity.source_checkpoint_hash,
        probe_configuration_hash=probe_configuration_hash,
        pairing_identity=identity,
        control_cache_key=build_control_cache_key(identity),
    )


def execute_isolated_pair(request: IsolatedPairRequest) -> PairAuditEvidence:
    _validate_isolation(request)
    source_before = source_snapshot(request.execution)
    _validate_source(request, source_before)
    cache_key = build_control_cache_key(request.identity)
    cache_allowed = _cache_allowed(request.identity)
    if not cache_allowed and request.control_cache is not None:
        cached = None
    else:
        cached = request.control_cache.get(cache_key) if request.control_cache is not None else None
    native = execute_native_pair(request.execution, request.candidate, cached, request.execution_order)
    _validate_arm_result(native.control, request.candidate, challenge=False)
    _validate_arm_result(native.challenge, request.candidate, challenge=True)
    if source_snapshot(request.execution) != source_before:
        raise PairExecutorError("SOURCE_DRIFT")
    _validate_relations(native.control.answer_relation, native.challenge.answer_relation)
    if cache_allowed and cached is None and request.control_cache is not None:
        request.control_cache.put(cache_key, native.cache_value)
    paired_identity = _PAIRED_IDENTITY.validate_python(
        {"paired_execution_identity_status": "matched", "pair_id": request.assessment_id}
    )
    shared_key = build_shared_assessment_key(
        request.identity,
        request.candidate,
        request.candidate_version,
        request.probe_configuration_hash,
    )
    evidence = PairAuditEvidence(
        assessment_id=request.assessment_id,
        adapter_name=native.adapter_name,
        paired_execution_identity=paired_identity,
        control_cache_key=cache_key,
        shared_assessment_key=shared_key,
        control_from_cache=cached is not None,
        control_answer_call_id=native.control.answer_relation.answer_call_id,
        challenge_answer_call_id=native.challenge.answer_relation.answer_call_id,
        control_displaced_noncandidate_entry_ids=(
            native.control.displaced_noncandidate_entry_ids
        ),
        challenge_displaced_noncandidate_entry_ids=(
            native.challenge.displaced_noncandidate_entry_ids
        ),
    )
    request.sinks.assessments.append_assessment(evidence)
    return evidence


def activation_decision(
    context: ActivationContext, candidate: ChallengeCandidate, *, later_native_write: bool
) -> ActivationDecision:
    if candidate.candidate_entry_id in context.grandfathered_entry_ids:
        return ActivationDecision("grandfathered", None)
    if later_native_write:
        checkpoint_id = context.evolved_branch_checkpoint_id
        if checkpoint_id is None or checkpoint_id == context.policy_activation_checkpoint_id:
            raise PairExecutorError("EVOLVED_BRANCH_CHECKPOINT_REQUIRED")
    else:
        checkpoint_id = context.policy_activation_checkpoint_id
    if candidate.source_checkpoint_id != checkpoint_id:
        raise PairExecutorError("CANDIDATE_CHECKPOINT_MISMATCH")
    match candidate.routability.routability:
        case "unsupported":
            return ActivationDecision("not_evaluable", "PROBE_MAPPING_UNSUPPORTED")
        case "challenge_routable_v1":
            if candidate.candidate_entry_id == context.controlled_root_entry_id:
                return ActivationDecision("assess", None)
            if candidate.baseline_family == "rag_frozen" and later_native_write:
                raise PairExecutorError("RAG_FROZEN_LATER_WRITE")
            if later_native_write and context.arm == "filter":
                return ActivationDecision("assess", None)
            return ActivationDecision("not_assessed", None)
        case unreachable:
            assert_never(unreachable)


def consume_routing(
    arm: Literal["contam", "filter"],
    routing: ChallengeRoutingDecision,
    shared_assessment_key: SharedAssessmentKey,
) -> RoutingConsumption:
    match arm:
        case "contam":
            return RoutingConsumption("shadow", None, shared_assessment_key)
        case "filter":
            return RoutingConsumption("apply", routing.route_target, shared_assessment_key)
        case unreachable:
            assert_never(unreachable)


def evaluability_rate(evaluable_count: int, attempted_count: int) -> float | None:
    if attempted_count == 0:
        return None
    return evaluable_count / attempted_count


def _validate_isolation(request: IsolatedPairRequest) -> None:
    isolation = request.isolation
    if isolation.control_session_id == isolation.challenge_session_id:
        raise PairExecutorError("DUPLICATE_SESSION_ID")
    if isolation.control_transcript == isolation.challenge_transcript:
        raise PairExecutorError("SHARED_TRANSCRIPT")
    control_client, challenge_client = execution_clients(request.execution)
    if control_client is challenge_client:
        raise PairExecutorError("SHARED_CLIENT")
    if request.execution.family != request.identity.baseline_family:
        raise PairExecutorError("PAIRED_EXECUTION_IDENTITY_MISMATCH")
    if request.execution_order == "challenge_first" and _cache_allowed(request.identity):
        raise PairExecutorError("COUNTERBALANCED_ORDER_REQUIRED")


def _validate_source(request: IsolatedPairRequest, source: SourceSnapshot) -> None:
    identity = request.identity
    candidate = request.candidate
    if (
        candidate.source_checkpoint_id != source.checkpoint_id
        or candidate.source_active_state_hash != source.canonical_sha256
        or identity.source_checkpoint_id != source.checkpoint_id
        or identity.source_checkpoint_hash != source.canonical_sha256
        or identity.noncandidate_memory_hash != source.noncandidate_memory_hash
        or identity.baseline_family != candidate.baseline_family
        or identity.rag_mode != candidate.rag_mode
        or identity.candidate_native_kind != candidate.candidate_native_kind
    ):
        raise PairExecutorError("CHECKPOINT_IDENTITY_MISMATCH")


def _validate_arm_result(
    result: PairArmResult, candidate: ChallengeCandidate, *, challenge: bool
) -> None:
    if result.updater_enabled or result.memory_write_event_id is not None:
        raise PairExecutorError("UPDATER_NOT_DISABLED")
    if result.candidate_exposure.candidate_entry_id != candidate.candidate_entry_id:
        raise PairExecutorError("CANDIDATE_EXPOSURE_BINDING_MISMATCH")
    if not challenge and result.candidate_exposure.candidate_final_context_inclusion:
        raise PairExecutorError("CONTROL_CANDIDATE_EXPOSURE")


def _validate_relations(control: AnswerCallRelation, challenge: AnswerCallRelation) -> None:
    if (
        control.answer_call_provenance_status != "explicit_matched"
        or challenge.answer_call_provenance_status != "explicit_matched"
        or control.answer_call_id == challenge.answer_call_id
    ):
        raise PairExecutorError("ANSWER_CALL_RELATION_UNRESOLVED")


def _cache_allowed(identity: PairingIdentity) -> bool:
    match identity.replicate_seed_contract:
        case "deterministic" | "seed_coupled":
            return True
        case "counterbalanced":
            return False
        case unreachable:
            assert_never(unreachable)
