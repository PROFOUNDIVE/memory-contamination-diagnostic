from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Final, Literal, TypeAlias, assert_never

from pydantic import TypeAdapter

from memcontam.clients.base import LLMClient
from memcontam.experiment.phase12.filter_challenge.adapters.bot_style import BoTStyleChallengeAdapter
from memcontam.experiment.phase12.filter_challenge.adapters.full_history import (
    FullHistoryProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.adapters.rag_frozen import (
    RagFrozenProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.adapters.reflexion_style import (
    ReflexionProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeCandidate,
    ChallengeRoutingDecision,
    PairedExecutionIdentity,
)


NativeAdapter: TypeAlias = (
    FullHistoryProvisionalAdapter
    | RagFrozenProvisionalAdapter
    | BoTStyleChallengeAdapter
    | ReflexionProvisionalAdapter
)
_PAIRED_IDENTITY: Final[TypeAdapter[PairedExecutionIdentity]] = TypeAdapter(PairedExecutionIdentity)


class PairExecutorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    checkpoint_id: str
    canonical_bytes: bytes
    canonical_sha256: str
    noncandidate_entry_ids: tuple[str, ...]
    noncandidate_memory_hash: str

    def __post_init__(self) -> None:
        if sha256(self.canonical_bytes).hexdigest() != self.canonical_sha256:
            raise PairExecutorError("SOURCE_HASH_MISMATCH")
        memory_hash = sha256("\0".join(self.noncandidate_entry_ids).encode()).hexdigest()
        if memory_hash != self.noncandidate_memory_hash:
            raise PairExecutorError("NONCANDIDATE_MEMORY_HASH_MISMATCH")


@dataclass(frozen=True, slots=True)
class PairingIdentity:
    source_checkpoint_id: str
    source_checkpoint_hash: str
    baseline_family: Literal["full_history", "rag_frozen", "bot_style", "reflexion_style"]
    rag_mode: Literal["frozen", "not_applicable"]
    candidate_native_kind: str
    probe_id: str
    prompt_payload_hash: str
    replicate_seed_contract: Literal["deterministic", "seed_coupled", "counterbalanced"]
    model_snapshot: str
    decoding_contract_hash: str
    fidelity_label: str
    tool_mode: str
    tool_permissions_hash: str
    raw_parser_version: str
    canonicalizer_version: str
    verifier_version: str
    base_prompt_hash: str
    formatter_version: str
    context_budget_capacity_hash: str
    retriever_index_capacity_hash: str
    noncandidate_memory_hash: str
    resource_retry_limit_hash: str


@dataclass(frozen=True, slots=True)
class ControlCacheKey:
    source_checkpoint_hash: str
    baseline_family: str
    rag_mode: str
    probe_id: str
    replicate_seed_contract: str
    model_snapshot: str
    decoding_contract_hash: str
    tool_mode: str
    tool_permissions_hash: str
    raw_parser_version: str
    canonicalizer_version: str
    verifier_version: str
    base_prompt_hash: str
    formatter_version: str
    context_budget_capacity_hash: str
    retriever_index_capacity_hash: str


@dataclass(frozen=True, slots=True)
class SharedAssessmentKey:
    candidate_entry_id: str
    candidate_content_hash: str
    candidate_native_kind: str
    candidate_version: str
    source_checkpoint_id: str
    source_checkpoint_hash: str
    probe_configuration_hash: str
    pairing_identity: PairingIdentity
    control_cache_key: ControlCacheKey


@dataclass(frozen=True, slots=True)
class ScriptedSession:
    session_id: str


@dataclass(frozen=True, slots=True)
class PairArmResult:
    answer_relation: AnswerCallRelation
    candidate_exposure: CandidateExposureRecord
    updater_enabled: bool
    memory_write_event_id: str | None
    displaced_noncandidate_entry_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IsolatedPairArm:
    identity: PairingIdentity
    client: LLMClient
    session: ScriptedSession
    transcript: list[str]
    run: Callable[[NativeAdapter], PairArmResult]


class ControlResultCache:
    def __init__(self) -> None:
        self._entries: dict[ControlCacheKey, PairArmResult] = {}

    def get(self, key: ControlCacheKey) -> PairArmResult | None:
        return self._entries.get(key)

    def put(self, key: ControlCacheKey, result: PairArmResult) -> None:
        self._entries[key] = result


@dataclass(frozen=True, slots=True)
class IsolatedPairRequest:
    assessment_id: str
    candidate: ChallengeCandidate
    candidate_version: str
    source: SourceSnapshot
    source_snapshot: Callable[[], SourceSnapshot]
    control: IsolatedPairArm
    challenge: IsolatedPairArm
    probe_configuration_hash: str
    ordinary_trial_ids: tuple[str, ...]
    control_cache: ControlResultCache | None


@dataclass(frozen=True, slots=True)
class PairAuditEvidence:
    assessment_id: str
    adapter_name: str
    paired_execution_identity: PairedExecutionIdentity
    control_cache_key: ControlCacheKey
    shared_assessment_key: SharedAssessmentKey
    control_from_cache: bool
    control_answer_call_id: str
    challenge_answer_call_id: str
    control_displaced_noncandidate_entry_ids: tuple[str, ...]
    challenge_displaced_noncandidate_entry_ids: tuple[str, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["paired_execution_identity"] = self.paired_execution_identity.model_dump(
            mode="json"
        )
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True, slots=True)
class ActivationContext:
    policy_activation_checkpoint_id: str
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


def build_control_cache_key(identity: PairingIdentity) -> ControlCacheKey:
    return ControlCacheKey(
        source_checkpoint_hash=identity.source_checkpoint_hash,
        baseline_family=identity.baseline_family,
        rag_mode=identity.rag_mode,
        probe_id=identity.probe_id,
        replicate_seed_contract=identity.replicate_seed_contract,
        model_snapshot=identity.model_snapshot,
        decoding_contract_hash=identity.decoding_contract_hash,
        tool_mode=identity.tool_mode,
        tool_permissions_hash=identity.tool_permissions_hash,
        raw_parser_version=identity.raw_parser_version,
        canonicalizer_version=identity.canonicalizer_version,
        verifier_version=identity.verifier_version,
        base_prompt_hash=identity.base_prompt_hash,
        formatter_version=identity.formatter_version,
        context_budget_capacity_hash=identity.context_budget_capacity_hash,
        retriever_index_capacity_hash=identity.retriever_index_capacity_hash,
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
    source_before = request.source_snapshot()
    _validate_source(request, source_before)
    adapter = select_adapter(request.candidate)
    key = build_control_cache_key(request.control.identity)
    control = request.control_cache.get(key) if _cache_allowed(request.control.identity) and request.control_cache else None
    control_from_cache = control is not None
    if control is None:
        control = request.control.run(adapter)
        if request.control_cache is not None and _cache_allowed(request.control.identity):
            request.control_cache.put(key, control)
    challenge = request.challenge.run(adapter)
    _validate_arm_result(control, request.candidate)
    _validate_arm_result(challenge, request.candidate)
    if request.source_snapshot() != source_before:
        raise PairExecutorError("SOURCE_DRIFT")
    _validate_relations(control.answer_relation, challenge.answer_relation)
    paired_identity = _PAIRED_IDENTITY.validate_python(
        {"paired_execution_identity_status": "matched", "pair_id": request.assessment_id}
    )
    return PairAuditEvidence(
        assessment_id=request.assessment_id,
        adapter_name=type(adapter).__name__,
        paired_execution_identity=paired_identity,
        control_cache_key=key,
        shared_assessment_key=build_shared_assessment_key(
            request.control.identity,
            request.candidate,
            request.candidate_version,
            request.probe_configuration_hash,
        ),
        control_from_cache=control_from_cache,
        control_answer_call_id=control.answer_relation.answer_call_id,
        challenge_answer_call_id=challenge.answer_relation.answer_call_id,
        control_displaced_noncandidate_entry_ids=control.displaced_noncandidate_entry_ids,
        challenge_displaced_noncandidate_entry_ids=challenge.displaced_noncandidate_entry_ids,
    )


def select_adapter(candidate: ChallengeCandidate) -> NativeAdapter:
    match (candidate.baseline_family, candidate.rag_mode, candidate.candidate_native_kind):
        case ("full_history", "not_applicable", "full_history_transcript"):
            return FullHistoryProvisionalAdapter()
        case ("rag_frozen", "frozen", "rag_document"):
            return RagFrozenProvisionalAdapter()
        case ("bot_style", "not_applicable", "thought_template"):
            return BoTStyleChallengeAdapter()
        case ("reflexion_style", "not_applicable", "verbal_reflection"):
            return ReflexionProvisionalAdapter()
        case _:
            raise PairExecutorError("PROBE_MAPPING_UNSUPPORTED")


def activation_decision(
    context: ActivationContext, candidate: ChallengeCandidate, *, later_native_write: bool
) -> ActivationDecision:
    if candidate.candidate_entry_id in context.grandfathered_entry_ids:
        return ActivationDecision("grandfathered", None)
    if candidate.source_checkpoint_id != context.policy_activation_checkpoint_id:
        raise PairExecutorError("ACTIVATION_CHECKPOINT_MISMATCH")
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
    arm: Literal["contam", "filter"], routing: ChallengeRoutingDecision
) -> RoutingConsumption:
    match arm:
        case "contam":
            return RoutingConsumption("shadow", None)
        case "filter":
            return RoutingConsumption("apply", routing.route_target)
        case unreachable:
            assert_never(unreachable)


def evaluability_rate(evaluable_count: int, attempted_count: int) -> float | None:
    if attempted_count == 0:
        return None
    return evaluable_count / attempted_count


def _validate_isolation(request: IsolatedPairRequest) -> None:
    if request.assessment_id in request.ordinary_trial_ids:
        raise PairExecutorError("ASSESSMENT_IN_ORDINARY_TRIALS")
    if request.control.identity != request.challenge.identity:
        raise PairExecutorError("PAIRED_EXECUTION_IDENTITY_MISMATCH")
    if request.control.client is request.challenge.client:
        raise PairExecutorError("SHARED_CLIENT")
    if request.control.session is request.challenge.session:
        raise PairExecutorError("SHARED_SESSION")
    if request.control.transcript is request.challenge.transcript:
        raise PairExecutorError("SHARED_TRANSCRIPT")


def _validate_source(request: IsolatedPairRequest, source: SourceSnapshot) -> None:
    identity = request.control.identity
    if source != request.source:
        raise PairExecutorError("SOURCE_DRIFT")
    if (
        request.candidate.source_checkpoint_id != source.checkpoint_id
        or request.candidate.source_active_state_hash != source.canonical_sha256
        or identity.source_checkpoint_id != source.checkpoint_id
        or identity.source_checkpoint_hash != source.canonical_sha256
        or identity.noncandidate_memory_hash != source.noncandidate_memory_hash
        or identity.baseline_family != request.candidate.baseline_family
        or identity.rag_mode != request.candidate.rag_mode
        or identity.candidate_native_kind != request.candidate.candidate_native_kind
    ):
        raise PairExecutorError("CHECKPOINT_IDENTITY_MISMATCH")


def _validate_arm_result(result: PairArmResult, candidate: ChallengeCandidate) -> None:
    if result.updater_enabled or result.memory_write_event_id is not None:
        raise PairExecutorError("UPDATER_NOT_DISABLED")
    if result.candidate_exposure.candidate_entry_id != candidate.candidate_entry_id:
        raise PairExecutorError("CANDIDATE_EXPOSURE_BINDING_MISMATCH")


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
