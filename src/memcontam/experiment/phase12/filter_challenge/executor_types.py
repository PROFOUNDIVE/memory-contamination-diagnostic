from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, TypeAlias, assert_never

from memcontam.baselines.reflexion_phase12 import (
    BaselineStepResultV3 as ReflexionStepResult,
    ReflexionTrialContextV3,
)
from memcontam.clients.base import LLMClient
from memcontam.experiment.phase12.filter_challenge.adapters.bot_style import (
    BoTChallengeExecution,
    BoTChallengeResult,
)
from memcontam.experiment.phase12.filter_challenge.adapters.full_history import (
    FullHistoryCachedControl,
    FullHistoryPairRequest,
)
from memcontam.experiment.phase12.filter_challenge.adapters.rag_frozen import (
    RagFrozenCachedControl,
    RagFrozenPairRequest,
)
from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    CandidateExposureRecord,
    ChallengeCandidate,
    PairedExecutionIdentity,
)
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint

BaselineFamily: TypeAlias = Literal[
    "full_history", "rag_frozen", "bot_style", "reflexion_style"
]
ExecutionOrder: TypeAlias = Literal["control_first", "challenge_first"]
ReplicateSeedContract: TypeAlias = Literal[
    "deterministic", "seed_coupled", "counterbalanced"
]


class PairExecutorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PairingIdentity:
    source_checkpoint_id: str
    source_checkpoint_hash: str
    baseline_family: BaselineFamily
    rag_mode: Literal["frozen", "not_applicable"]
    candidate_native_kind: str
    probe_id: str
    prompt_payload_hash: str
    replicate_seed_contract: ReplicateSeedContract
    replicate_id: int
    paired_seed_replay_id: str
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
    prompt_payload_hash: str
    replicate_seed_contract: str
    replicate_id: int
    paired_seed_replay_id: str
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
class PairIsolation:
    control_session_id: str
    challenge_session_id: str
    control_transcript: tuple[str, ...]
    challenge_transcript: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeIdentityProjection:
    baseline_family: BaselineFamily
    control_model_snapshot: str
    challenge_model_snapshot: str


@dataclass(frozen=True, slots=True)
class FullHistoryExecutionRequest:
    family: Literal["full_history"]
    native_request: FullHistoryPairRequest


@dataclass(frozen=True, slots=True)
class RagFrozenExecutionRequest:
    family: Literal["rag_frozen"]
    source_checkpoint_id: str
    native_request: RagFrozenPairRequest


@dataclass(frozen=True, slots=True)
class BoTExecutionRequest:
    family: Literal["bot_style"]
    control: BoTChallengeExecution
    challenge: BoTChallengeExecution


@dataclass(frozen=True, slots=True)
class ReflexionExecutionRequest:
    family: Literal["reflexion_style"]
    source_checkpoint: Phase12Checkpoint
    control_trial: ReflexionTrialContextV3
    challenge_trial: ReflexionTrialContextV3


NativeExecutionRequest: TypeAlias = (
    FullHistoryExecutionRequest
    | RagFrozenExecutionRequest
    | BoTExecutionRequest
    | ReflexionExecutionRequest
)


@dataclass(frozen=True, slots=True)
class PairArmResult:
    answer_relation: AnswerCallRelation
    candidate_exposure: CandidateExposureRecord
    updater_enabled: bool
    memory_write_event_id: str | None
    displaced_noncandidate_entry_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FullHistoryControlValue:
    family: Literal["full_history"]
    native: FullHistoryCachedControl
    normalized: PairArmResult


@dataclass(frozen=True, slots=True)
class RagFrozenControlValue:
    family: Literal["rag_frozen"]
    native: RagFrozenCachedControl
    normalized: PairArmResult


@dataclass(frozen=True, slots=True)
class BoTControlValue:
    family: Literal["bot_style"]
    native: BoTChallengeResult
    normalized: PairArmResult


@dataclass(frozen=True, slots=True)
class ReflexionControlValue:
    family: Literal["reflexion_style"]
    native: ReflexionStepResult
    normalized: PairArmResult


ControlCacheValue: TypeAlias = (
    FullHistoryControlValue | RagFrozenControlValue | BoTControlValue | ReflexionControlValue
)


class ControlResultCache:
    def __init__(self) -> None:
        self._entries: dict[ControlCacheKey, ControlCacheValue] = {}

    def get(self, key: ControlCacheKey) -> ControlCacheValue | None:
        return self._entries.get(key)

    def put(self, key: ControlCacheKey, result: ControlCacheValue) -> None:
        self._entries[key] = result


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    checkpoint_id: str
    canonical_bytes: bytes
    canonical_sha256: str
    noncandidate_memory_bytes: bytes
    noncandidate_memory_hash: str


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
        payload["paired_execution_identity"] = self.paired_execution_identity.model_dump(mode="json")
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class OrdinaryTrialSink(Protocol):
    def append_trial(self, trial_id: str) -> None: ...


class OrdinaryCallSink(Protocol):
    def append_call(self, call_id: str) -> None: ...


class AssessmentSink(Protocol):
    def append_assessment(self, evidence: PairAuditEvidence) -> None: ...


@dataclass(frozen=True, slots=True)
class PairExecutionSinks:
    ordinary_trials: OrdinaryTrialSink
    ordinary_calls: OrdinaryCallSink
    assessments: AssessmentSink


@dataclass(frozen=True, slots=True)
class IsolatedPairRequest:
    assessment_id: str
    candidate: ChallengeCandidate
    candidate_version: str
    identity: PairingIdentity
    execution: NativeExecutionRequest
    isolation: PairIsolation
    execution_order: ExecutionOrder
    probe_configuration_hash: str
    sinks: PairExecutionSinks
    control_cache: ControlResultCache | None = None


def execution_clients(execution: NativeExecutionRequest) -> tuple[LLMClient, LLMClient]:
    match execution:
        case FullHistoryExecutionRequest(native_request=request):
            return request.control_client, request.challenge_client
        case RagFrozenExecutionRequest(native_request=request):
            return request.control_trial.client, request.challenge_trial.client
        case BoTExecutionRequest(control=control, challenge=challenge):
            return control.client, challenge.client
        case ReflexionExecutionRequest(control_trial=control, challenge_trial=challenge):
            return control.client, challenge.client
        case unreachable:
            assert_never(unreachable)


def runtime_identity_projection(execution: NativeExecutionRequest) -> RuntimeIdentityProjection:
    match execution:
        case FullHistoryExecutionRequest(native_request=request):
            return RuntimeIdentityProjection("full_history", request.model, request.model)
        case RagFrozenExecutionRequest(native_request=request):
            return RuntimeIdentityProjection(
                "rag_frozen", request.control_trial.model, request.challenge_trial.model
            )
        case BoTExecutionRequest(control=control, challenge=challenge):
            return RuntimeIdentityProjection("bot_style", control.model, challenge.model)
        case ReflexionExecutionRequest(control_trial=control, challenge_trial=challenge):
            return RuntimeIdentityProjection("reflexion_style", control.model, challenge.model)
        case unreachable:
            assert_never(unreachable)
