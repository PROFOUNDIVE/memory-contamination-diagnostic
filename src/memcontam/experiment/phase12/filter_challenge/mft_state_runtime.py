from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import memcontam.experiment.phase12.filter_challenge.executor_identity as executor_identity
import memcontam.experiment.phase12.filter_challenge.executor_source as executor_source
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.filter_challenge.adapters.full_history import (
    FullHistoryPairRequest,
)
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    FullHistoryExecutionRequest,
    IsolatedPairRequest,
    PairAuditEvidence,
    PairExecutionSinks,
    PairingIdentity,
    PairIsolation,
)
from memcontam.experiment.phase12.filter_challenge.executor_identity_types import (
    RuntimeIdentityProjection,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import (
    JsonValue,
)
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState, serialize_checkpoint
from memcontam.tasks.base import TaskInstance


class ScriptedCallFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _ClientAttempt:
    status: Literal["output", "failure"]


class _ScriptedClient:
    def __init__(self, attempts: tuple[_ClientAttempt, ...]) -> None:
        self.attempts = attempts
        self.records: list[Literal["output", "failure"]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model, config
        attempt = self.attempts[len(self.records)]
        self.records.append(attempt.status)
        if attempt.status == "failure":
            raise ScriptedCallFailure("SCRIPTED_CHALLENGE_FAILURE")
        return LLMResponse(content="final: 24", raw={}, token_usage={}, latency_ms=0)


class _SinkSpy:
    def __init__(self) -> None:
        self.trials: list[str] = []
        self.calls: list[str] = []
        self.assessments: list[PairAuditEvidence] = []

    def append_trial(self, trial_id: str) -> None:
        self.trials.append(trial_id)

    def append_call(self, call_id: str) -> None:
        self.calls.append(call_id)

    def append_assessment(self, evidence: PairAuditEvidence) -> None:
        self.assessments.append(evidence)


@dataclass(frozen=True, slots=True)
class _RuntimeCase:
    request: IsolatedPairRequest
    expected_identity: PairingIdentity
    sink: _SinkSpy
    control_client: _ScriptedClient
    challenge_client: _ScriptedClient


def runtime_case(
    *, mutate_identity: bool = False, include_failure: bool = False
) -> _RuntimeCase:
    source_entry = NativeEntry(
        "synthetic-build-source",
        "full_history_transcript",
        NATIVE_ENTRY_V1,
        "history",
        "source history",
        canonical_content_hash("source history"),
    )
    checkpoint = serialize_checkpoint(NativeState("fh_bounded", (source_entry,), {"records": []}))
    candidate = ChallengeCandidate.model_validate(
        {
            "candidate_entry_id": "synthetic-build-candidate",
            "candidate_native_content": "candidate history",
            "candidate_native_kind": "full_history_transcript",
            "baseline_family": "full_history",
            "rag_mode": "not_applicable",
            "source_checkpoint_id": checkpoint.identity.checkpoint_id,
            "source_active_state_hash": checkpoint.canonical_sha256,
            "routability": {
                "routability": "challenge_routable_v1",
                "challenge_suite_key": "synthetic-build-suite-key",
            },
        }
    )
    control = _ScriptedClient((_ClientAttempt("output"),))
    challenge_attempts: tuple[_ClientAttempt, ...] = (_ClientAttempt("output"),)
    if include_failure:
        challenge_attempts += (_ClientAttempt("failure"),)
    challenge = _ScriptedClient(challenge_attempts)
    execution = FullHistoryExecutionRequest(
        "full_history",
        FullHistoryPairRequest(
            _task(),
            checkpoint,
            candidate,
            control,
            challenge,
            "replay",
            {
                "mode": "context_bounded_pair_atomic",
                "token_encoding": "cl100k_base",
                "context_window_tokens": 100_000,
                "max_output_tokens": 1,
                "fixed_prompt_overhead_tokens": 0,
                "safety_margin_tokens": 0,
            },
        ),
    )
    source = executor_source.source_snapshot(execution)
    projection, _ = executor_identity.runtime_identity_projections(execution)
    identity = _pairing_identity(source.canonical_sha256, projection)
    expected_identity = identity
    if mutate_identity:
        identity = replace(identity, model_snapshot="mutated-model")
    sink = _SinkSpy()
    request = IsolatedPairRequest(
        "synthetic-build-assessment",
        candidate,
        "native-v1",
        identity,
        execution,
        PairIsolation("synthetic-control-session", "synthetic-challenge-session", (), ()),
        "control_first",
        "synthetic-build-probe-config",
        PairExecutionSinks(sink, sink, sink),
    )
    return _RuntimeCase(request, expected_identity, sink, control, challenge)


def _pairing_identity(
    source_hash: str, projection: RuntimeIdentityProjection
) -> PairingIdentity:
    return PairingIdentity(
        source_checkpoint_id=projection.source_checkpoint_id,
        source_checkpoint_hash=source_hash,
        baseline_family="full_history",
        rag_mode="not_applicable",
        candidate_native_kind="full_history_transcript",
        probe_id="synthetic-build-game24-probe-1",
        prompt_payload_hash=projection.prompt_payload_hash,
        replicate_seed_contract="deterministic",
        replicate_id=0,
        paired_seed_replay_id="synthetic-build-seed-replay-0",
        model_snapshot=projection.model_snapshot,
        decoding_contract_hash=projection.decoding_contract_hash,
        fidelity_label="synthetic-build-fidelity",
        tool_mode=projection.tool_mode or "text_only",
        tool_permissions_hash=projection.tool_permissions_hash or "synthetic-permissions",
        raw_parser_version="synthetic-build-parser-v1",
        canonicalizer_version="synthetic-build-canonicalizer-v1",
        verifier_version=projection.verifier_version,
        base_prompt_hash="synthetic-build-base-prompt",
        formatter_version="synthetic-build-formatter-v1",
        context_budget_capacity_hash=projection.context_budget_capacity_hash,
        retriever_index_capacity_hash=(projection.retriever_index_capacity_hash or "synthetic-retriever"),
        noncandidate_memory_hash=source_hash,
        resource_retry_limit_hash=(projection.resource_retry_limit_hash or "synthetic-retry"),
    )


def arm_config(identity: PairingIdentity, *, candidate_entry_id: str | None) -> dict[str, JsonValue]:
    return {
        "candidate_entry_id": candidate_entry_id,
        "pairing_identity": asdict(identity),
        "updater_enabled": False,
    }


def config_diff(
    control: dict[str, JsonValue], challenge: dict[str, JsonValue]
) -> tuple[str, ...]:
    diff = tuple(key for key in control if control[key] != challenge[key] and key != "pairing_identity")
    control_identity, challenge_identity = control["pairing_identity"], challenge["pairing_identity"]
    if isinstance(control_identity, dict) and isinstance(challenge_identity, dict):
        diff += tuple(
            f"pairing_identity.{key}"
            for key in control_identity
            if control_identity[key] != challenge_identity[key]
        )
    return diff


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="synthetic-build-sample",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
    )
