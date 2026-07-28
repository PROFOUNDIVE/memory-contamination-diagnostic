from __future__ import annotations

import json
from dataclasses import dataclass

from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.experiment.phase12.filter_challenge.executor import (
    ControlResultCache,
    IsolatedPairRequest,
    PairAuditEvidence,
    PairExecutionSinks,
    PairingIdentity,
    PairIsolation,
)
from memcontam.experiment.phase12.filter_challenge.executor_source import source_snapshot
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    ExecutionOrder,
    NativeExecutionRequest,
    ReplicateSeedContract,
)
from memcontam.experiment.phase12.filter_challenge.provenance import (
    AnswerCallProvenanceObserver,
)
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, Phase12Checkpoint
from memcontam.tasks.base import TaskInstance

class ScriptedClient:
    def __init__(self, label: str, order: list[str] | None = None) -> None:
        self.label = label
        self.order = order
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model, config
        self.calls += 1
        if self.order is not None:
            self.order.append(self.label)
        return LLMResponse(content="final: 24", raw={}, token_usage={}, latency_ms=0)


class Embedder:
    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_query(self, text: str) -> list[float]:
        assert text
        return [1.0, 0.0]


BotConfigValue = str | int | float | bool | None | Embedder | AnswerCallProvenanceObserver


class BotClient:
    def __init__(self, label: str, order: list[str] | None = None) -> None:
        self.label = label
        self.order = order
        self.calls = 0

    def chat(
        self, messages: list[dict[str, str]], model: str, config: dict[str, BotConfigValue]
    ) -> LLMResponse:
        del messages, model
        self.calls += 1
        if self.order is not None:
            self.order.append(self.label)
        stage = config["method_stage"]
        if stage == "bot_problem_distill":
            content = json.dumps(
                {
                    "key_information": "numbers = [1, 3, 4, 6], target = 24",
                    "restrictions": "Use every number exactly once.",
                    "distilled_task": "Construct an expression equal to 24.",
                }
            )
        else:
            content = json.dumps(
                {
                    "selected_structure": "retrieved-template",
                    "solution_trace": "Use the selected template.",
                    "final_answer": "final: 24",
                }
            )
        return LLMResponse(content=content, raw={}, token_usage={}, latency_ms=0)


class SinkSpy:
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
class ExecutorCase:
    request: IsolatedPairRequest
    sink: SinkSpy
    control_calls: ScriptedClient | BotClient | None
    challenge_calls: ScriptedClient | BotClient | None


def task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample-1", task_name="game24", input={"numbers": [1, 3, 4, 6]}
    )


def native_entry(entry_id: str, kind: str, component: str, content: str) -> NativeEntry:
    return NativeEntry(
        entry_id,
        kind,
        NATIVE_ENTRY_V1,
        component,
        content,
        canonical_content_hash(content),
    )


def candidate(
    checkpoint: Phase12Checkpoint, family: str, kind: str, content: str
) -> ChallengeCandidate:
    return ChallengeCandidate.model_validate(
        {
            "candidate_entry_id": "candidate",
            "candidate_native_content": content,
            "candidate_native_kind": kind,
            "baseline_family": family,
            "rag_mode": "frozen" if family == "rag_frozen" else "not_applicable",
            "source_checkpoint_id": checkpoint.identity.checkpoint_id,
            "source_active_state_hash": checkpoint.canonical_sha256,
            "routability": {
                "routability": "challenge_routable_v1",
                "challenge_suite_key": "suite-1",
            },
        }
    )


def pair_request(
    execution: NativeExecutionRequest,
    challenge_candidate: ChallengeCandidate,
    contract: ReplicateSeedContract = "deterministic",
    *,
    assessment_id: str = "assessment-1",
    order: ExecutionOrder = "control_first",
    replicate_id: int = 0,
    cache: ControlResultCache | None = None,
) -> tuple[IsolatedPairRequest, SinkSpy]:
    source = source_snapshot(execution)
    identity = PairingIdentity(
        source_checkpoint_id=source.checkpoint_id,
        source_checkpoint_hash=source.canonical_sha256,
        baseline_family=challenge_candidate.baseline_family,
        rag_mode=challenge_candidate.rag_mode,
        candidate_native_kind=challenge_candidate.candidate_native_kind,
        probe_id="probe-1",
        prompt_payload_hash="prompt-payload",
        replicate_seed_contract=contract,
        replicate_id=replicate_id,
        paired_seed_replay_id=f"seed-replay-{replicate_id}",
        model_snapshot="replay",
        decoding_contract_hash="decoding",
        fidelity_label="fidelity-1",
        tool_mode="text_only",
        tool_permissions_hash="permissions",
        raw_parser_version="parser-1",
        canonicalizer_version="canonicalizer-1",
        verifier_version="verifier-1",
        base_prompt_hash="base-prompt",
        formatter_version="formatter-1",
        context_budget_capacity_hash="context-capacity",
        retriever_index_capacity_hash="retriever-capacity",
        noncandidate_memory_hash=source.noncandidate_memory_hash,
        resource_retry_limit_hash="retry-resources",
    )
    sink = SinkSpy()
    return (
        IsolatedPairRequest(
            assessment_id,
            challenge_candidate,
            "native-v1",
            identity,
            execution,
            PairIsolation(
                "control-session",
                "challenge-session",
                ("control transcript",),
                ("challenge transcript",),
            ),
            order,
            "probe-config",
            PairExecutionSinks(sink, sink, sink),
            cache,
        ),
        sink,
    )
