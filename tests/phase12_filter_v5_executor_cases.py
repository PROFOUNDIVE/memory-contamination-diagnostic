from __future__ import annotations

from dataclasses import replace

from memcontam.baselines.reflexion_phase12 import ReflexionTrialContextV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3, RagFrozenTrialContextV3
from memcontam.clients.replay import ReplayClient
from memcontam.experiment.phase12.filter_challenge.adapters.bot_style import (
    BoTChallengeExecution,
)
from memcontam.experiment.phase12.filter_challenge.adapters.full_history import (
    FullHistoryPairRequest,
)
from memcontam.experiment.phase12.filter_challenge.adapters.rag_frozen import (
    RagFrozenPairRequest,
)
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.experiment.phase12.filter_challenge.executor import (
    BoTExecutionRequest,
    ControlResultCache,
    FullHistoryExecutionRequest,
    RagFrozenExecutionRequest,
    ReflexionExecutionRequest,
)
from memcontam.experiment.phase12.filter_challenge.executor_source import source_snapshot
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    ExecutionOrder,
    ReplicateSeedContract,
)
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY, BranchIndex
from memcontam.rag.phase12_corpus import BranchCorpus, Document
from phase12_filter_v5_executor_support import (
    BotClient,
    Embedder,
    ExecutorCase,
    ScriptedClient,
    candidate,
    native_entry,
    pair_request,
    task,
)


def full_history_case(
    contract: ReplicateSeedContract = "deterministic",
    *,
    assessment_id: str = "assessment-1",
    order: ExecutionOrder = "control_first",
    cache: ControlResultCache | None = None,
) -> ExecutorCase:
    source = native_entry(
        "source", "full_history_transcript", "history", "source history"
    )
    checkpoint = serialize_checkpoint(NativeState("fh_bounded", (source,), {"records": []}))
    challenge_candidate = candidate(
        checkpoint, "full_history", "full_history_transcript", "candidate history"
    )
    control = ScriptedClient("control")
    challenge = ScriptedClient("challenge")
    native = FullHistoryExecutionRequest(
        "full_history",
        FullHistoryPairRequest(
            task(),
            checkpoint,
            challenge_candidate,
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
    request, sink = pair_request(
        native,
        challenge_candidate,
        contract,
        assessment_id=assessment_id,
        order=order,
        cache=cache,
    )
    return ExecutorCase(request, sink, control, challenge)


def rag_case() -> ExecutorCase:
    documents = tuple(Document(name, name) for name in ("source-a", "source-b", "source-c"))
    corpus = BranchCorpus(
        "clean", documents, tuple(item.document_id for item in documents), "corpus"
    )
    embedder = Embedder()
    index = BranchIndex(
        "clean",
        documents,
        {"production_identity": BGE_M3_PRIMARY_IDENTITY},
        {item.document_id: (0.0, 1.0) for item in documents},
        "index",
        embedder,
    )
    state = RagFrozenStateV3("clean", corpus, index)
    control = ScriptedClient("control")
    challenge = ScriptedClient("challenge")
    placeholder = ChallengeCandidate.model_validate(
        {
            "candidate_entry_id": "candidate",
            "candidate_native_content": "candidate",
            "candidate_native_kind": "rag_document",
            "baseline_family": "rag_frozen",
            "rag_mode": "frozen",
            "source_checkpoint_id": "rag-source",
            "source_active_state_hash": "pending",
            "routability": {
                "routability": "challenge_routable_v1",
                "challenge_suite_key": "suite-1",
            },
        }
    )
    control_trial = RagFrozenTrialContextV3(
        task(), control, "replay", "pair", "rag-control", "rag", "clean", "frozen"
    )
    challenge_trial = RagFrozenTrialContextV3(
        task(),
        challenge,
        "replay",
        "pair",
        "rag-challenge",
        "rag",
        "clean",
        "frozen",
        included_document_ids=("candidate", "source-a", "source-b"),
    )
    provisional = RagFrozenExecutionRequest(
        "rag_frozen",
        "rag-source",
        RagFrozenPairRequest(control_trial, challenge_trial, state, placeholder),
    )
    snapshot = source_snapshot(provisional)
    challenge_candidate = placeholder.model_copy(
        update={"source_active_state_hash": snapshot.canonical_sha256}
    )
    execution = replace(
        provisional,
        native_request=replace(
            provisional.native_request, candidate=challenge_candidate
        ),
    )
    request, sink = pair_request(execution, challenge_candidate)
    return ExecutorCase(request, sink, control, challenge)


def bot_case(
    contract: ReplicateSeedContract = "deterministic",
    *,
    order: ExecutionOrder = "control_first",
    cache: ControlResultCache | None = None,
    event_order: list[str] | None = None,
) -> ExecutorCase:
    source = native_entry(
        "template-z", "thought_template", "buffer", "Use source template."
    )
    checkpoint = serialize_checkpoint(
        NativeState(
            "bot_style",
            (source,),
            {"active_capacity": 2, "clean_competitor_ids": [], "templates": [source.entry_id]},
        )
    )
    challenge_candidate = candidate(
        checkpoint, "bot_style", "thought_template", "candidate template"
    )
    control = BotClient("control", event_order)
    challenge = BotClient("challenge", event_order)

    def execution(client: BotClient, arm: str) -> BoTChallengeExecution:
        return BoTChallengeExecution(
            checkpoint,
            task(),
            client,
            "replay",
            BotBufferIdentity(f"pair-{arm}", "game24", "bot_style", arm, "replay"),
            {"embedding_provider": Embedder(), "tool_mode": "text_only"},
            verifier=lambda _answer: True,
        )

    native = BoTExecutionRequest(
        "bot_style", execution(control, "clean"), execution(challenge, "contam")
    )
    request, sink = pair_request(
        native, challenge_candidate, contract, order=order, cache=cache
    )
    return ExecutorCase(request, sink, control, challenge)


def reflexion_case() -> ExecutorCase:
    entries = tuple(
        native_entry(name, "verbal_reflection", "reflections", f"reflection {name}")
        for name in ("one", "two", "three")
    )
    checkpoint = serialize_checkpoint(
        NativeState(
            "reflexion_style",
            entries,
            {
                "active_capacity": 4,
                "first_injected_eviction_trial_id": None,
                "injected_root_id": None,
                "reflections": [entry.entry_id for entry in entries],
            },
        )
    )
    challenge_candidate = candidate(
        checkpoint, "reflexion_style", "verbal_reflection", "candidate reflection"
    )
    control = ReplayClient(
        responses_by_sample={"sample-1": {"reflexion_generate": "final: 24"}}
    )
    challenge = ReplayClient(
        responses_by_sample={"sample-1": {"reflexion_generate": "final: 24"}}
    )

    def trial(client: ReplayClient, trial_id: str) -> ReflexionTrialContextV3:
        return ReflexionTrialContextV3(
            task(),
            client,
            "replay",
            "pair",
            trial_id,
            "reflexion",
            "contam",
            {},
            1,
            verifier=lambda answer, _task: answer == "24",
        )

    native = ReflexionExecutionRequest(
        "reflexion_style",
        checkpoint,
        trial(control, "reflexion-control"),
        trial(challenge, "reflexion-challenge"),
    )
    request, sink = pair_request(native, challenge_candidate)
    return ExecutorCase(request, sink, None, None)
