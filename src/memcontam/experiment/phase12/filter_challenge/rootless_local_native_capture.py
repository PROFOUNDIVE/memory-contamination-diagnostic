from __future__ import annotations

from functools import lru_cache
from typing import Literal, assert_never

from memcontam.baselines.reflexion_phase12 import (
    ReflexionPhase12Adapter,
    ReflexionTrialContextV3,
)
from memcontam.experiment.phase12.filter_challenge.adapters.bot_style import (
    BoTChallengeExecution,
    BoTStyleChallengeAdapter,
)
from memcontam.experiment.phase12.filter_challenge.adapters.reflexion_style import (
    ReflexionFrozenCheckpoint,
    ReflexionProvisionalAdapter,
    reflexion_source_state,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    Baseline,
    CandidateClass,
    ScheduledCall,
    Task,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    RootlessContractError,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_native_capture_support import (
    MODEL,
    CaptureClient,
    CaptureEmbedder,
    CapturedMessages,
    candidate_fixture,
    captured,
    checkpoint_fixture,
    task_fixture,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_native_storage_capture import (
    capture_storage_messages,
)
from memcontam.memory.bot_buffer import BotBufferIdentity


class NativePredecessorParseError(RootlessContractError):
    pass


def capture_native_messages(
    call: ScheduledCall, predecessor_text: str | None
) -> CapturedMessages:
    return _capture_native_messages(
        call.call_id,
        call.task,
        call.baseline,
        call.probe_id,
        call.side,
        call.candidate_class,
        call.replicate,
        call.native_stage,
        predecessor_text,
    )


@lru_cache(maxsize=None)
def _capture_native_messages(
    call_id: str,
    task: Task,
    baseline: Baseline,
    probe_id: str,
    side: Literal["control", "challenge"],
    candidate_class: CandidateClass | None,
    replicate: int | None,
    native_stage: Literal["answer", "bot_problem_distill", "bot_instantiate_solve"],
    predecessor_text: str | None,
) -> CapturedMessages:
    call = ScheduledCall(
        call_id=call_id,
        task=task,
        baseline=baseline,
        probe_id=probe_id,
        side=side,
        candidate_class=candidate_class,
        replicate=replicate,
        native_stage=native_stage,
    )
    match call.baseline:
        case "full_history" | "rag_frozen":
            return capture_storage_messages(call)
        case "bot_style":
            return _capture_bot(call, predecessor_text)
        case "reflexion_style":
            return _capture_reflexion(call)
        case unreachable:
            assert_never(unreachable)


def _capture_bot(
    call: ScheduledCall, predecessor_text: str | None
) -> CapturedMessages:
    task, answer = task_fixture(call)
    checkpoint = checkpoint_fixture(call)
    candidate = candidate_fixture(
        call, checkpoint.canonical_sha256, checkpoint.identity.checkpoint_id
    )
    client = CaptureClient(answer, predecessor_text)
    execution = BoTChallengeExecution(
        checkpoint,
        task,
        client,
        MODEL,
        BotBufferIdentity(
            f"rootless-{call.side}", call.task, "bot_style", call.side, MODEL
        ),
        {
            "embedding_provider": CaptureEmbedder(candidate.candidate_native_content),
            "tool_mode": "text_only",
        },
        verifier=lambda _answer: True,
    )
    adapter = BoTStyleChallengeAdapter()
    if call.side == "control":
        adapter.execute_control(execution)
    else:
        adapter.execute(execution, candidate)
    stage = (
        "bot_problem_distill"
        if call.native_stage == "bot_problem_distill"
        else "bot_instantiate_solve"
    )
    if stage == "bot_instantiate_solve" and predecessor_text is not None and not any(
        recorded_stage == stage for recorded_stage, _messages in client.calls
    ):
        raise NativePredecessorParseError(
            "DOWNSTREAM_NOT_ISSUED_AFTER_PARSE_FAILURE"
        )
    return captured(client, stage)


def _capture_reflexion(call: ScheduledCall) -> CapturedMessages:
    task, answer = task_fixture(call)
    checkpoint = checkpoint_fixture(call)
    candidate = candidate_fixture(
        call, checkpoint.canonical_sha256, checkpoint.identity.checkpoint_id
    )
    client = CaptureClient(answer)
    trial = ReflexionTrialContextV3(
        task,
        client,
        MODEL,
        f"rootless-{call.side}",
        f"rootless-{call.side}",
        "rootless",
        "contam",
        {},
        1,
        verifier=lambda _answer, _task: True,
    )
    if call.side == "control":
        ReflexionPhase12Adapter().execute(trial, reflexion_source_state(checkpoint))
    else:
        ReflexionProvisionalAdapter().execute(
            ReflexionFrozenCheckpoint(checkpoint, trial), candidate
        )
    return captured(client, "reflexion_generate")
