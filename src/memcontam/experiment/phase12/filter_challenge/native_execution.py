from __future__ import annotations

from dataclasses import replace
from typing import assert_never

from memcontam.baselines.reflexion_phase12 import ReflexionPhase12Adapter
from memcontam.experiment.phase12.filter_challenge.adapters.bot_style import (
    BoTStyleChallengeAdapter,
)
from memcontam.experiment.phase12.filter_challenge.adapters.reflexion_style import (
    ReflexionFrozenCheckpoint,
    ReflexionProvisionalAdapter,
    reflexion_source_state,
)
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    BoTControlValue,
    BoTExecutionRequest,
    ControlCacheValue,
    ExecutionOrder,
    FullHistoryExecutionRequest,
    NativeExecutionRequest,
    PairExecutorError,
    RagFrozenExecutionRequest,
    ReflexionControlValue,
    ReflexionExecutionRequest,
)
from memcontam.experiment.phase12.filter_challenge.provenance import (
    AnswerCallProvenanceObserver,
)
from memcontam.experiment.phase12.filter_challenge.native_normalization import (
    NativePairResult,
    answer_source_ids as _answer_source_ids,
    arm as _arm,
    bot_execution as _bot_execution,
    exposure as _exposure,
    relation as _relation,
    without_candidate as _without_candidate,
)
from memcontam.experiment.phase12.filter_challenge.native_storage_execution import (
    execute_storage_pair,
)


def execute_native_pair(
    execution: NativeExecutionRequest,
    candidate: ChallengeCandidate,
    cached: ControlCacheValue | None,
    order: ExecutionOrder,
) -> NativePairResult:
    match execution:
        case FullHistoryExecutionRequest() | RagFrozenExecutionRequest():
            return execute_storage_pair(execution, candidate, cached, order)
        case BoTExecutionRequest(control=control_execution, challenge=challenge_execution):
            control_observer = AnswerCallProvenanceObserver()
            challenge_observer = AnswerCallProvenanceObserver()
            control_execution = _bot_execution(control_execution, control_observer)
            challenge_execution = _bot_execution(challenge_execution, challenge_observer)
            match cached:
                case None:
                    if order == "control_first":
                        bot_control_native = BoTStyleChallengeAdapter().execute_control(
                            control_execution
                        )
                        bot_challenge_native = BoTStyleChallengeAdapter().execute(
                            challenge_execution, candidate
                        )
                    else:
                        bot_challenge_native = BoTStyleChallengeAdapter().execute(
                            challenge_execution, candidate
                        )
                        bot_control_native = BoTStyleChallengeAdapter().execute_control(
                            control_execution
                        )
                    bot_control_relation = _relation(
                        bot_control_native.outcome, control_observer
                    )
                    bot_control = _arm(
                        bot_control_native.outcome,
                        bot_control_relation,
                        _exposure(candidate, bot_control_native.final_context_source_ids),
                        (),
                    )
                case BoTControlValue(native=bot_control_native, normalized=bot_control):
                    pass
                case bot_mismatch:
                    if bot_mismatch is not None:
                        raise PairExecutorError("CONTROL_CACHE_FAMILY_MISMATCH")
                    assert_never(bot_mismatch)
            if cached is not None:
                bot_challenge_native = BoTStyleChallengeAdapter().execute(
                    challenge_execution, candidate
                )
            bot_challenge = _arm(
                bot_challenge_native.outcome,
                _relation(bot_challenge_native.outcome, challenge_observer),
                _exposure(candidate, bot_challenge_native.final_context_source_ids),
                _without_candidate(bot_challenge_native.displaced_template_ids, candidate),
            )
            bot_cache_value = BoTControlValue(
                "bot_style", bot_control_native, bot_control
            )
            return NativePairResult(
                type(BoTStyleChallengeAdapter()).__name__,
                bot_control,
                bot_challenge,
                bot_cache_value,
            )
        case ReflexionExecutionRequest(
            source_checkpoint=checkpoint,
            control_trial=control_trial,
            challenge_trial=challenge_trial,
        ):
            control_observer = AnswerCallProvenanceObserver()
            challenge_observer = AnswerCallProvenanceObserver()
            control_trial = replace(
                control_trial,
                run_id=f"{control_trial.run_id}:control",
                config={
                    **control_trial.config,
                    "update_enabled": False,
                    "_logging_answer_call_provenance_observer": control_observer,
                },
            )
            challenge_trial = replace(
                challenge_trial,
                run_id=f"{challenge_trial.run_id}:challenge",
                config={
                    **challenge_trial.config,
                    "update_enabled": False,
                    "_logging_answer_call_provenance_observer": challenge_observer,
                },
            )
            match cached:
                case None:
                    if order == "control_first":
                        reflexion_control_native = ReflexionPhase12Adapter().execute(
                            control_trial, reflexion_source_state(checkpoint)
                        )
                        reflexion_challenge_native = ReflexionProvisionalAdapter().execute(
                            ReflexionFrozenCheckpoint(checkpoint, challenge_trial), candidate
                        )
                    else:
                        reflexion_challenge_native = ReflexionProvisionalAdapter().execute(
                            ReflexionFrozenCheckpoint(checkpoint, challenge_trial), candidate
                        )
                        reflexion_control_native = ReflexionPhase12Adapter().execute(
                            control_trial, reflexion_source_state(checkpoint)
                        )
                    reflexion_control = _arm(
                        reflexion_control_native.outcome,
                        _relation(reflexion_control_native.outcome, control_observer),
                        _exposure(
                            candidate, _answer_source_ids(reflexion_control_native.outcome)
                        ),
                        (),
                    )
                case ReflexionControlValue(
                    native=reflexion_control_native, normalized=reflexion_control
                ):
                    reflexion_challenge_native = ReflexionProvisionalAdapter().execute(
                        ReflexionFrozenCheckpoint(checkpoint, challenge_trial), candidate
                    )
                case reflexion_mismatch:
                    if reflexion_mismatch is not None:
                        raise PairExecutorError("CONTROL_CACHE_FAMILY_MISMATCH")
                    assert_never(reflexion_mismatch)
            reflexion_challenge = _arm(
                reflexion_challenge_native.outcome,
                _relation(reflexion_challenge_native.outcome, challenge_observer),
                reflexion_challenge_native,
                _without_candidate(
                    reflexion_challenge_native.displaced_reflection_ids, candidate
                ),
            )
            reflexion_cache_value = ReflexionControlValue(
                "reflexion_style", reflexion_control_native, reflexion_control
            )
            return NativePairResult(
                type(ReflexionProvisionalAdapter()).__name__,
                reflexion_control,
                reflexion_challenge,
                reflexion_cache_value,
            )
        case unreachable:
            assert_never(unreachable)
