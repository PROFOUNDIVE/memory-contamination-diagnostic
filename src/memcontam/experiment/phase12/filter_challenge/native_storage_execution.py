from __future__ import annotations

from dataclasses import replace
from typing import TypeAlias, assert_never

from memcontam.experiment.phase12.filter_challenge.adapters.full_history import (
    FullHistoryCachedControl,
    FullHistoryProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.adapters.rag_frozen import (
    RagFrozenCachedControl,
    RagFrozenProvisionalAdapter,
)
from memcontam.experiment.phase12.filter_challenge.contracts import ChallengeCandidate
from memcontam.experiment.phase12.filter_challenge.executor_types import (
    ControlCacheValue,
    ExecutionOrder,
    FullHistoryControlValue,
    FullHistoryExecutionRequest,
    PairExecutorError,
    RagFrozenControlValue,
    RagFrozenExecutionRequest,
)
from memcontam.experiment.phase12.filter_challenge.native_normalization import (
    NativePairResult,
    arm,
    exposure,
    without_candidate,
)

StorageExecution: TypeAlias = FullHistoryExecutionRequest | RagFrozenExecutionRequest


def execute_storage_pair(
    execution: StorageExecution,
    candidate: ChallengeCandidate,
    cached: ControlCacheValue | None,
    order: ExecutionOrder,
) -> NativePairResult:
    match execution:
        case FullHistoryExecutionRequest(native_request=native):
            if native.candidate != candidate:
                raise PairExecutorError("CANDIDATE_REQUEST_MISMATCH")
            match cached:
                case None:
                    fh_cached_control = None
                case FullHistoryControlValue(native=fh_value):
                    fh_cached_control = fh_value
                case fh_mismatch:
                    if fh_mismatch is not None:
                        raise PairExecutorError("CONTROL_CACHE_FAMILY_MISMATCH")
                    assert_never(fh_mismatch)
            fh_result = FullHistoryProvisionalAdapter().execute(
                replace(native, cached_control=fh_cached_control, execution_order=order)
            )
            fh_control = cached.normalized if cached is not None else arm(
                fh_result.control_outcome,
                fh_result.control_answer_relation,
                exposure(candidate, fh_result.control_final_source_ids),
                (),
            )
            fh_challenge = arm(
                fh_result.challenge_outcome,
                fh_result.challenge_answer_relation,
                fh_result.candidate_exposure,
                without_candidate(fh_result.challenge_removed_entry_ids, candidate),
            )
            fh_cache_value = FullHistoryControlValue(
                "full_history",
                FullHistoryCachedControl(
                    fh_result.control_outcome, fh_result.control_answer_relation
                ),
                fh_control,
            )
            return NativePairResult(
                type(FullHistoryProvisionalAdapter()).__name__,
                fh_control,
                fh_challenge,
                fh_cache_value,
            )
        case RagFrozenExecutionRequest(native_request=native):
            if native.candidate != candidate:
                raise PairExecutorError("CANDIDATE_REQUEST_MISMATCH")
            match cached:
                case None:
                    rag_cached_control = None
                case RagFrozenControlValue(native=rag_value):
                    rag_cached_control = rag_value
                case rag_mismatch:
                    if rag_mismatch is not None:
                        raise PairExecutorError("CONTROL_CACHE_FAMILY_MISMATCH")
                    assert_never(rag_mismatch)
            rag_result = RagFrozenProvisionalAdapter().execute(
                replace(native, cached_control=rag_cached_control, execution_order=order)
            )
            rag_control = cached.normalized if cached is not None else arm(
                rag_result.control.outcome,
                rag_result.control_answer_relation,
                exposure(candidate, rag_result.control_final_source_ids),
                (),
            )
            rag_challenge = arm(
                rag_result.challenge.outcome,
                rag_result.challenge_answer_relation,
                rag_result.candidate_exposure,
                without_candidate(
                    tuple(rag_result.challenge.context_event.removed_entry_ids), candidate
                ),
            )
            rag_cache_value = RagFrozenControlValue(
                "rag_frozen",
                RagFrozenCachedControl(
                    rag_result.control, rag_result.control_answer_relation
                ),
                rag_control,
            )
            return NativePairResult(
                type(RagFrozenProvisionalAdapter()).__name__,
                rag_control,
                rag_challenge,
                rag_cache_value,
            )
        case unreachable:
            assert_never(unreachable)
