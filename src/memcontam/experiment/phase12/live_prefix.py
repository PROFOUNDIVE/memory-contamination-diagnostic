from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from memcontam.experiment.phase12.checkpoint_selection import (
    MEMORY_BASELINES,
    CommonCheckpointSelection,
    select_common_checkpoint,
)
from memcontam.experiment.phase12.contracts import BaselineConditionSpec
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY, RuntimeTrialResult
from memcontam.memory.checkpoint_v3 import NativeState, Phase12Checkpoint, serialize_checkpoint
from memcontam.tasks.base import TaskInstance


class LivePrefixError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class LiveCleanPrefixResult:
    seed: int
    checkpoints_by_baseline: dict[str, tuple[Phase12Checkpoint, ...]]
    trial_results_by_baseline: dict[str, tuple[RuntimeTrialResult, ...]]
    selection: CommonCheckpointSelection
    suffix_tasks: tuple[TaskInstance, ...]
    nomem_suffix_tasks: tuple[TaskInstance, ...]


def run_live_clean_prefix(
    *,
    seed: int,
    contexts: Sequence[Game24RuntimeContext],
    conditions: Mapping[str, BaselineConditionSpec],
    suffix_horizon: int,
) -> LiveCleanPrefixResult:
    _validate_contexts(seed, contexts, conditions, suffix_horizon)
    trial_indices = tuple(context.identities.order_key for context in contexts)
    typed_indices = tuple(index for index in trial_indices if type(index) is int)
    checkpoints_by_baseline: dict[str, tuple[Phase12Checkpoint, ...]] = {}
    trial_results_by_baseline: dict[str, tuple[RuntimeTrialResult, ...]] = {}

    for baseline in MEMORY_BASELINES:
        entry = LIVE_BASELINE_REGISTRY.get(baseline)
        if entry is None:
            raise LivePrefixError("LIVE_MEMORY_BASELINE_MISSING")
        condition = conditions[baseline]
        state = entry.initial_state(contexts[0].for_condition(condition.condition_id))
        checkpoints: list[Phase12Checkpoint] = []
        trial_results: list[RuntimeTrialResult] = []
        for context, index in zip(contexts, typed_indices, strict=True):
            trial_context = context.for_condition(condition.condition_id)
            result = entry.execute_trial(trial_context, state)
            if not isinstance(result, RuntimeTrialResult):
                raise LivePrefixError("INVALID_LIVE_TRIAL_RESULT")
            _write_result(trial_context, result)
            state = result.state
            checkpoints.append(serialize_checkpoint(_checkpoint_state(entry.serialize_state(state), index)))
            trial_results.append(result)
        checkpoints_by_baseline[baseline] = tuple(checkpoints)
        trial_results_by_baseline[baseline] = tuple(trial_results)

    selection = select_common_checkpoint(
        seed=seed,
        checkpoints_by_baseline=checkpoints_by_baseline,
        conditions=conditions,
        trial_indices=typed_indices,
        suffix_horizon=suffix_horizon,
    )
    contexts_by_index = {context.identities.order_key: context for context in contexts}
    suffix_tasks = tuple(
        contexts_by_index[index].task for index in selection.suffix_trial_indices
    )
    return LiveCleanPrefixResult(
        seed=seed,
        checkpoints_by_baseline=checkpoints_by_baseline,
        trial_results_by_baseline=trial_results_by_baseline,
        selection=selection,
        suffix_tasks=suffix_tasks,
        nomem_suffix_tasks=suffix_tasks,
    )


def _validate_contexts(
    seed: int,
    contexts: Sequence[Game24RuntimeContext],
    conditions: Mapping[str, BaselineConditionSpec],
    suffix_horizon: int,
) -> None:
    if type(seed) is not int:
        raise LivePrefixError("INVALID_CALIBRATION_SEED")
    if type(suffix_horizon) is not int or suffix_horizon < 1:
        raise LivePrefixError("INVALID_SUFFIX_HORIZON")
    if not contexts:
        raise LivePrefixError("PREFIX_CONTEXTS_REQUIRED")
    if any(context.branch != "clean" for context in contexts):
        raise LivePrefixError("CLEAN_PREFIX_REQUIRED")
    indices = tuple(context.identities.order_key for context in contexts)
    if (
        any(type(index) is not int or index < 1 for index in indices)
        or indices != tuple(sorted(indices))
        or len(set(indices)) != len(indices)
    ):
        raise LivePrefixError("INVALID_PREFIX_TRIAL_ORDER")
    if set(conditions) != set(MEMORY_BASELINES):
        raise LivePrefixError("PRIMARY_CONDITION_PANEL_REQUIRED")


def _checkpoint_state(snapshot: object, index: int) -> NativeState:
    if not isinstance(snapshot, NativeState):
        raise LivePrefixError("NATIVE_CHECKPOINT_REQUIRED")
    return NativeState(
        baseline=snapshot.baseline,
        entries=snapshot.entries,
        native_state={**snapshot.native_state, "checkpoint_index": index},
        schema_version=snapshot.schema_version,
    )


def _write_result(context: Game24RuntimeContext, result: RuntimeTrialResult) -> None:
    callbacks = context.writer_callbacks
    if callbacks.on_outcome is not None:
        callbacks.on_outcome(result)
    if result.retrieval_event is not None and callbacks.on_retrieval is not None:
        callbacks.on_retrieval(result.retrieval_event)
    if result.context_event is not None and callbacks.on_context is not None:
        callbacks.on_context(result.context_event)
    if callbacks.on_native_entry is not None:
        for native_entry in result.native_entries:
            callbacks.on_native_entry(native_entry)
    if callbacks.on_write_envelope is not None:
        for envelope in result.write_envelopes:
            callbacks.on_write_envelope(envelope)


__all__ = ["LiveCleanPrefixResult", "LivePrefixError", "run_live_clean_prefix"]
