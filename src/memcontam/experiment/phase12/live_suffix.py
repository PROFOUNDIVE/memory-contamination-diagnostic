from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import Arm, LiveBranchEvent, LiveThreeArmBranches
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY, RuntimeEntry, RuntimeTrialResult


_ARMS: tuple[Arm, ...] = ("clean", "contam", "filter")


class LiveSuffixError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class LiveSuffixTrial:
    baseline: str
    arm: Arm
    suffix_id: str
    order_key: int | str
    model: str
    decoding: Mapping[str, object]
    outcome: BaselineExecutionOutcome


@dataclass(frozen=True)
class LiveMemorySuffixRun:
    baseline: str
    horizon: int
    trials: tuple[LiveSuffixTrial, ...]
    events: tuple[LiveBranchEvent, ...]


@dataclass(frozen=True)
class LiveNoMemSuffixRun:
    horizon: int
    trials: tuple[LiveSuffixTrial, ...]
    display_aliases: tuple[Arm, ...]
    underlying_execution_count: int = 1


@dataclass(frozen=True)
class LiveMatchedSuffixResult:
    suffix_ids: tuple[str, ...]
    memory_runs: Mapping[str, LiveMemorySuffixRun]
    nomem: LiveNoMemSuffixRun


def run_live_matched_suffix(
    *,
    branches_by_baseline: Mapping[str, LiveThreeArmBranches],
    contexts: Sequence[Game24RuntimeContext],
    registry: Mapping[str, RuntimeEntry] = LIVE_BASELINE_REGISTRY,
) -> LiveMatchedSuffixResult:
    suffix_contexts = _validate_contexts(contexts)
    memory_runs = {
        baseline: _run_memory_suffix(branches, suffix_contexts, registry)
        for baseline, branches in branches_by_baseline.items()
    }
    if not memory_runs:
        raise LiveSuffixError("MEMORY_SUFFIX_REQUIRED")
    nomem = _run_nomem_suffix(suffix_contexts, registry)
    return LiveMatchedSuffixResult(
        tuple(context.task.sample_id for context in suffix_contexts), memory_runs, nomem
    )


def _validate_contexts(contexts: Sequence[Game24RuntimeContext]) -> tuple[Game24RuntimeContext, ...]:
    suffix_contexts = tuple(contexts)
    if not suffix_contexts or any(context.task.task_name != "game24" for context in suffix_contexts):
        raise LiveSuffixError("GAME24_SUFFIX_REQUIRED")
    if any(context.branch != "clean" for context in suffix_contexts):
        raise LiveSuffixError("CLEAN_SUFFIX_CONTEXT_REQUIRED")
    indices = tuple(context.identities.order_key for context in suffix_contexts)
    if (
        any(type(index) is not int or index < 1 for index in indices)
        or indices != tuple(sorted(indices))
        or len(set(indices)) != len(indices)
    ):
        raise LiveSuffixError("INVALID_SUFFIX_TASK_ORDER")
    if len({context.task.sample_id for context in suffix_contexts}) != len(suffix_contexts):
        raise LiveSuffixError("SUFFIX_TASK_DRIFT")
    if len({context.model for context in suffix_contexts}) != 1 or any(
        dict(context.decoding) != dict(suffix_contexts[0].decoding) for context in suffix_contexts
    ):
        raise LiveSuffixError("SUFFIX_RUNTIME_DRIFT")
    return suffix_contexts


def _run_memory_suffix(
    branches: LiveThreeArmBranches,
    contexts: tuple[Game24RuntimeContext, ...],
    registry: Mapping[str, RuntimeEntry],
) -> LiveMemorySuffixRun:
    entry = registry.get(branches.baseline)
    if entry is None or branches.baseline == "nomem":
        raise LiveSuffixError("MEMORY_RUNTIME_REQUIRED")
    if branches.model != contexts[0].model or dict(branches.decoding) != dict(contexts[0].decoding):
        raise LiveSuffixError("SUFFIX_RUNTIME_DRIFT")
    trials: list[LiveSuffixTrial] = []
    for arm in _ARMS:
        state = deepcopy(branches.arms[arm].state)
        for context in contexts:
            trial_context = _arm_context(context, arm, branches.baseline)
            result = entry.execute_trial(trial_context, state)
            if not isinstance(result, RuntimeTrialResult):
                raise LiveSuffixError("INVALID_LIVE_TRIAL_RESULT")
            _write_result(trial_context, result)
            trials.append(
                LiveSuffixTrial(
                    branches.baseline,
                    arm,
                    context.task.sample_id,
                    context.identities.order_key,
                    context.model,
                    dict(context.decoding),
                    result.outcome,
                )
            )
            state = result.state
    return LiveMemorySuffixRun(branches.baseline, len(contexts), tuple(trials), branches.events)


def _run_nomem_suffix(
    contexts: tuple[Game24RuntimeContext, ...], registry: Mapping[str, RuntimeEntry]
) -> LiveNoMemSuffixRun:
    entry = registry.get("nomem")
    if entry is None:
        raise LiveSuffixError("NOMEM_RUNTIME_REQUIRED")
    state = entry.initial_state(contexts[0].for_condition("nomem"))
    trials: list[LiveSuffixTrial] = []
    for context in contexts:
        trial_context = _arm_context(context, "clean", "nomem")
        result = entry.execute_trial(trial_context, state)
        if not isinstance(result, RuntimeTrialResult) or result.state is not state:
            raise LiveSuffixError("NOMEM_SINGLETON_REQUIRED")
        _write_result(trial_context, result)
        trials.append(
            LiveSuffixTrial(
                "nomem",
                "clean",
                context.task.sample_id,
                context.identities.order_key,
                context.model,
                dict(context.decoding),
                result.outcome,
            )
        )
    return LiveNoMemSuffixRun(len(contexts), tuple(trials), _ARMS)


def _arm_context(context: Game24RuntimeContext, arm: Arm, condition_id: str) -> Game24RuntimeContext:
    identities = RuntimeIdentities(
        context.identities.run_id,
        f"{context.identities.trial_id}:{condition_id}:{arm}",
        context.identities.order_key,
        condition_id,
    )
    return replace(context, branch=arm, identities=identities)


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


__all__ = [
    "LiveMatchedSuffixResult",
    "LiveMemorySuffixRun",
    "LiveNoMemSuffixRun",
    "LiveSuffixError",
    "LiveSuffixTrial",
    "run_live_matched_suffix",
]
