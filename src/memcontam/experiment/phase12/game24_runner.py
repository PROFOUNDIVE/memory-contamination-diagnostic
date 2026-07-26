from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping

from memcontam.clients.base import LLMClient
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY, RuntimeTrialResult
from memcontam.tasks.base import TaskInstance


Branch = Literal["clean", "correct", "irrelevant", "contam", "filter"]


@dataclass(frozen=True)
class RuntimeIdentities:
    run_id: str
    trial_id: str
    order_key: int | str
    condition_id: str = ""


@dataclass(frozen=True)
class RuntimeWriterCallbacks:
    on_outcome: Callable[[RuntimeTrialResult], None] | None = None
    on_retrieval: Callable[[object], None] | None = None
    on_context: Callable[[object], None] | None = None
    on_native_entry: Callable[[object], None] | None = None
    on_write_envelope: Callable[[object], None] | None = None


@dataclass(frozen=True)
class Game24RuntimeContext:
    task: TaskInstance
    client: LLMClient
    model: str
    verifier: Callable[[str, TaskInstance], Any]
    decoding: Mapping[str, Any]
    branch: Branch
    identities: RuntimeIdentities
    writer_callbacks: RuntimeWriterCallbacks = field(default_factory=RuntimeWriterCallbacks)
    embedding_provider: object | None = None
    baseline_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    initial_states: Mapping[str, object] = field(default_factory=dict)
    condition: object | None = None
    maturity_horizon: int = 1

    def __post_init__(self) -> None:
        if self.task.task_name != "game24":
            raise ValueError("GAME24_TASK_REQUIRED")
        if not self.model or not self.identities.run_id or not self.identities.trial_id:
            raise ValueError("RUNTIME_IDENTITIES_REQUIRED")

    def for_condition(self, condition_id: str) -> Game24RuntimeContext:
        return replace(self, identities=replace(self.identities, condition_id=condition_id))


def run_clean_game24_trial(
    context: Game24RuntimeContext,
) -> dict[str, RuntimeTrialResult]:
    if context.branch != "clean":
        raise ValueError("CLEAN_BRANCH_REQUIRED")
    results: dict[str, RuntimeTrialResult] = {}
    for baseline, entry in LIVE_BASELINE_REGISTRY.items():
        baseline_context = context.for_condition(baseline)
        result = entry.execute_trial(baseline_context, entry.initial_state(baseline_context))
        _write(baseline_context.writer_callbacks, result)
        results[baseline] = result
    return results


def _write(callbacks: RuntimeWriterCallbacks, result: RuntimeTrialResult) -> None:
    if callbacks.on_outcome is not None:
        callbacks.on_outcome(result)
    if result.retrieval_event is not None and callbacks.on_retrieval is not None:
        callbacks.on_retrieval(result.retrieval_event)
    if result.context_event is not None and callbacks.on_context is not None:
        callbacks.on_context(result.context_event)
    if callbacks.on_native_entry is not None:
        for entry in result.native_entries:
            callbacks.on_native_entry(entry)
    if callbacks.on_write_envelope is not None:
        for envelope in result.write_envelopes:
            callbacks.on_write_envelope(envelope)
