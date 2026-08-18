from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, assert_never

from memcontam.clients.base import LLMClient
from memcontam.experiment.phase12.game24_runner import (
    RuntimeIdentities,
    RuntimeWriterCallbacks,
)
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.experiment.phase12.runtime_registry import (
    PHASE13_CORE_BASELINE_REGISTRY,
    RuntimeTrialResult,
)
from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_core_datasets import (
    load_core_task,
    paired_trajectory_order,
    validate_core_datasets,
)
from memcontam.tasks.base import TaskInstance


OrdinaryTask: TypeAlias = Literal[
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
    "gpqa_diamond",
]
OrdinaryBaseline: TypeAlias = Literal[
    "fh_bounded",
    "rag_frozen",
    "bot_style",
    "reflexion_style",
    "dc_rs",
]
OrdinaryArm: TypeAlias = Literal["clean", "correct", "irrelevant", "contam"]
ORDINARY_TASKS: tuple[OrdinaryTask, ...] = (
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
    "gpqa_diamond",
)
ORDINARY_BASELINES: tuple[OrdinaryBaseline, ...] = (
    "fh_bounded",
    "rag_frozen",
    "bot_style",
    "reflexion_style",
    "dc_rs",
)
_CORE_TASKS = frozenset({"mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond"})


class ProspectiveOrdinaryError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ProspectiveOrdinaryContext:
    task: TaskInstance
    client: LLMClient
    model: str
    verifier: Callable[[str, TaskInstance], Any]
    decoding: Mapping[str, Any]
    identities: RuntimeIdentities
    branch: OrdinaryArm = "clean"
    writer_callbacks: RuntimeWriterCallbacks = field(default_factory=RuntimeWriterCallbacks)
    embedding_provider: Any | None = None
    baseline_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    initial_states: Mapping[str, Any] = field(default_factory=dict)
    condition: Any | None = None
    maturity_horizon: int = 1

    def for_condition(self, condition_id: str) -> ProspectiveOrdinaryContext:
        return ProspectiveOrdinaryContext(
            task=self.task,
            client=self.client,
            model=self.model,
            verifier=self.verifier,
            decoding=self.decoding,
            identities=RuntimeIdentities(
                self.identities.run_id,
                self.identities.trial_id,
                self.identities.order_key,
                condition_id,
            ),
            branch=self.branch,
            writer_callbacks=self.writer_callbacks,
            embedding_provider=self.embedding_provider,
            baseline_configs=self.baseline_configs,
            initial_states=self.initial_states,
            condition=self.condition,
            maturity_horizon=self.maturity_horizon,
        )


@dataclass(frozen=True, slots=True)
class ProspectiveOrdinaryRun:
    task_name: OrdinaryTask
    baseline: OrdinaryBaseline
    run_id: str
    model: str
    client: LLMClient
    verifier: Callable[[str, TaskInstance], Any]
    decoding: Mapping[str, Any]
    arm: OrdinaryArm = "clean"
    branch: LiveArmBranch | None = None
    tasks: tuple[TaskInstance, ...] = ()
    core_bundle: Path | None = None
    trajectory_seed: int | None = None
    writer_callbacks: RuntimeWriterCallbacks = field(default_factory=RuntimeWriterCallbacks)
    embedding_provider: Any | None = None
    baseline_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    initial_states: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_name not in ORDINARY_TASKS:
            raise ProspectiveOrdinaryError("ORDINARY_TASK_REQUIRED")
        if self.baseline not in ORDINARY_BASELINES:
            raise ProspectiveOrdinaryError("ORDINARY_BASELINE_REQUIRED")
        if not self.run_id or not self.model:
            raise ProspectiveOrdinaryError("ORDINARY_RUNTIME_IDENTITY_REQUIRED")
        if self.branch is None and self.arm != "clean":
            raise ProspectiveOrdinaryError("ORDINARY_BRANCH_REQUIRED")
        if self.branch is not None and (
            self.branch.arm != self.arm
            or self.branch.checkpoint.state.baseline != self.baseline
            or self.branch.root_count != (0 if self.arm == "clean" else 1)
        ):
            raise ProspectiveOrdinaryError("ORDINARY_BRANCH_IDENTITY_MISMATCH")
        is_core = self.task_name in _CORE_TASKS
        if is_core and (
            self.core_bundle is None
            or type(self.trajectory_seed) is not int
            or self.tasks
        ):
            raise ProspectiveOrdinaryError("CORE_TRAJECTORY_INPUT_REQUIRED")
        if not is_core and (
            not self.tasks or self.core_bundle is not None or self.trajectory_seed is not None
        ):
            raise ProspectiveOrdinaryError("NATIVE_TRAJECTORY_INPUT_REQUIRED")
        if self.tasks and (
            any(task.task_name != self.task_name for task in self.tasks)
            or len({task.sample_id for task in self.tasks}) != len(self.tasks)
        ):
            raise ProspectiveOrdinaryError("TASK_TRAJECTORY_NOT_ISOLATED")


@dataclass(frozen=True, slots=True)
class ProspectiveOrdinaryResult:
    task_name: OrdinaryTask
    baseline: OrdinaryBaseline
    arm: OrdinaryArm
    sample_ids: tuple[str, ...]
    trials: tuple[RuntimeTrialResult, ...]


def execute_prospective_ordinary(run: ProspectiveOrdinaryRun) -> ProspectiveOrdinaryResult:
    if run.task_name in _CORE_TASKS and run.baseline == "rag_frozen":
        raise ProspectiveOrdinaryError("CORE_RAG_SCIENTIFIC_PREREQUISITE_UNAVAILABLE")
    tasks = _ordered_tasks(run)
    entry = PHASE13_CORE_BASELINE_REGISTRY[run.baseline]
    contexts = tuple(_context(run, task, index) for index, task in enumerate(tasks, start=1))
    state = entry.initial_state(contexts[0]) if run.branch is None else deepcopy(run.branch.state)
    results: list[RuntimeTrialResult] = []
    for context in contexts:
        result = entry.execute_trial(context, state)
        _write(context.writer_callbacks, result)
        results.append(result)
        state = result.state
    return ProspectiveOrdinaryResult(
        run.task_name,
        run.baseline,
        run.arm,
        tuple(task.sample_id for task in tasks),
        tuple(results),
    )


def _ordered_tasks(run: ProspectiveOrdinaryRun) -> tuple[TaskInstance, ...]:
    if run.task_name not in _CORE_TASKS:
        return run.tasks
    assert run.core_bundle is not None
    assert run.trajectory_seed is not None
    validate_core_datasets(run.core_bundle, trajectory_seed=run.trajectory_seed)
    rows = load_core_task(run.core_bundle, _core_task(run.task_name))
    return paired_trajectory_order(rows, trajectory_seed=run.trajectory_seed)


def _core_task(task: OrdinaryTask) -> CoreTask:
    match task:
        case "mmlu_pro_engineering" | "mmlu_pro_physics" | "gpqa_diamond":
            return task
        case "game24" | "math_equation_balancer" | "word_sorting":
            raise ProspectiveOrdinaryError("CORE_TRAJECTORY_INPUT_REQUIRED")
        case unreachable:
            assert_never(unreachable)


def _context(
    run: ProspectiveOrdinaryRun,
    task: TaskInstance,
    order_key: int,
) -> ProspectiveOrdinaryContext:
    return ProspectiveOrdinaryContext(
        task=task,
        client=run.client,
        model=run.model,
        verifier=run.verifier,
        decoding=run.decoding,
        identities=RuntimeIdentities(
            run.run_id,
            (
                f"{run.run_id}:trial:{order_key}:{task.sample_id}"
                if run.arm == "clean"
                else f"{run.run_id}:{run.arm}:trial:{order_key}:{task.sample_id}"
            ),
            order_key,
            run.baseline if run.arm == "clean" else f"{run.baseline}:{run.arm}",
        ),
        branch=run.arm,
        writer_callbacks=run.writer_callbacks,
        embedding_provider=run.embedding_provider,
        baseline_configs=run.baseline_configs,
        initial_states=run.initial_states,
    )


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


__all__ = [
    "ORDINARY_BASELINES",
    "ORDINARY_TASKS",
    "OrdinaryArm",
    "ProspectiveOrdinaryContext",
    "ProspectiveOrdinaryError",
    "ProspectiveOrdinaryResult",
    "ProspectiveOrdinaryRun",
    "execute_prospective_ordinary",
]
