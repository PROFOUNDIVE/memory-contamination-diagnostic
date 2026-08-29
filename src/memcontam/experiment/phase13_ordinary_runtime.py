from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, assert_never

from memcontam.clients.base import LLMClient
from memcontam.clients.provider_profile import model_client_binding_error, request_binding_error
from memcontam.experiment.phase12.game24_runner import RuntimeIdentities, RuntimeWriterCallbacks
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.experiment.phase12.runtime_registry import (
    PHASE13_CORE_BASELINE_REGISTRY,
    RuntimeTrialResult,
)
from memcontam.readiness.phase13_route_capacity import bind_capacity_configs, capacity_contract_error
from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_cost_activation import (
    Phase13CostActivationError,
    validate_activated_cost_policy,
)
from memcontam.readiness.phase13_cost_policy import bind_cost_policy_client
from memcontam.readiness import phase13_capacity_realization as capacity_realization
from memcontam.readiness.phase13_core_datasets import (
    load_core_task,
    paired_trajectory_order,
    validate_core_datasets,
)
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.tasks.base import TaskInstance

_validated_common_capacity_tokens = capacity_realization.validated_common_capacity_tokens

OrdinaryTask: TypeAlias = Literal[
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
]
OrdinaryBaseline: TypeAlias = Literal[
    "fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"
]
ProspectiveBaseline: TypeAlias = OrdinaryBaseline | Literal["nomem"]
OrdinaryArm: TypeAlias = Literal["clean", "correct", "irrelevant", "contam"]
ORDINARY_TASKS: tuple[OrdinaryTask, ...] = (
    "game24",
    "math_equation_balancer",
    "word_sorting",
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
)
ORDINARY_BASELINES: tuple[OrdinaryBaseline, ...] = (
    "fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"
)
PROSPECTIVE_BASELINES: tuple[ProspectiveBaseline, ...] = ("nomem", *ORDINARY_BASELINES)
CORE_TASKS = frozenset({"mmlu_pro_engineering", "mmlu_pro_physics"})
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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
    baseline: ProspectiveBaseline
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
    allow_test_client: bool = False
    production_identity: ProductionOrdinaryRunIdentity | None = None

    def __post_init__(self) -> None:
        if self.task_name not in ORDINARY_TASKS:
            raise ProspectiveOrdinaryError("ORDINARY_TASK_REQUIRED")
        if self.baseline not in PROSPECTIVE_BASELINES:
            raise ProspectiveOrdinaryError("ORDINARY_BASELINE_REQUIRED")
        if not self.run_id or not self.model:
            raise ProspectiveOrdinaryError("ORDINARY_RUNTIME_IDENTITY_REQUIRED")
        binding_error = model_client_binding_error(
            self.model,
            self.client,
            self.allow_test_client,
        ) or request_binding_error(
            self.decoding.get("service_tier", "default"),
            self.decoding.get("max_output_tokens", 512),
        )
        if binding_error is not None:
            raise ProspectiveOrdinaryError(binding_error)
        if self.model == "gpt-5.6-luna":
            try:
                validate_activated_cost_policy(REPOSITORY_ROOT)
            except Phase13CostActivationError as error:
                raise ProspectiveOrdinaryError(error.code) from error
        contract_error = capacity_contract_error(
            self.baseline_configs,
            _validated_common_capacity_tokens(),
        )
        if contract_error is not None:
            raise ProspectiveOrdinaryError(contract_error)
        if self.branch is None and self.arm != "clean":
            raise ProspectiveOrdinaryError("ORDINARY_BRANCH_REQUIRED")
        if self.baseline == "nomem" and (self.arm != "clean" or self.branch is not None):
            raise ProspectiveOrdinaryError("NOMEM_SINGLETON_CLEAN_REQUIRED")
        if self.branch is not None and (
            self.branch.arm != self.arm
            or self.branch.checkpoint.state.baseline != self.baseline
            or self.branch.root_count != (0 if self.arm == "clean" else 1)
        ):
            raise ProspectiveOrdinaryError("ORDINARY_BRANCH_IDENTITY_MISMATCH")
        is_core = self.task_name in CORE_TASKS
        if is_core and (
            self.core_bundle is None
            or type(self.trajectory_seed) is not int
            or self.tasks
        ):
            raise ProspectiveOrdinaryError("CORE_TRAJECTORY_INPUT_REQUIRED")
        if not is_core and (not self.tasks or self.core_bundle is not None):
            raise ProspectiveOrdinaryError("NATIVE_TRAJECTORY_INPUT_REQUIRED")
        if self.tasks and (
            any(task.task_name != self.task_name for task in self.tasks)
            or len({task.sample_id for task in self.tasks}) != len(self.tasks)
        ):
            raise ProspectiveOrdinaryError("TASK_TRAJECTORY_NOT_ISOLATED")


@dataclass(frozen=True, slots=True)
class ProspectiveOrdinaryResult:
    task_name: OrdinaryTask
    baseline: ProspectiveBaseline
    arm: OrdinaryArm
    sample_ids: tuple[str, ...]
    trials: tuple[RuntimeTrialResult, ...]


def execute_prospective_ordinary(run: ProspectiveOrdinaryRun) -> ProspectiveOrdinaryResult:
    if (run.task_name, run.baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells:
        raise ProspectiveOrdinaryError("EXCLUDED_CURRENT_MAIN_PROSPECTIVE_RAG_EXTENSION")
    tasks = _ordered_tasks(run)
    _validate_live_dispatch_identity(run, tasks)
    entry = PHASE13_CORE_BASELINE_REGISTRY[run.baseline]
    contexts = tuple(_context(run, task, index) for index, task in enumerate(tasks, start=1))
    state = entry.initial_state(contexts[0]) if run.branch is None else deepcopy(run.branch.state)
    results: list[RuntimeTrialResult] = []
    for context in contexts:
        result = entry.execute_trial(context, state)
        _write(context.writer_callbacks, result)
        results.append(result)
        state = result.state
        if result.outcome.status == "failed":
            break
    return ProspectiveOrdinaryResult(
        run.task_name,
        run.baseline,
        run.arm,
        tuple(task.sample_id for task in tasks),
        tuple(results),
    )


def execute_readiness0_trial(run: ProspectiveOrdinaryRun) -> ProspectiveOrdinaryResult:
    if (run.task_name, run.baseline) in CORE_MAIN_REGISTRY.current_main_excluded_cells:
        raise ProspectiveOrdinaryError("EXCLUDED_CURRENT_MAIN_PROSPECTIVE_RAG_EXTENSION")
    tasks = _ordered_tasks(run)
    _validate_live_dispatch_identity(run, tasks)
    context = _context(run, tasks[0], 1)
    entry = PHASE13_CORE_BASELINE_REGISTRY[run.baseline]
    state = entry.initial_state(context) if run.branch is None else deepcopy(run.branch.state)
    result = entry.execute_trial(context, state)
    _write(context.writer_callbacks, result)
    return ProspectiveOrdinaryResult(
        run.task_name, run.baseline, run.arm, tuple(task.sample_id for task in tasks), (result,)
    )


def _validate_live_dispatch_identity(
    run: ProspectiveOrdinaryRun,
    tasks: tuple[TaskInstance, ...],
) -> None:
    from memcontam.clients.openai_responses import OpenAIResponsesClient

    if run.model != "gpt-5.6-luna" or not isinstance(run.client, OpenAIResponsesClient):
        return
    identity = run.production_identity
    if identity is None or run.trajectory_seed != identity.trajectory_seed:
        raise ProspectiveOrdinaryError("PRODUCTION_TRAJECTORY_SEED_MISMATCH")
    from memcontam.readiness.phase13_main_checkpoint import (
        Phase13MainCheckpointError,
        production_identity_from_checkpoint,
    )

    try:
        expected = production_identity_from_checkpoint(
            REPOSITORY_ROOT / "data/phase13/main/mr_p4",
            REPOSITORY_ROOT,
            task=run.task_name,
            trajectory_seed=identity.trajectory_seed,
            execution_template_id=identity.execution_template_id,
            registration_packet_sha256=identity.registration_packet_sha256,
        )
    except Phase13MainCheckpointError as error:
        raise ProspectiveOrdinaryError(error.code) from error
    if (
        identity.checkpoint_registry_sha256 != expected.checkpoint_registry_sha256
        or identity.ordered_sample_ids_sha256 != expected.ordered_sample_ids_sha256
    ):
        raise ProspectiveOrdinaryError("PRODUCTION_CHECKPOINT_IDENTITY_MISMATCH")
    ordered_sample_ids_sha256 = hashlib.sha256(
        json.dumps(tuple(task.sample_id for task in tasks), separators=(",", ":")).encode()
    ).hexdigest()
    if identity.ordered_sample_ids_sha256 != ordered_sample_ids_sha256:
        raise ProspectiveOrdinaryError("PRODUCTION_SAMPLE_ORDER_MISMATCH")


def _ordered_tasks(run: ProspectiveOrdinaryRun) -> tuple[TaskInstance, ...]:
    if run.task_name not in CORE_TASKS:
        return run.tasks
    assert run.core_bundle is not None
    assert run.trajectory_seed is not None
    validate_core_datasets(run.core_bundle, trajectory_seed=run.trajectory_seed)
    rows = load_core_task(run.core_bundle, _core_task(run.task_name))
    ordered = paired_trajectory_order(rows, trajectory_seed=run.trajectory_seed)
    if run.model != "gpt-5.6-luna":
        return ordered
    if run.production_identity is None:
        raise ProspectiveOrdinaryError("PRODUCTION_TRAJECTORY_SEED_MISMATCH")
    for start in range(len(ordered) - 49):
        suffix = ordered[start : start + 50]
        digest = hashlib.sha256(
            json.dumps(tuple(row.sample_id for row in suffix), separators=(",", ":")).encode()
        ).hexdigest()
        if digest == run.production_identity.ordered_sample_ids_sha256:
            return suffix
    raise ProspectiveOrdinaryError("PRODUCTION_SAMPLE_ORDER_MISMATCH")


def _core_task(task: OrdinaryTask) -> CoreTask:
    match task:
        case "mmlu_pro_engineering" | "mmlu_pro_physics":
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
        client=bind_cost_policy_client(run.client, REPOSITORY_ROOT),
        model=run.model,
        verifier=run.verifier,
        decoding={**run.decoding, "max_output_tokens": 512, "service_tier": "default"},
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
        baseline_configs=bind_capacity_configs(
            run.baseline_configs,
            _validated_common_capacity_tokens(),
        ),
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
    "execute_readiness0_trial",
    "execute_prospective_ordinary",
]
