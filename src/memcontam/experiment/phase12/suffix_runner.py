from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

from memcontam.baselines.execution import execute_baseline
from memcontam.experiment.phase12.branching import BranchSet, MaterializedBranch, NoMemAliasRecord
from memcontam.experiment.phase12.contracts import (
    MemoryArmExecutionKey,
    NoMemExecutionKey,
    RunTemplateSpec,
)
from memcontam.logging.schema_v3 import (
    CheckpointEvent,
    InterventionEvent,
    MemoryArmExecutionKey as LogMemoryArmExecutionKey,
    MemoryBranchTrialLog,
    NoMemExecutionKey as LogNoMemExecutionKey,
    NoMemTrialLog,
)
from memcontam.memory.checkpoint_v3 import (
    NativeEntry,
    NativeState,
    Phase12Checkpoint,
    serialize_checkpoint,
)
from memcontam.tasks.base import TaskInstance


__all__ = [
    "AliasRecord",
    "NoMemSuffixRunResult",
    "SuffixEventLedger",
    "SuffixExecutionError",
    "SuffixPolicy",
    "SuffixRunResult",
    "SuffixRunSet",
    "SuffixStep",
    "SuffixWriterFactory",
    "WriterFactory",
    "materialize_nomem_aliases",
    "run_matched_suffix",
]


Arm = Literal["clean", "correct", "irrelevant", "contam", "filter"]
_ARMS: tuple[Arm, ...] = ("clean", "correct", "irrelevant", "contam", "filter")


class SuffixExecutionError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class SuffixStep:
    state: NativeState


class SuffixPolicy(Protocol):
    def execute(
        self, task: TaskInstance, state: NativeState, seed: int, trial_id: str
    ) -> SuffixStep: ...


class WriterFactory(Protocol):
    def create(self, arm: str) -> tuple[SuffixPolicy, Any]: ...


class SuffixEventLedger:
    def __init__(self) -> None:
        self.trials: list[MemoryBranchTrialLog | NoMemTrialLog] = []
        self.events: list[CheckpointEvent | InterventionEvent] = []

    def append_trial(self, trial: MemoryBranchTrialLog | NoMemTrialLog) -> None:
        self.trials.append(trial)

    def append_event(self, event: CheckpointEvent | InterventionEvent) -> None:
        self.events.append(event)


class SuffixWriterFactory:
    def __init__(
        self,
        policies: Mapping[str, SuffixPolicy],
        writers: Mapping[str, SuffixEventLedger | Any] | None = None,
    ) -> None:
        self._policies = dict(policies)
        self._writers = dict(writers or {})
        self._nomem_executed = False

    def create(self, arm: str) -> tuple[SuffixPolicy, SuffixEventLedger | Any]:
        try:
            policy = self._policies[arm]
        except KeyError as error:
            raise SuffixExecutionError("MISSING_SUFFIX_POLICY") from error
        return policy, self._writers.setdefault(arm, SuffixEventLedger())

    def claim_nomem(self) -> bool:
        if self._nomem_executed:
            return False
        self._nomem_executed = True
        return True


@dataclass(frozen=True)
class SuffixRunResult:
    pair_id: str
    arm: Arm
    execution_key: MemoryArmExecutionKey
    prefix_run_id: str
    checkpoint_id: str
    checkpoint_index: int
    trials: tuple[MemoryBranchTrialLog, ...]
    checkpoints: tuple[Phase12Checkpoint, ...]


@dataclass(frozen=True)
class NoMemSuffixRunResult:
    pair_id: str
    execution_key: NoMemExecutionKey
    trials: tuple[NoMemTrialLog, ...]
    underlying_execution_count: Literal[1] = 1


@dataclass(frozen=True)
class AliasRecord:
    display_arm: Arm
    execution_key: NoMemExecutionKey
    source_pair_id: str


@dataclass(frozen=True)
class SuffixRunSet:
    pair_id: str
    runs: tuple[SuffixRunResult, ...] = ()
    nomem: NoMemSuffixRunResult | None = None
    aliases: tuple[AliasRecord, ...] = ()


def run_matched_suffix(
    branches: BranchSet | NoMemAliasRecord,
    suffix: Sequence[TaskInstance] | Mapping[str, Sequence[TaskInstance]],
    spec: RunTemplateSpec,
    writer_factory: WriterFactory,
    *,
    seed: int = 0,
) -> SuffixRunSet:
    tasks = _matched_tasks(suffix)
    if isinstance(branches, NoMemAliasRecord):
        return _run_nomem_suffix(branches, tasks, spec, writer_factory, seed)
    if not isinstance(branches, BranchSet) or not isinstance(
        spec.execution_key, MemoryArmExecutionKey
    ):
        raise SuffixExecutionError("MEMORY_ARM_EXECUTION_KEY_REQUIRED")
    if spec.prefix_template_key_or_none is None:
        raise SuffixExecutionError("PREFIX_CHECKPOINT_REQUIRED")

    _validate_filter_visibility(branches)
    pair_id = f"pair:{branches.source_checkpoint.identity.checkpoint_id}:{spec.run_template_id}"
    prefix_run_id = f"prefix:{branches.source_checkpoint.identity.checkpoint_id}"
    checkpoint_index = _checkpoint_index(tasks)
    branch_by_arm = {
        "clean": branches.clean,
        "correct": branches.correct,
        "irrelevant": branches.irrelevant,
        "contam": branches.contam,
    }
    interventions = {intervention.arm: intervention for intervention in branches.interventions}
    runs = tuple(
        _run_memory_arm(
            arm,
            branch_by_arm.get(arm),
            branches.filter.active if arm == "filter" else None,
            tasks,
            spec,
            writer_factory,
            seed,
            pair_id,
            prefix_run_id,
            checkpoint_index,
            branches.source_checkpoint.identity.checkpoint_id,
            None if arm == "clean" else interventions.get(arm),
        )
        for arm in _ARMS
    )
    return SuffixRunSet(pair_id=pair_id, runs=runs)


def materialize_nomem_aliases(result: SuffixRunSet) -> tuple[AliasRecord, ...]:
    if result.nomem is None or result.runs or result.nomem.underlying_execution_count != 1:
        raise SuffixExecutionError("INVALID_NOMEM_ALIAS_RESULT")
    return tuple(AliasRecord(arm, result.nomem.execution_key, result.pair_id) for arm in _ARMS)


def _run_memory_arm(
    arm: Arm,
    branch: MaterializedBranch | None,
    filter_checkpoint: Phase12Checkpoint | None,
    tasks: tuple[TaskInstance, ...],
    spec: RunTemplateSpec,
    writer_factory: WriterFactory,
    seed: int,
    pair_id: str,
    prefix_run_id: str,
    checkpoint_index: int,
    source_checkpoint_id: str,
    intervention: Any,
) -> SuffixRunResult:
    checkpoint = filter_checkpoint if arm == "filter" else branch.checkpoint if branch else None
    if checkpoint is None:
        raise SuffixExecutionError("MISSING_BRANCH_CHECKPOINT")
    policy, writer = _create(writer_factory, arm)
    run_id = _run_id(writer, spec, arm)
    state = checkpoint.state
    trials: list[MemoryBranchTrialLog] = []
    checkpoints: list[Phase12Checkpoint] = []
    intervention_id = None if arm == "clean" else f"{pair_id}:intervention:{arm}"

    for position, task in enumerate(tasks, start=1):
        absolute_trial_index, event_time = _task_timing(task)
        trial_id = f"{run_id}:trial:{absolute_trial_index}:{task.sample_id}"
        step = execute_baseline(policy, task, state, seed, trial_id)
        if not isinstance(step, SuffixStep):
            raise SuffixExecutionError("INVALID_SUFFIX_STEP")
        _validate_native_update(state, step.state)
        if state.baseline == "rag_frozen" and step.state != state:
            raise SuffixExecutionError("RAG_SUFFIX_WRITE_FORBIDDEN")
        candidate_id = None if intervention is None else intervention.candidate_triplet_id
        render_id = None if intervention is None else intervention.native_render_id
        trial = MemoryBranchTrialLog(
            absolute_trial_index=absolute_trial_index,
            event_time=event_time,
            parse_status="not_produced",
            execution_status="completed",
            failure_class=None,
            analysis_inclusion="included",
            inclusion_reason="suffix_execution",
            context_event_id_or_none=None,
            retrieval_event_ids=[],
            tool_event_ids=[],
            auxiliary_context_inclusion_or_none=None,
            operational_attribution_or_none=None,
            trial_kind="memory_branch",
            execution_key=LogMemoryArmExecutionKey(kind="memory_arm", arm=arm),
            branch_id=arm,
            prefix_run_id=prefix_run_id,
            checkpoint_id=source_checkpoint_id,
            checkpoint_index=checkpoint_index,
            candidate_triplet_id_or_none=candidate_id,
            native_render_id_or_none=render_id,
            intervention_event_id_or_none=intervention_id,
            admission_event_ids=[],
            memory_event_ids=[],
        )
        _append_trial(writer, trial_id, trial)
        if position == 1 and intervention_id is not None:
            _append_event(
                writer,
                InterventionEvent(
                    record_type="intervention_event",
                    event_id=intervention_id,
                    run_id=run_id,
                    trial_id=trial_id,
                    event_seq=0,
                    intervention_id=intervention_id,
                    arm=arm,
                    candidate_triplet_id=candidate_id or "",
                    native_render_id=render_id or "",
                ),
            )
        state = step.state
        persisted = serialize_checkpoint(state)
        _append_event(
            writer,
            CheckpointEvent(
                record_type="checkpoint_event",
                event_id=f"{trial_id}:checkpoint",
                run_id=run_id,
                trial_id=trial_id,
                event_seq=0,
                checkpoint_id=persisted.identity.checkpoint_id,
                checkpoint_index=absolute_trial_index,
                memory_hash=persisted.identity.sha256,
            ),
        )
        trials.append(trial)
        checkpoints.append(persisted)

    return SuffixRunResult(
        pair_id=pair_id,
        arm=arm,
        execution_key=MemoryArmExecutionKey(kind="memory_arm", arm=arm),
        prefix_run_id=prefix_run_id,
        checkpoint_id=source_checkpoint_id,
        checkpoint_index=checkpoint_index,
        trials=tuple(trials),
        checkpoints=tuple(checkpoints),
    )


def _run_nomem_suffix(
    aliases: NoMemAliasRecord,
    tasks: tuple[TaskInstance, ...],
    spec: RunTemplateSpec,
    writer_factory: WriterFactory,
    seed: int,
) -> SuffixRunSet:
    if (
        not isinstance(spec.execution_key, NoMemExecutionKey)
        or spec.prefix_template_key_or_none is not None
    ):
        raise SuffixExecutionError("NOMEM_ARM_FORBIDDEN")
    if aliases.underlying_execution_count != 1 or aliases.display_alias_count != 5:
        raise SuffixExecutionError("INVALID_NOMEM_ALIAS_RESULT")
    if not _claim_nomem(writer_factory):
        raise SuffixExecutionError("DUPLICATE_NOMEM_EXECUTION")
    policy, writer = _create(writer_factory, "nomem")
    run_id = _run_id(writer, spec, "nomem")
    state = NativeState("no_memory", (), {})
    trials: list[NoMemTrialLog] = []
    for task in tasks:
        absolute_trial_index, event_time = _task_timing(task)
        trial_id = f"{run_id}:trial:{absolute_trial_index}:{task.sample_id}"
        step = execute_baseline(policy, task, state, seed, trial_id)
        if not isinstance(step, SuffixStep) or step.state != state:
            raise SuffixExecutionError("NOMEM_STATE_MUTATION")
        trial = NoMemTrialLog(
            absolute_trial_index=absolute_trial_index,
            event_time=event_time,
            parse_status="not_produced",
            execution_status="completed",
            failure_class=None,
            analysis_inclusion="included",
            inclusion_reason="suffix_execution",
            context_event_id_or_none=None,
            retrieval_event_ids=[],
            tool_event_ids=[],
            auxiliary_context_inclusion_or_none=None,
            operational_attribution_or_none=None,
            trial_kind="nomem_singleton",
            execution_key=LogNoMemExecutionKey(kind="nomem_singleton", key="*"),
        )
        _append_trial(writer, trial_id, trial)
        trials.append(trial)
    result = NoMemSuffixRunResult(
        pair_id=f"pair:nomem:{spec.run_template_id}",
        execution_key=spec.execution_key,
        trials=tuple(trials),
    )
    run_set = SuffixRunSet(pair_id=result.pair_id, nomem=result)
    return SuffixRunSet(
        pair_id=run_set.pair_id,
        nomem=result,
        aliases=materialize_nomem_aliases(run_set),
    )


def _matched_tasks(
    suffix: Sequence[TaskInstance] | Mapping[str, Sequence[TaskInstance]],
) -> tuple[TaskInstance, ...]:
    if isinstance(suffix, Mapping):
        if set(suffix) != set(_ARMS):
            raise SuffixExecutionError("SUFFIX_TASK_DRIFT")
        sequences = tuple(tuple(suffix[arm]) for arm in _ARMS)
        tasks = sequences[0]
        if any(_task_signature(sequence) != _task_signature(tasks) for sequence in sequences[1:]):
            raise SuffixExecutionError("SUFFIX_TASK_DRIFT")
    else:
        tasks = tuple(suffix)
    if not tasks or not all(isinstance(task, TaskInstance) for task in tasks):
        raise SuffixExecutionError("INVALID_SUFFIX_TASK")
    indices = tuple(_task_timing(task)[0] for task in tasks)
    if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
        raise SuffixExecutionError("INVALID_SUFFIX_TASK_ORDER")
    return tasks


def _task_signature(tasks: Sequence[TaskInstance]) -> tuple[dict[str, Any], ...]:
    return tuple(task.model_dump(mode="json") for task in tasks)


def _task_timing(task: TaskInstance) -> tuple[int, int | str]:
    absolute_trial_index = task.metadata.get("absolute_trial_index")
    event_time = task.metadata.get("event_time", absolute_trial_index)
    if type(absolute_trial_index) is not int or absolute_trial_index < 1:
        raise SuffixExecutionError("INVALID_SUFFIX_TASK")
    if not isinstance(event_time, (int, str)) or event_time == "":
        raise SuffixExecutionError("INVALID_SUFFIX_TASK")
    return absolute_trial_index, event_time


def _checkpoint_index(tasks: Sequence[TaskInstance]) -> int:
    first_index, _ = _task_timing(tasks[0])
    return first_index - 1


def _validate_filter_visibility(branches: BranchSet) -> None:
    active_ids = set(_entry_ids(branches.filter.active.state.entries))
    quarantine_ids = set(_entry_ids(branches.filter.quarantine.state.entries))
    if active_ids & quarantine_ids:
        raise SuffixExecutionError("QUARANTINE_EXPOSURE")


def _validate_native_update(previous: NativeState, current: NativeState) -> None:
    if current.baseline != previous.baseline:
        raise SuffixExecutionError("NATIVE_BASELINE_DRIFT")
    before_ids = _entry_ids(previous.entries)
    after_ids = _entry_ids(current.entries)
    if after_ids[: len(before_ids)] != before_ids or len(set(after_ids)) != len(after_ids):
        raise SuffixExecutionError("NATIVE_UPDATE_ORDER_DRIFT")


def _entry_ids(entries: Sequence[str | NativeEntry]) -> tuple[str, ...]:
    return tuple(entry.entry_id if isinstance(entry, NativeEntry) else entry for entry in entries)


def _create(factory: WriterFactory, arm: str) -> tuple[SuffixPolicy, Any]:
    try:
        policy, writer = factory.create(arm)
    except AttributeError:
        raise SuffixExecutionError("INVALID_WRITER_FACTORY")

    return policy, writer


def _claim_nomem(factory: WriterFactory) -> bool:
    claim = getattr(factory, "claim_nomem", None)
    if not callable(claim):
        raise SuffixExecutionError("INVALID_WRITER_FACTORY")
    return bool(claim())


def _run_id(writer: Any, spec: RunTemplateSpec, arm: str) -> str:
    run_dir = getattr(writer, "run_dir", None)
    name = getattr(run_dir, "name", None)
    return name if isinstance(name, str) and name else f"{spec.run_template_id}:{arm}"


def _append_trial(writer: Any, trial_id: str, trial: MemoryBranchTrialLog | NoMemTrialLog) -> None:
    if isinstance(writer, SuffixEventLedger):
        writer.append_trial(trial)
    else:
        writer.append_trial(trial_id, trial)


def _append_event(writer: Any, event: CheckpointEvent | InterventionEvent) -> None:
    writer.append_event(event)
