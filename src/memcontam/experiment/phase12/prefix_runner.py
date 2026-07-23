from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from memcontam.experiment.phase12.checkpoint_store import (
    CheckpointStore,
    PrefixLawIdentity,
    PrefixReuseError,
    default_checkpoint_store,
)
from memcontam.experiment.phase12.contracts import PrefixExecutionKey, PrefixTemplateSpec
from memcontam.logging.schema_v3 import (
    AdmissionEvent,
    CheckpointEvent,
    PrefixExecutionKey as LogPrefixExecutionKey,
    PrefixTrialLog,
)
from memcontam.memory.admission import AdmissionContext, evaluate_admission
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, Phase12Checkpoint, serialize_checkpoint


__all__ = [
    "PrefixEventLedger",
    "PrefixExecutionError",
    "PrefixMemoryEvent",
    "PrefixMemoryWrite",
    "PrefixReuseError",
    "PrefixRunResult",
    "PrefixRunSpec",
    "PrefixStep",
    "PrefixTask",
    "run_clean_prefix",
]


class PrefixExecutionError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PrefixTask:
    absolute_trial_index: int
    task_id: str
    input_value: str
    resource_limits: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.absolute_trial_index < 1 or not self.task_id:
            raise PrefixExecutionError("INVALID_PREFIX_TASK")
        if any(not isinstance(value, int) or value < 0 for value in self.resource_limits.values()):
            raise PrefixExecutionError("INVALID_PREFIX_TASK")

    def to_identity_mapping(self) -> dict[str, Any]:
        return {
            "absolute_trial_index": self.absolute_trial_index,
            "input_value": self.input_value,
            "resource_limits": dict(self.resource_limits),
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class PrefixRunSpec:
    template: PrefixTemplateSpec
    tasks: tuple[PrefixTask, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.template.execution_key, PrefixExecutionKey):
            raise PrefixExecutionError("PREFIX_EXECUTION_KEY_REQUIRED")
        indices = tuple(task.absolute_trial_index for task in self.tasks)
        if not self.tasks or indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise PrefixExecutionError("INVALID_PREFIX_TASK_ORDER")


@dataclass(frozen=True)
class PrefixMemoryWrite:
    entry: NativeEntry
    envelope: MemoryCardEnvelopeV3

    def __post_init__(self) -> None:
        if (
            self.entry.entry_id != self.envelope.entry_id
            or self.entry.semantic_kind != self.envelope.semantic_kind
            or self.entry.native_component != self.envelope.native_component
            or self.entry.content != self.envelope.content
            or self.entry.content_hash != self.envelope.content_hash
        ):
            raise PrefixExecutionError("NATIVE_WRITE_MISMATCH")


@dataclass(frozen=True)
class PrefixStep:
    state: NativeState
    writes: tuple[PrefixMemoryWrite, ...] = ()
    eligible: bool = True


class PrefixPolicy(Protocol):
    def initial_state(self, spec: PrefixRunSpec, seed: int) -> NativeState: ...

    def execute(
        self, task: PrefixTask, state: NativeState, seed: int, trial_id: str
    ) -> PrefixStep: ...


@dataclass(frozen=True)
class PrefixMemoryEvent:
    event_id: str
    trial_id: str
    entry: NativeEntry
    writer_event_id: str


class PrefixEventLedger:
    def __init__(self, checkpoint_store: CheckpointStore | None = None) -> None:
        self.checkpoint_store = checkpoint_store or CheckpointStore()
        self.trials: list[PrefixTrialLog] = []
        self.memory_events: list[PrefixMemoryEvent] = []
        self.admission_events: list[AdmissionEvent] = []
        self.checkpoint_events: list[CheckpointEvent] = []

    def append_trial(self, trial: PrefixTrialLog) -> None:
        self.trials.append(trial)

    def append_memory_event(self, event: PrefixMemoryEvent) -> None:
        self.memory_events.append(event)

    def append_event(self, event: AdmissionEvent | CheckpointEvent) -> None:
        if isinstance(event, AdmissionEvent):
            self.admission_events.append(event)
        else:
            self.checkpoint_events.append(event)


@dataclass(frozen=True)
class PrefixRunResult:
    prefix_identity: PrefixLawIdentity
    execution_key: PrefixExecutionKey
    prefix_run_id: str
    checkpoint: Phase12Checkpoint
    checkpoints: tuple[Phase12Checkpoint, ...]
    trials: tuple[PrefixTrialLog, ...]
    memory_events: tuple[PrefixMemoryEvent, ...]
    admission_events: tuple[AdmissionEvent, ...]
    checkpoint_events: tuple[CheckpointEvent, ...]

    @property
    def is_suffix_aggregate_eligible(self) -> bool:
        return False

    def planning_summary(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint.identity.checkpoint_id,
            "checkpoint_sha256": self.checkpoint.identity.sha256,
            "evidence_layer": self.prefix_identity.evidence_layer,
            "prefix_identity": self.prefix_identity.to_mapping(),
            "prefix_run_id": self.prefix_run_id,
        }


def run_clean_prefix(
    spec: PrefixRunSpec,
    seed: int,
    policy: PrefixPolicy,
    writer: PrefixEventLedger | Any,
) -> PrefixRunResult:
    prefix_identity = PrefixLawIdentity.from_template(
        spec.template,
        seed=seed,
        task_sequence=tuple(task.to_identity_mapping() for task in spec.tasks),
    )
    store = _checkpoint_store(writer)
    existing = store.reuse(prefix_identity)
    if existing is not None:
        return existing

    prefix_run_id = f"prefix-{prefix_identity.fingerprint[:16]}"
    state = policy.initial_state(spec, seed)
    if not isinstance(state, NativeState):
        raise PrefixExecutionError("INVALID_INITIAL_NATIVE_STATE")

    known_envelopes: list[MemoryCardEnvelopeV3] = []
    trials: list[PrefixTrialLog] = []
    memory_events: list[PrefixMemoryEvent] = []
    admission_events: list[AdmissionEvent] = []
    checkpoint_events: list[CheckpointEvent] = []
    checkpoints: list[Phase12Checkpoint] = []

    for task in spec.tasks:
        trial_id = f"{prefix_run_id}:trial:{task.absolute_trial_index}:{task.task_id}"
        step = policy.execute(task, state, seed, trial_id)
        if not isinstance(step, PrefixStep):
            raise PrefixExecutionError("INVALID_PREFIX_STEP")
        _validate_native_update(state, step)
        if not step.eligible and step.state != state:
            raise PrefixExecutionError("INELIGIBLE_STATE_MUTATION")

        trial_memory_events, trial_admissions = _write_events(
            prefix_run_id=prefix_run_id,
            trial_id=trial_id,
            state=state,
            writes=step.writes,
            known_envelopes=known_envelopes,
        )
        checkpoint_event: CheckpointEvent | None = None
        if step.eligible:
            checkpoint = serialize_checkpoint(step.state)
            checkpoints.append(checkpoint)
            checkpoint_event = CheckpointEvent(
                record_type="checkpoint_event",
                event_id=f"{trial_id}:checkpoint",
                run_id=prefix_run_id,
                trial_id=trial_id,
                event_seq=0,
                checkpoint_id=checkpoint.identity.checkpoint_id,
                checkpoint_index=task.absolute_trial_index,
                memory_hash=checkpoint.identity.sha256,
            )

        trial = PrefixTrialLog(
            absolute_trial_index=task.absolute_trial_index,
            event_time=task.absolute_trial_index,
            parse_status="not_produced",
            execution_status="completed",
            failure_class=None,
            analysis_inclusion="excluded_prespecified",
            inclusion_reason="prefix_evidence_excluded",
            context_event_id_or_none=None,
            retrieval_event_ids=[],
            tool_event_ids=[],
            auxiliary_context_inclusion_or_none=None,
            operational_attribution_or_none=None,
            trial_kind="branch_free_prefix",
            execution_key=LogPrefixExecutionKey(kind="branch_free_prefix"),
            prefix_run_id=prefix_run_id,
            checkpoint_event_ids=[] if checkpoint_event is None else [checkpoint_event.event_id],
            admission_event_ids=[event.event_id for event in trial_admissions],
            memory_event_ids=[event.event_id for event in trial_memory_events],
        )
        _append_trial(writer, trial_id, trial)
        for event in trial_memory_events:
            _append_memory_event(writer, event)
        for event in trial_admissions:
            _append_event(writer, event)
        if checkpoint_event is not None:
            _append_event(writer, checkpoint_event)

        trials.append(trial)
        memory_events.extend(trial_memory_events)
        admission_events.extend(trial_admissions)
        if checkpoint_event is not None:
            checkpoint_events.append(checkpoint_event)
        state = step.state

    if not checkpoints:
        raise PrefixExecutionError("NO_ELIGIBLE_CHECKPOINT")
    result = PrefixRunResult(
        prefix_identity=prefix_identity,
        execution_key=spec.template.execution_key,
        prefix_run_id=prefix_run_id,
        checkpoint=checkpoints[-1],
        checkpoints=tuple(checkpoints),
        trials=tuple(trials),
        memory_events=tuple(memory_events),
        admission_events=tuple(admission_events),
        checkpoint_events=tuple(checkpoint_events),
    )
    store.save(result)
    return result


def _checkpoint_store(writer: PrefixEventLedger | Any) -> CheckpointStore:
    store = getattr(writer, "checkpoint_store", None)
    return store if isinstance(store, CheckpointStore) else default_checkpoint_store()


def _validate_native_update(state: NativeState, step: PrefixStep) -> None:
    if step.state.baseline != state.baseline:
        raise PrefixExecutionError("NATIVE_BASELINE_DRIFT")
    before_ids = _entry_ids(state.entries)
    after_ids = _entry_ids(step.state.entries)
    if after_ids[: len(before_ids)] != before_ids or len(set(after_ids)) != len(after_ids):
        raise PrefixExecutionError("NATIVE_UPDATE_ORDER_DRIFT")
    write_ids = tuple(write.entry.entry_id for write in step.writes)
    appended_ids = after_ids[len(before_ids) :]
    if write_ids and appended_ids != write_ids:
        raise PrefixExecutionError("NATIVE_UPDATE_ORDER_DRIFT")
    if any(write.envelope.baseline != state.baseline for write in step.writes):
        raise PrefixExecutionError("NATIVE_WRITE_MISMATCH")


def _write_events(
    *,
    prefix_run_id: str,
    trial_id: str,
    state: NativeState,
    writes: tuple[PrefixMemoryWrite, ...],
    known_envelopes: list[MemoryCardEnvelopeV3],
) -> tuple[list[PrefixMemoryEvent], list[AdmissionEvent]]:
    memory_events: list[PrefixMemoryEvent] = []
    admission_events: list[AdmissionEvent] = []
    known_entry_ids = set(_entry_ids(state.entries))
    for index, write in enumerate(writes, start=1):
        if write.entry.entry_id in known_entry_ids:
            raise PrefixExecutionError("DUPLICATE_NATIVE_WRITE")
        memory_event = PrefixMemoryEvent(
            event_id=f"{trial_id}:memory:{index}",
            trial_id=trial_id,
            entry=write.entry,
            writer_event_id=write.envelope.writer_event_id,
        )
        decision = evaluate_admission(
            write.envelope,
            AdmissionContext(
                writer_event_ids=frozenset({write.envelope.writer_event_id}),
                trial_record_ids=frozenset({trial_id}),
                evidence_envelopes=tuple(known_envelopes),
                active_envelopes=tuple(known_envelopes),
            ),
        )
        admission_event = AdmissionEvent(
            record_type="admission_event",
            event_id=f"{trial_id}:admission:{index}",
            run_id=prefix_run_id,
            trial_id=trial_id,
            event_seq=0,
            admission_id=memory_event.event_id,
            decision="admit" if decision.admitted else "quarantine",
        )
        if not decision.admitted:
            raise PrefixExecutionError(f"PREFIX_WRITE_REJECTED:{decision.reason}")
        memory_events.append(memory_event)
        admission_events.append(admission_event)
        known_envelopes.append(write.envelope)
        known_entry_ids.add(write.entry.entry_id)
    return memory_events, admission_events


def _append_trial(writer: PrefixEventLedger | Any, trial_id: str, trial: PrefixTrialLog) -> None:
    if isinstance(writer, PrefixEventLedger):
        writer.append_trial(trial)
    else:
        writer.append_trial(trial_id, trial)


def _append_memory_event(writer: PrefixEventLedger | Any, event: PrefixMemoryEvent) -> None:
    append = getattr(writer, "append_memory_event", None)
    if callable(append):
        append(event)


def _append_event(writer: PrefixEventLedger | Any, event: AdmissionEvent | CheckpointEvent) -> None:
    writer.append_event(event)


def _entry_ids(entries: tuple[str | NativeEntry, ...]) -> tuple[str, ...]:
    return tuple(entry.entry_id if isinstance(entry, NativeEntry) else entry for entry in entries)
