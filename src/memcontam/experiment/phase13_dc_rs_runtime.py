from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from memcontam.baselines import dynamic_cheatsheet_phase12 as dc
from memcontam.clients.base import LLMClient
from memcontam.experiment.phase13_dc_rs_validation import (
    ORDINARY_TASKS,
    DcRsRuntimeError,
    OrdinaryHistoryIdentity,
    configured_budget,
    validate_ordinary_history,
    validate_state,
    validate_task,
)
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import CHECKPOINT_V3, NativeEntry, NativeState
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


@dataclass(frozen=True, slots=True)
class Phase13DcRsContext:
    task: TaskInstance
    client: LLMClient
    model: str
    verifier: Callable[[str, TaskInstance], Any]
    decoding: Mapping[str, Any]
    branch: Literal["clean", "correct", "irrelevant", "contam", "filter"]
    identities: Any
    embedding_provider: Any | None = None
    baseline_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    initial_states: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task.task_name not in ORDINARY_TASKS:
            raise DcRsRuntimeError("PROSPECTIVE_ORDINARY_TASK_REQUIRED")
        if not self.model or not self.identities.run_id or not self.identities.trial_id:
            raise DcRsRuntimeError("RUNTIME_IDENTITIES_REQUIRED")


@dataclass(frozen=True, slots=True)
class DcRsRuntimeExecution:
    result: dc.BaselineStepResultV3
    state: dc.DcRsStateV3


def initial_state(context: Any) -> dc.DcRsStateV3:
    state = context.initial_states.get(
        "dc_rs",
        dc.DcRsStateV3(archive=[], allow_unparented_strategies=True),
    )
    if not isinstance(state, dc.DcRsStateV3):
        raise DcRsRuntimeError("INVALID_DC_RS_STATE")
    state.allow_unparented_strategies = True
    validate_state(state, configured_budget(context), "INVALID_DC_RS_STATE")
    validate_ordinary_history(state, _ordinary_identity(context))
    return state


def execute(context: Any, state: Any) -> DcRsRuntimeExecution:
    if not isinstance(state, dc.DcRsStateV3):
        raise DcRsRuntimeError("INVALID_DC_RS_STATE")
    if context.task.task_name not in ORDINARY_TASKS:
        raise DcRsRuntimeError("PROSPECTIVE_ORDINARY_TASK_REQUIRED")
    validate_task(context.task, "INVALID_DC_RS_TASK")
    configured = dict(context.baseline_configs.get("dc_rs", {}))
    if context.branch == "filter":
        raise DcRsRuntimeError("DC_RS_FILTER_UNSUPPORTED")
    if configured.get("tool_mode", "text_only") != "text_only":
        raise DcRsRuntimeError("DC_RS_TEXT_ONLY_REQUIRED")
    budget = configured_budget(context)
    validate_state(state, budget, "INVALID_DC_RS_STATE")
    validate_ordinary_history(state, _ordinary_identity(context))
    if context.embedding_provider is None:
        raise DcRsRuntimeError("DC_RS_EMBEDDING_PROVIDER_REQUIRED")
    metadata = context.embedding_provider.metadata
    if (
        metadata.get("model_id") != BgeM3EmbeddingProvider.MODEL_ID
        or metadata.get("revision") != BgeM3EmbeddingProvider.REVISION
        or metadata.get("vector_dimension") != BgeM3EmbeddingProvider.VECTOR_DIMENSION
        or metadata.get("normalize_embeddings") is not True
    ):
        raise DcRsRuntimeError("DC_RS_BGE_M3_CONTRACT_REQUIRED")
    if not isinstance(context.embedding_provider, BgeM3EmbeddingProvider) and not (
        context.model == "replay"
        and configured.get("embedding_mode") == "test_double"
        and metadata.get("embedding_library_version") == "test"
    ):
        raise DcRsRuntimeError("DC_RS_BGE_M3_CONTRACT_REQUIRED")
    config = {
        **dict(context.decoding),
        **configured,
        "arm": context.branch,
        "baseline": "dc_rs",
        "model": context.model,
        "run_id": context.identities.run_id,
        "sample_id": context.task.sample_id,
        "tool_mode": "text_only",
    }
    adapter = dc.DcRsPhase12Adapter(
        embedding_provider=context.embedding_provider,
        cache_dir=config.get("cache_dir"),
    )
    result = adapter.execute(
        dc.DcRsTrialContextV3(
            task=context.task,
            client=context.client,
            model=context.model,
            run_id=context.identities.run_id,
            trial_id=context.identities.trial_id,
            condition_id=context.identities.condition_id or "dc_rs",
            branch=context.branch,
            config=config,
            order_key=context.identities.order_key,
            verifier=context.verifier,
        ),
        state,
    )
    return DcRsRuntimeExecution(result, state)


def _ordinary_identity(context: Any) -> OrdinaryHistoryIdentity:
    return OrdinaryHistoryIdentity(
        task_name=context.task.task_name,
        run_id=context.identities.run_id,
        trial_id=context.identities.trial_id,
        order_key=context.identities.order_key,
    )


def serialize(state: Any) -> NativeState:
    if (
        not isinstance(state, dc.DcRsStateV3)
        or state.filter_state is not None
        or state.allow_unparented_strategies is not True
    ):
        raise DcRsRuntimeError("DC_RS_STATE_SERIALIZATION_UNSUPPORTED")
    validate_state(state, None, "DC_RS_STATE_SERIALIZATION_UNSUPPORTED")
    archive = tuple(dc._archive_native(dc._archive_entry(entry)) for entry in state.archive)
    strategies = tuple(
        dc._strategy_entry(entry, allow_unparented=state.allow_unparented_strategies)
        for entry in state.strategies or ()
    )
    return NativeState(
        "dc_rs",
        (*archive, *strategies),
        {
            "archive": [dc._archive_entry(entry).model_dump(mode="json") for entry in state.archive],
            "injected_root_id": state.injected_root_id,
            "strategy_ids": [entry.entry_id for entry in strategies],
            "allow_unparented_strategies": state.allow_unparented_strategies,
        },
    )


def restore(snapshot: Any, context: Any) -> dc.DcRsStateV3:
    if (
        not isinstance(snapshot, NativeState)
        or snapshot.baseline != "dc_rs"
        or snapshot.schema_version != CHECKPOINT_V3
        or set(snapshot.native_state)
        != {
            "archive",
            "injected_root_id",
            "strategy_ids",
            "allow_unparented_strategies",
        }
    ):
        raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT")
    archive_rows = snapshot.native_state.get("archive")
    strategy_ids = snapshot.native_state.get("strategy_ids")
    allow_unparented = snapshot.native_state.get("allow_unparented_strategies")
    injected_root_id = snapshot.native_state.get("injected_root_id")
    if (
        not isinstance(archive_rows, list)
        or not isinstance(strategy_ids, list)
        or not all(isinstance(item, str) for item in strategy_ids)
        or allow_unparented is not True
        or not (injected_root_id is None or isinstance(injected_root_id, str))
        or not all(isinstance(entry, NativeEntry) for entry in snapshot.entries)
    ):
        raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT")
    try:
        archive: list[MemoryEntry] = [
            MemoryEntry.model_validate(row) for row in archive_rows
        ]
    except ValueError as error:
        raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT") from error
    archive_ids = [entry.entry_id for entry in archive]
    archive_entries = tuple(dc._archive_native(dc._archive_entry(entry)) for entry in archive)
    if tuple(snapshot.entries[: len(archive_entries)]) != archive_entries:
        raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT")
    strategies: list[NativeEntry] = []
    try:
        for raw_entry in snapshot.entries[len(archive_entries) :]:
            entry = raw_entry
            assert isinstance(entry, NativeEntry)
            if canonical_content_hash(entry.content) != entry.content_hash:
                raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT")
            if entry.native_component == "strategy":
                strategies.append(dc._strategy_entry(entry, allow_unparented=allow_unparented))
            else:
                raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT")
    except dc.DcRsContractError as error:
        raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT") from error
    if (
        [entry.entry_id for entry in strategies] != strategy_ids
        or any(
            not set(entry.direct_parent_ids).issubset(archive_ids)
            for entry in strategies
        )
        or (injected_root_id is not None and injected_root_id not in archive_ids)
    ):
        raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT")
    archive_state: list[MemoryEntry | NativeEntry] = list(archive)
    strategy_state: list[MemoryEntry | NativeEntry] = list(strategies)
    restored = dc.DcRsStateV3(
        archive=archive_state,
        strategies=strategy_state,
        injected_root_id=injected_root_id,
        allow_unparented_strategies=allow_unparented,
    )
    try:
        budget = configured_budget(context)
        validate_state(restored, budget, "INVALID_DC_RS_SNAPSHOT")
        validate_ordinary_history(restored, _ordinary_identity(context))
    except DcRsRuntimeError as error:
        raise DcRsRuntimeError("INVALID_DC_RS_SNAPSHOT") from error
    return restored
