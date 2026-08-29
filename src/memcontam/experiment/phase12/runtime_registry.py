from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, TypeVar

from memcontam.baselines.bot_phase12 import BoTPhase12Adapter, BoTStateV3, BoTTrialContextV3
from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.full_history_phase12 import (
    FullHistoryPhase12Adapter,
    FullHistoryStateV3,
    TrialContextV3,
)
from memcontam.baselines.no_memory import NoMemoryAdapter
from memcontam.baselines.reflexion_phase12 import (
    ReflexionPhase12Adapter,
    ReflexionStateV3,
    ReflexionTrialContextV3,
)
from memcontam.baselines.retrieval_rag_phase12 import (
    RagFrozenPhase12Adapter,
    RagFrozenStateV3,
    RagFrozenTrialContextV3,
)
from memcontam.experiment.phase12.maturity import evaluate_maturity
from memcontam.experiment import phase13_dc_rs_runtime as dc_runtime
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.readiness.phase13_cost_policy import (
    Phase13CostPolicyError,
    Phase13ProviderCallError,
)
from memcontam.clients.recording import MethodCallRecorder


class RuntimeStateError(ValueError):
    pass


@dataclass(frozen=True)
class NoMemRuntimeState:
    pass


NOMEM_SINGLETON = NoMemRuntimeState()
T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeTrialResult:
    outcome: BaselineExecutionOutcome
    state: object
    retrieval_event: object | None = None
    context_event: object | None = None
    native_entries: tuple[NativeEntry, ...] = ()
    write_envelopes: tuple[MemoryCardEnvelopeV3, ...] = ()


@dataclass(frozen=True)
class RuntimeEntry:
    initial_state: Callable[[Any], object]
    execute_trial: Callable[[Any, object], RuntimeTrialResult]
    serialize_state: Callable[[object], object]
    restore_state: Callable[[object, Any], object]
    maturity_view: Callable[[object, Any], object | None]


def _initial_state(baseline: str, context: Any, expected: type[T], default: T | None = None) -> T:
    state = context.initial_states.get(baseline)
    if state is None:
        state = default
    if not isinstance(state, expected):
        raise RuntimeStateError(f"MISSING_{baseline.upper()}_STATE")
    return state


def _nomem_initial_state(context: Any) -> NoMemRuntimeState:
    del context
    return NOMEM_SINGLETON


def _fh_initial_state(context: Any) -> FullHistoryStateV3:
    return _initial_state("fh_bounded", context, FullHistoryStateV3, FullHistoryStateV3(records=[]))


def _rag_initial_state(context: Any) -> RagFrozenStateV3:
    return _initial_state("rag_frozen", context, RagFrozenStateV3)


def _bot_initial_state(context: Any) -> BoTStateV3:
    return _initial_state("bot_style", context, BoTStateV3)


def _reflexion_initial_state(context: Any) -> ReflexionStateV3:
    return _initial_state("reflexion_style", context, ReflexionStateV3, ReflexionStateV3(reflections=[]))


def _config(context: Any, baseline: str) -> dict[str, Any]:
    return {
        **dict(context.decoding),
        **dict(context.baseline_configs.get(baseline, {})),
        "arm": context.branch,
        "baseline": baseline,
        "model": context.model,
        "run_id": context.identities.run_id,
        "sample_id": context.task.sample_id,
    }


def _provider_failure(
    error: Exception, method_calls: tuple[object, ...] = ()
) -> BaselineExecutionOutcome:
    return BaselineExecutionOutcome(
        status="failed",
        error_type="ProviderCallFailure",
        failure_disposition="provider_call_failed",
        scientific_ineligibility_reason="provider_call_failed",
        method_calls=method_calls,
        metadata={
            "exception_type": type(error).__name__,
            "message": str(error),
            "provider_attempts_count": getattr(error, "provider_attempts_count", 1),
            "provider_latency_ms": getattr(error, "provider_latency_ms", 0),
            "provider_status": getattr(error, "provider_status", None),
            "provider_incomplete_reason": getattr(error, "provider_incomplete_reason", None),
            "provider_usage": getattr(error, "provider_usage", None),
            "provider_token_usage": getattr(error, "provider_token_usage", None),
            "provider_cost_usd": getattr(error, "provider_cost_usd", None),
            "provider_response_id": getattr(error, "provider_response_id", None),
        },
    )


def _nomem_execute(context: Any, state: object) -> RuntimeTrialResult:
    if state is not NOMEM_SINGLETON:
        raise RuntimeStateError("NOMEM_SINGLETON_REQUIRED")
    outcome = NoMemoryAdapter().execute(
        context.task,
        MemoryState(),
        client=context.client,
        model=context.model,
        config=_config(context, "nomem"),
        verifier=context.verifier,
    )
    return RuntimeTrialResult(outcome, NOMEM_SINGLETON)


def _fh_execute(context: Any, state: object) -> RuntimeTrialResult:
    if not isinstance(state, FullHistoryStateV3):
        raise RuntimeStateError("INVALID_FH_BOUNDED_STATE")
    config = _config(context, "fh_bounded")
    result = FullHistoryPhase12Adapter().execute(
        TrialContextV3(
            task=context.task,
            client=context.client,
            model=context.model,
            trial_id=context.identities.trial_id,
            condition_id=context.identities.condition_id,
            fh_mode=config.get("fh_mode", "bounded"),
            context_config=config,
            context_budget_id=config.get("context_budget_id", "fh-bounded"),
            order_key=context.identities.order_key,
            verifier=context.verifier,
        ),
        state,
    )
    return RuntimeTrialResult(
        result.outcome,
        state,
        native_entries=()
        if result.write_envelope is None
        else (_entry_from_envelope(result.write_envelope),),
        write_envelopes=() if result.write_envelope is None else (result.write_envelope,),
    )


def _rag_execute(context: Any, state: object) -> RuntimeTrialResult:
    if not isinstance(state, RagFrozenStateV3):
        raise RuntimeStateError("INVALID_RAG_FROZEN_STATE")
    result = RagFrozenPhase12Adapter().execute(
        RagFrozenTrialContextV3(
            task=context.task,
            client=context.client,
            model=context.model,
            run_id=context.identities.run_id,
            trial_id=context.identities.trial_id,
            condition_id=context.identities.condition_id,
            branch=context.branch,
            rag_mode="frozen",
            verifier=context.verifier,
        ),
        state,
    )
    return RuntimeTrialResult(
        result.outcome,
        state,
        retrieval_event=result.retrieval_event,
        context_event=result.context_event,
    )


def _bot_execute(context: Any, state: object) -> RuntimeTrialResult:
    if not isinstance(state, BoTStateV3):
        raise RuntimeStateError("INVALID_BOT_STYLE_STATE")
    config = _config(context, "bot_style")
    result = BoTPhase12Adapter().execute(
        BoTTrialContextV3(
            task=context.task,
            client=context.client,
            model=context.model,
            run_id=context.identities.run_id,
            trial_id=context.identities.trial_id,
            condition_id=context.identities.condition_id,
            branch=context.branch,
            config={**config, "embedding_provider": context.embedding_provider},
            order_key=context.identities.order_key,
            verifier=lambda answer: context.verifier(answer, context.task),
        ),
        state,
    )
    return RuntimeTrialResult(
        result.outcome,
        state,
        retrieval_event=result.retrieval_event,
        context_event=result.context_event,
        native_entries=() if result.native_entry is None else (result.native_entry,),
        write_envelopes=() if result.write_envelope is None else (result.write_envelope,),
    )


def _reflexion_execute(context: Any, state: object) -> RuntimeTrialResult:
    if not isinstance(state, ReflexionStateV3):
        raise RuntimeStateError("INVALID_REFLEXION_STYLE_STATE")
    result = ReflexionPhase12Adapter().execute(
        ReflexionTrialContextV3(
            task=context.task,
            client=context.client,
            model=context.model,
            run_id=context.identities.run_id,
            trial_id=context.identities.trial_id,
            condition_id=context.identities.condition_id,
            branch=context.branch,
            config=_config(context, "reflexion_style"),
            order_key=context.identities.order_key,
            verifier=context.verifier,
        ),
        state,
    )
    return RuntimeTrialResult(
        result.outcome,
        state,
        native_entries=result.native_reflections,
        write_envelopes=() if result.write_envelope is None else (result.write_envelope,),
    )


def _dc_execute(context: Any, state: object) -> RuntimeTrialResult:
    recorder = MethodCallRecorder(
        context.client, trial_context={"trial_id": context.identities.trial_id}
    )
    try:
        execution = dc_runtime.execute(replace(context, client=recorder), state)
    except (dc_runtime.DcRsRuntimeError, dc_runtime.dc.DcRsContractError) as error:
        raise RuntimeStateError(error.code) from error
    except (Phase13CostPolicyError, Phase13ProviderCallError) as error:
        return RuntimeTrialResult(
            _provider_failure(error, tuple(recorder.get_records())), state
        )
    result = execution.result
    native_entries = tuple(
        entry
        for entry in (
            dc_runtime.dc._archive_native(result.archive_entry),
            result.strategy_entry,
        )
        if entry is not None
    )
    envelopes = tuple(
        envelope
        for envelope in (result.archive_envelope, result.strategy_envelope)
        if envelope is not None
    )
    return RuntimeTrialResult(
        result.outcome,
        execution.state,
        native_entries=native_entries,
        write_envelopes=envelopes,
    )


def _dc_initial_state(context: Any) -> object:
    try:
        return dc_runtime.initial_state(context)
    except (dc_runtime.DcRsRuntimeError, dc_runtime.dc.DcRsContractError) as error:
        raise RuntimeStateError(error.code) from error


def _dc_serialize(state: object) -> object:
    try:
        return dc_runtime.serialize(state)
    except (dc_runtime.DcRsRuntimeError, dc_runtime.dc.DcRsContractError) as error:
        raise RuntimeStateError(error.code) from error


def _dc_restore(snapshot: object, context: Any) -> object:
    try:
        return dc_runtime.restore(snapshot, context)
    except (dc_runtime.DcRsRuntimeError, dc_runtime.dc.DcRsContractError) as error:
        raise RuntimeStateError(error.code) from error


def _native_entry(entry: MemoryEntry | NativeEntry, semantic_kind: str, component: str) -> NativeEntry:
    if isinstance(entry, NativeEntry):
        return entry
    parents = entry.metadata.get("direct_parent_ids", [])
    if not isinstance(parents, list) or any(not isinstance(item, str) for item in parents):
        parents = []
    return NativeEntry(
        entry_id=entry.entry_id,
        semantic_kind=semantic_kind,
        schema_version=NATIVE_ENTRY_V1,
        native_component=component,
        content=entry.content,
        content_hash=canonical_content_hash(entry.content),
        direct_parent_ids=tuple(parents),
    )


def _entry_from_envelope(envelope: Any) -> NativeEntry:
    return NativeEntry(
        entry_id=envelope.entry_id,
        semantic_kind=envelope.semantic_kind,
        schema_version=NATIVE_ENTRY_V1,
        native_component=envelope.native_component,
        content=envelope.content,
        content_hash=envelope.content_hash,
        direct_parent_ids=envelope.direct_parent_ids,
    )


def _nomem_serialize(state: object) -> NoMemRuntimeState:
    if state is not NOMEM_SINGLETON:
        raise RuntimeStateError("NOMEM_SINGLETON_REQUIRED")
    return NOMEM_SINGLETON


def _nomem_restore(snapshot: object, context: Any) -> NoMemRuntimeState:
    del context
    if snapshot is not NOMEM_SINGLETON:
        raise RuntimeStateError("NOMEM_SNAPSHOT_INVALID")
    return NOMEM_SINGLETON


def _fh_serialize(state: object) -> NativeState:
    if not isinstance(state, FullHistoryStateV3) or state.filter_state is not None:
        raise RuntimeStateError("FH_STATE_SERIALIZATION_UNSUPPORTED")
    return NativeState(
        "fh_bounded",
        tuple(_native_entry(record, "full_history_transcript", "history") for record in state.records),
        {
            "first_eviction_trial_id": state.first_eviction_trial_id,
            "injected_root_id": state.injected_root_id,
            "injected_root_was_visible": state.injected_root_was_visible,
            "records": [record.model_dump() for record in state.records],
        },
    )


def _fh_restore(snapshot: object, context: Any) -> FullHistoryStateV3:
    del context
    state = _native_state(snapshot, "fh_bounded")
    records = state.native_state.get("records")
    if not isinstance(records, list):
        raise RuntimeStateError("FH_SNAPSHOT_INVALID")
    return FullHistoryStateV3(
        records=[MemoryEntry.model_validate(record) for record in records],
        injected_root_id=state.native_state.get("injected_root_id"),
        injected_root_was_visible=bool(state.native_state.get("injected_root_was_visible", False)),
        first_eviction_trial_id=state.native_state.get("first_eviction_trial_id"),
    )


def _rag_serialize(state: object) -> NativeState:
    if not isinstance(state, RagFrozenStateV3) or state.corpus is None or state.index is None:
        raise RuntimeStateError("RAG_STATE_SERIALIZATION_UNSUPPORTED")
    return NativeState(
        "rag_frozen",
        tuple(
            NativeEntry(
                entry_id=document.document_id,
                semantic_kind="rag_document",
                schema_version=NATIVE_ENTRY_V1,
                native_component="corpus",
                content=document.text,
                content_hash=canonical_content_hash(document.text),
            )
            for document in state.corpus.active_documents
        ),
        {
            "branch": state.branch,
            "corpus_id": state.corpus.serialization_id,
            "index_id": state.index.serialization_id,
            "read_only": True,
        },
    )


def _rag_restore(snapshot: object, context: Any) -> RagFrozenStateV3:
    state = _native_state(snapshot, "rag_frozen")
    runtime = _rag_initial_state(context)
    if (
        runtime.corpus is None
        or runtime.index is None
        or state.native_state.get("branch") != runtime.branch
        or state.native_state.get("corpus_id") != runtime.corpus.serialization_id
        or state.native_state.get("index_id") != runtime.index.serialization_id
        or state.native_state.get("read_only") is not True
    ):
        raise RuntimeStateError("RAG_SNAPSHOT_MISMATCH")
    if tuple(_entry_id(entry) for entry in state.entries) != tuple(
        document.document_id for document in runtime.corpus.active_documents
    ):
        raise RuntimeStateError("RAG_SNAPSHOT_MISMATCH")
    return runtime


def _bot_serialize(state: object) -> NativeState:
    if not isinstance(state, BoTStateV3) or state.filter_state is not None:
        raise RuntimeStateError("BOT_STATE_SERIALIZATION_UNSUPPORTED")
    return NativeState(
        "bot_style",
        tuple(_native_entry(entry, "thought_template", "buffer") for entry in state.entries),
        {
            "active_capacity": state.active_capacity,
            "clean_competitor_ids": list(state.clean_competitor_ids),
            "templates": [_entry_id(entry) for entry in state.entries],
        },
    )


def _bot_restore(snapshot: object, context: Any) -> BoTStateV3:
    del context
    state = _native_state(snapshot, "bot_style")
    if not all(isinstance(entry, NativeEntry) for entry in state.entries):
        raise RuntimeStateError("BOT_SNAPSHOT_INVALID")
    return BoTStateV3(
        entries=[entry for entry in state.entries if isinstance(entry, NativeEntry)],
        clean_competitor_ids=tuple(state.native_state.get("clean_competitor_ids", ())),
        active_capacity=state.native_state.get("active_capacity"),
    )


def _reflexion_serialize(state: object) -> NativeState:
    if not isinstance(state, ReflexionStateV3) or state.filter_state is not None:
        raise RuntimeStateError("REFLEXION_STATE_SERIALIZATION_UNSUPPORTED")
    return NativeState(
        "reflexion_style",
        tuple(_native_entry(entry, "verbal_reflection", "reflections") for entry in state.reflections),
        {
            "active_capacity": state.active_capacity,
            "first_injected_eviction_trial_id": state.first_injected_eviction_trial_id,
            "injected_root_id": state.injected_root_id,
            "reflections": [_entry_id(entry) for entry in state.reflections],
        },
    )


def _reflexion_restore(snapshot: object, context: Any) -> ReflexionStateV3:
    del context
    state = _native_state(snapshot, "reflexion_style")
    if not all(isinstance(entry, NativeEntry) for entry in state.entries):
        raise RuntimeStateError("REFLEXION_SNAPSHOT_INVALID")
    return ReflexionStateV3(
        reflections=[entry for entry in state.entries if isinstance(entry, NativeEntry)],
        injected_root_id=state.native_state.get("injected_root_id"),
        active_capacity=state.native_state.get("active_capacity"),
        first_injected_eviction_trial_id=state.native_state.get("first_injected_eviction_trial_id"),
    )


def _native_state(snapshot: object, baseline: str) -> NativeState:
    if not isinstance(snapshot, NativeState) or snapshot.baseline != baseline:
        raise RuntimeStateError("INVALID_NATIVE_SNAPSHOT")
    return snapshot


def _entry_id(entry: str | NativeEntry | MemoryEntry) -> str:
    return entry if isinstance(entry, str) else entry.entry_id


def _maturity(serialize: Callable[[object], object], state: object, context: Any) -> object | None:
    condition = context.condition
    if condition is None:
        return None
    snapshot = serialize(state)
    if not isinstance(snapshot, NativeState):
        return None
    return evaluate_maturity(
        condition,
        serialize_checkpoint(snapshot),
        getattr(context, "maturity_horizon", 1),
    )


LIVE_BASELINE_REGISTRY: Mapping[str, RuntimeEntry] = {
    "nomem": RuntimeEntry(
        _nomem_initial_state,
        _nomem_execute,
        _nomem_serialize,
        _nomem_restore,
        lambda state, context: _maturity(_nomem_serialize, state, context),
    ),
    "fh_bounded": RuntimeEntry(
        _fh_initial_state,
        _fh_execute,
        _fh_serialize,
        _fh_restore,
        lambda state, context: _maturity(_fh_serialize, state, context),
    ),
    "rag_frozen": RuntimeEntry(
        _rag_initial_state,
        _rag_execute,
        _rag_serialize,
        _rag_restore,
        lambda state, context: _maturity(_rag_serialize, state, context),
    ),
    "bot_style": RuntimeEntry(
        _bot_initial_state,
        _bot_execute,
        _bot_serialize,
        _bot_restore,
        lambda state, context: _maturity(_bot_serialize, state, context),
    ),
    "reflexion_style": RuntimeEntry(
        _reflexion_initial_state,
        _reflexion_execute,
        _reflexion_serialize,
        _reflexion_restore,
        lambda state, context: _maturity(_reflexion_serialize, state, context),
    ),
}


PHASE13_CORE_BASELINE_REGISTRY: Mapping[str, RuntimeEntry] = {
    **LIVE_BASELINE_REGISTRY,
    "dc_rs": RuntimeEntry(
        _dc_initial_state,
        _dc_execute,
        _dc_serialize,
        _dc_restore,
        lambda _state, _context: None,
    ),
}
