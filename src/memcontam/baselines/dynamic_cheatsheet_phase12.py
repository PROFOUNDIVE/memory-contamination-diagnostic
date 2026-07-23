from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

from memcontam.baselines import dynamic_cheatsheet_optional as legacy_dc
from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.memory.admission import AdmissionContext, AdmissionDecision, AdmissionError
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry
from memcontam.memory.embeddings import EmbeddingProvider
from memcontam.memory.filtered_state import (
    CandidateWrite,
    FilterTransition,
    FilteredCheckpoint,
    route_candidate_write,
)
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json
from memcontam.tools.base import (
    ToolExecutionError,
    ToolExecutor,
    ToolInfrastructureError,
    ToolPolicyError,
    ToolResult,
    ToolRuntimeContract,
)
from memcontam.tools.execution_loop import LlmCall, ToolProtocolError, run_tool_loop


__all__ = [
    "BaselineStepResultV3",
    "DcRsContractError",
    "DcRsPhase12Adapter",
    "DcRsStateV3",
    "DcRsToolContractError",
    "DcRsTrialContextV3",
    "StrategyCandidateState",
    "curate_pre_generation",
]


Branch = Literal["clean", "correct", "irrelevant", "contam", "filter"]
LineageStatus = Literal["exact", "unavailable", "approximate"]

_MAX_TOOL_TRACE_EVENTS = 3
_MAX_TOOL_TRACE_CHARS = 4096


class DcRsContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DcRsToolContractError(DcRsContractError):
    pass


@dataclass(frozen=True)
class DcRsTrialContextV3:
    task: TaskInstance
    client: LLMClient
    model: str
    run_id: str
    trial_id: str
    condition_id: str
    branch: Branch
    config: Mapping[str, Any]
    order_key: int | str
    verifier: Any = None
    current_outcome: Any = None
    outcome: Any = None
    verifier_result: Any = None
    current_verifier_result: Any = None
    current_generated_output: Any = None
    current_parsed_answer: Any = None

    def __post_init__(self) -> None:
        if not all((self.run_id, self.trial_id, self.condition_id)):
            raise DcRsContractError("INVALID_TRIAL_CONTEXT")
        if not isinstance(self.order_key, (int, str)) or isinstance(self.order_key, bool):
            raise DcRsContractError("INVALID_TRIAL_CONTEXT")


@dataclass
class DcRsStateV3:
    archive: list[MemoryEntry | NativeEntry]
    strategies: list[MemoryEntry | NativeEntry] | None = None
    injected_root_id: str | None = None
    filter_state: FilteredCheckpoint | None = None
    admission_context: AdmissionContext | None = None

    def __post_init__(self) -> None:
        self.strategies = list(self.strategies or ())
        archive_ids = [_archive_entry(entry).entry_id for entry in self.archive]
        strategy_ids = [_strategy_entry(entry).entry_id for entry in self.strategies]
        if len(set((*archive_ids, *strategy_ids))) != len((*archive_ids, *strategy_ids)):
            raise DcRsContractError("DUPLICATE_COMPONENT")
        if (self.filter_state is None) != (self.admission_context is None):
            raise DcRsContractError("FILTER_ADMISSION_CONTEXT_REQUIRED")
        if self.injected_root_id is not None:
            if self.injected_root_id in strategy_ids:
                raise DcRsContractError("DIRECT_STRATEGY_INJECTION")
            if self.injected_root_id not in archive_ids:
                raise DcRsContractError("INVALID_INJECTED_ARCHIVE")


@dataclass(frozen=True)
class StrategyCandidateState:
    content: str
    parser_status: str
    explicit_source_ids: tuple[str, ...]
    retrieved_archive_ids: tuple[str, ...]
    lineage_status: LineageStatus


@dataclass(frozen=True)
class BaselineStepResultV3:
    outcome: BaselineExecutionOutcome
    strategy_candidate: StrategyCandidateState
    strategy_entry: NativeEntry | None
    strategy_envelope: MemoryCardEnvelopeV3 | None
    strategy_admission: AdmissionDecision
    strategy_transition: FilterTransition | None
    archive_entry: MemoryEntry
    archive_envelope: MemoryCardEnvelopeV3
    archive_transition: FilterTransition | None


def curate_pre_generation(
    curator_output: str,
    *,
    fallback_strategy: str,
    retrieved_archive_ids: Sequence[str],
    inferred_parent_ids: Sequence[str] = (),
) -> StrategyCandidateState:
    """Parse a curator result without turning visible archive entries into parents."""
    if inferred_parent_ids:
        raise DcRsContractError("IMPLICIT_PARENT_UNION")
    content, parser_status = legacy_dc._extract_cheatsheet(curator_output, fallback_strategy)
    source_ids = _explicit_source_ids(curator_output)
    if len(source_ids) != len(set(source_ids)) or any(not entry_id for entry_id in source_ids):
        raise DcRsContractError("INVALID_EXPLICIT_SOURCE_IDS")
    return StrategyCandidateState(
        content=content,
        parser_status=parser_status,
        explicit_source_ids=source_ids,
        retrieved_archive_ids=tuple(retrieved_archive_ids),
        lineage_status="exact" if source_ids else "unavailable",
    )


class DcRsPhase12Adapter:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.cache_dir = cache_dir

    def execute(self, trial: DcRsTrialContextV3, state: DcRsStateV3) -> BaselineStepResultV3:
        _validate_trial_and_state(trial, state)
        provider = self.embedding_provider or trial.config.get("embedding_provider")
        if provider is None:
            raise DcRsContractError("EMBEDDING_PROVIDER_REQUIRED")

        active_archive = _active_archive(state)
        active_strategies = _active_strategies(state)
        prior_strategy = active_strategies[-1] if active_strategies else None
        prior_content = "" if prior_strategy is None else prior_strategy.content
        memory_before = _memory_snapshots(active_strategies, active_archive)
        retriever = legacy_dc.DynamicCheatsheetRetrievalSynthesisPolicy(
            embedding_provider=provider,
            cache_dir=self.cache_dir,
        )
        canonical_task = canonical_task_json(trial.task)
        retrieved_records = retriever._retrieve_pairs(canonical_task, active_archive, trial.trial_id)
        archive_by_id = {entry.entry_id: entry for entry in active_archive}
        retrieved_archive = [archive_by_id[record.document_id] for record in retrieved_records]
        recorder = MethodCallRecorder(trial.client)
        call_config = {**dict(trial.config), "sample_id": trial.task.sample_id}
        curation_message, curation_spans = legacy_dc._synthesis_message_with_sources(
            canonical_task,
            [] if prior_strategy is None else [_strategy_memory(prior_strategy)],
            retrieved_archive,
        )
        curation_message = {
            **curation_message,
            "content": (
                f"{curation_message['content']}\n\n"
                "If you directly use archive interactions, append "
                "<source_ids>comma-separated archive IDs</source_ids>. "
                f"Available archive IDs: {', '.join(entry.entry_id for entry in retrieved_archive)}"
            ),
        }
        curated = recorder.chat(
            [curation_message],
            model=trial.model,
            config={
                **_curator_call_config(call_config),
                "method_stage": "dc_rs_synthesize",
                "source_spans": curation_spans,
            },
        )
        if _is_tool_action(curated.content):
            raise DcRsToolContractError("CURATOR_TOOL_FORBIDDEN")
        candidate = curate_pre_generation(
            curated.content,
            fallback_strategy=prior_content,
            retrieved_archive_ids=tuple(entry.entry_id for entry in retrieved_archive),
        )
        if candidate.explicit_source_ids and not set(candidate.explicit_source_ids).issubset(
            {entry.entry_id for entry in active_archive}
        ):
            candidate = replace(candidate, lineage_status="approximate")
        strategy_entry, strategy_envelope, strategy_admission, strategy_transition = _admit_strategy(
            candidate, state, trial, prior_strategy
        )

        admitted_strategies = _active_strategies(state)
        strategy_content = "" if not admitted_strategies else admitted_strategies[-1].content
        tool_mode = _tool_mode(call_config)
        generation_builder = (
            legacy_dc._dc_rs_tool_generation_message
            if tool_mode == "python_sandbox"
            else legacy_dc._dc_rs_generation_message
        )
        generation_message, generation_spans = generation_builder(
            canonical_task,
            strategy_content,
            recorder.get_records()[-1].call_id,
            curation_spans,
            trial.config.get("_logging_target_set_id"),
        )
        generation_config = {
            **call_config,
            "method_stage": "dc_rs_generate",
            "source_spans": generation_spans,
        }
        generated = recorder.chat([generation_message], model=trial.model, config=generation_config)
        tool_events: tuple[Any, ...] = ()
        tool_trace: str | None = None
        if tool_mode == "python_sandbox":
            tool_result = _run_generation_tool_loop(
                trial, recorder, generation_message, generation_config, generated.content
            )
            generated_output = tool_result.answer
            tool_events = tool_result.tool_events
            tool_trace = _canonical_tool_trace(recorder, tool_events, generated_output)
        else:
            generated_output = generated.content
        archive_entry = _archive_write(generated_output, canonical_task, trial, tool_trace=tool_trace)
        state.archive.append(archive_entry)
        archive_envelope = _archive_envelope(archive_entry, trial)
        archive_transition = _route_write(state, _archive_native(archive_entry), archive_envelope, trial)

        try:
            parsed_answer = parse_final_answer(generated_output)
        except ValueError:
            outcome = _outcome(
                "failed",
                generated_output,
                None,
                None,
                recorder,
                state,
                retrieved_records,
                trial,
                memory_before,
                "BaselineOutputError",
                "dc_rs_invalid_final_answer",
                "invalid_final_answer",
                metadata={"tool_events": tool_events},
            )
        else:
            archive_entry.metadata["parsed_answer"] = parsed_answer
            try:
                verifier_result = trial.verifier(parsed_answer, trial.task) if trial.verifier else None
            except Exception:
                outcome = _outcome(
                    "failed",
                    generated_output,
                    parsed_answer,
                    None,
                    recorder,
                    state,
                    retrieved_records,
                    trial,
                    memory_before,
                    "VerifierContractError",
                    "verifier_contract_failed",
                    "verifier_contract_failed",
                    metadata={"tool_events": tool_events},
                )
            else:
                outcome = _outcome(
                    "succeeded",
                    generated_output,
                    parsed_answer,
                    verifier_result,
                    recorder,
                    state,
                    retrieved_records,
                    trial,
                    memory_before,
                    metadata={"tool_events": tool_events},
                )
        return BaselineStepResultV3(
            outcome=outcome,
            strategy_candidate=candidate,
            strategy_entry=strategy_entry,
            strategy_envelope=strategy_envelope,
            strategy_admission=strategy_admission,
            strategy_transition=strategy_transition,
            archive_entry=archive_entry,
            archive_envelope=archive_envelope,
            archive_transition=archive_transition,
        )


def _validate_trial_and_state(trial: DcRsTrialContextV3, state: DcRsStateV3) -> None:
    if (trial.branch == "filter") != (state.filter_state is not None):
        raise DcRsContractError("FILTER_STATE_BRANCH_MISMATCH")
    if any(
        value is not None
        for value in (
            trial.current_outcome,
            trial.outcome,
            trial.verifier_result,
            trial.current_verifier_result,
            trial.current_generated_output,
            trial.current_parsed_answer,
        )
    ):
        raise DcRsContractError("CURRENT_OUTCOME_LEAKAGE")
    if any(key in trial.config for key in ("inferred_parent_ids", "direct_parent_ids", "parent_entry_ids")):
        raise DcRsContractError("IMPLICIT_PARENT_UNION")
    if any(
        key in trial.config
        for key in (
            "current_outcome",
            "outcome",
            "verifier_result",
            "current_verifier_result",
            "current_generated_output",
            "current_parsed_answer",
        )
    ):
        raise DcRsContractError("CURRENT_OUTCOME_LEAKAGE")


def _curator_call_config(config: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {
        "tool_executor",
        "tool_runtime_contract",
        "max_tool_rounds",
        "_tool_event_writer",
        "current_outcome",
        "outcome",
        "verifier_result",
        "current_verifier_result",
        "current_generated_output",
        "current_parsed_answer",
        "verifier",
    }
    return {key: value for key, value in config.items() if key not in forbidden} | {
        "tool_mode": "text_only"
    }


def _is_tool_action(content: str) -> bool:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and isinstance(value.get("action"), str)


def _tool_mode(config: Mapping[str, Any]) -> Literal["text_only", "python_sandbox"]:
    mode = config.get("tool_mode", "text_only")
    if mode not in {"text_only", "python_sandbox"}:
        raise DcRsToolContractError("INVALID_TOOL_MODE")
    return mode


class _BoundedToolExecutor:
    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor
        self.scientific_capable = executor.scientific_capable

    def execute(self, request: Any, contract: ToolRuntimeContract) -> ToolResult:
        if len(request.code) > _MAX_TOOL_TRACE_CHARS:
            raise ToolExecutionError("UNBOUNDED_TOOL_TRACE")
        result = self._executor.execute(request, contract)
        if len(result.stdout) > _MAX_TOOL_TRACE_CHARS or len(result.stderr) > _MAX_TOOL_TRACE_CHARS:
            raise ToolExecutionError("UNBOUNDED_TOOL_TRACE")
        return result


def _run_generation_tool_loop(
    trial: DcRsTrialContextV3,
    recorder: MethodCallRecorder,
    generation_message: dict[str, str],
    generation_config: Mapping[str, Any],
    initial_content: str,
) -> Any:
    executor = trial.config.get("tool_executor")
    policy = trial.config.get("tool_runtime_contract")
    rounds = trial.config.get("max_tool_rounds", _MAX_TOOL_TRACE_EVENTS)
    if (
        executor is None
        or not hasattr(executor, "execute")
        or not hasattr(executor, "scientific_capable")
        or not isinstance(policy, ToolRuntimeContract)
    ):
        raise DcRsToolContractError("TOOL_CONTRACT_REQUIRED")
    if type(rounds) is not int or not 1 <= rounds <= _MAX_TOOL_TRACE_EVENTS:
        raise DcRsToolContractError("UNBOUNDED_TOOL_TRACE")
    calls = recorder.get_records()
    if not calls or calls[-1].call_id is None:
        raise DcRsToolContractError("MISSING_INITIAL_CALL")
    try:
        return run_tool_loop(
            LlmCall(
                call_id=calls[-1].call_id,
                content=initial_content,
                messages=[generation_message],
                model=trial.model,
                config=generation_config,
                run_id=trial.run_id,
                trial_id=trial.trial_id,
                max_rounds=rounds,
            ),
            recorder,
            _BoundedToolExecutor(executor),
            policy,
            writer=trial.config.get("_tool_event_writer"),
        )
    except (ToolExecutionError, ToolInfrastructureError, ToolPolicyError, ToolProtocolError) as error:
        raise DcRsToolContractError(error.code) from error


def _canonical_tool_trace(
    recorder: MethodCallRecorder, tool_events: Sequence[Any], final_answer: str
) -> str:
    if len(tool_events) > _MAX_TOOL_TRACE_EVENTS or len(final_answer) > _MAX_TOOL_TRACE_CHARS:
        raise DcRsToolContractError("UNBOUNDED_TOOL_TRACE")
    calls = {call.call_id: call for call in recorder.get_records()}
    executions: list[dict[str, Any]] = []
    for event in tool_events:
        if (
            not isinstance(event.output, str)
            or not isinstance(event.stderr, str)
            or len(event.output) > _MAX_TOOL_TRACE_CHARS
            or len(event.stderr) > _MAX_TOOL_TRACE_CHARS
        ):
            raise DcRsToolContractError("UNBOUNDED_TOOL_TRACE")
        parent = calls.get(event.parent_call_id)
        if parent is None or event.continuation_call_id not in calls:
            raise DcRsToolContractError("MALFORMED_TOOL_TRACE")
        try:
            action = json.loads(parent.raw_response or "")
        except json.JSONDecodeError as error:
            raise DcRsToolContractError("MALFORMED_TOOL_TRACE") from error
        code = action.get("code") if action.get("action") == "execute_python" else None
        if not isinstance(code, str) or len(code) > _MAX_TOOL_TRACE_CHARS:
            raise DcRsToolContractError("MALFORMED_TOOL_TRACE")
        if hashlib.sha256(code.encode("utf-8")).hexdigest() != event.code_hash:
            raise DcRsToolContractError("MALFORMED_TOOL_TRACE")
        executions.append(
            {
                "code": code,
                "code_hash": event.code_hash,
                "exit_code": event.exit_code,
                "stderr": event.stderr,
                "stdout": event.output,
            }
        )
    return json.dumps({"executions": executions, "final": final_answer}, sort_keys=True, separators=(",", ":"))


def _active_archive(state: DcRsStateV3) -> list[MemoryEntry]:
    archive = [_archive_entry(entry) for entry in state.archive]
    return _active_components(state, archive, "FILTER_ACTIVE_ARCHIVE_MISSING")


def _active_strategies(state: DcRsStateV3) -> list[NativeEntry]:
    strategies = [_strategy_entry(entry) for entry in state.strategies or ()]
    return _active_components(state, strategies, "FILTER_ACTIVE_STRATEGY_MISSING")


def _active_components(
    state: DcRsStateV3,
    components: Sequence[MemoryEntry | NativeEntry],
    missing_code: str,
) -> list[Any]:
    if state.filter_state is None:
        return list(components)
    active_ids = {
        entry.entry_id if isinstance(entry, NativeEntry) else entry for entry in state.filter_state.reader_entries
    }
    known = {entry.entry_id: entry for entry in components}
    required = active_ids & set(known)
    selected = [entry for entry in components if entry.entry_id in active_ids]
    component_ids = {entry.entry_id for entry in selected}
    if component_ids != required:
        raise DcRsContractError(missing_code)
    return selected


def _admit_strategy(
    candidate: StrategyCandidateState,
    state: DcRsStateV3,
    trial: DcRsTrialContextV3,
    prior_strategy: NativeEntry | None,
) -> tuple[
    NativeEntry | None,
    MemoryCardEnvelopeV3 | None,
    AdmissionDecision,
    FilterTransition | None,
]:
    if candidate.parser_status != "accepted" or not candidate.explicit_source_ids:
        return None, None, AdmissionDecision("", False, "UNAVAILABLE_LINEAGE"), None
    active_ids = {entry.entry_id for entry in _active_archive(state)}
    lineage_status: LineageStatus = (
        "exact" if set(candidate.explicit_source_ids).issubset(active_ids) else "approximate"
    )
    entry = NativeEntry(
        entry_id=f"dc_rs_strategy:{trial.trial_id}",
        semantic_kind="dynamic_cheatsheet",
        schema_version=NATIVE_ENTRY_V1,
        native_component="strategy",
        content=candidate.content,
        content_hash=canonical_content_hash(candidate.content),
        direct_parent_ids=candidate.explicit_source_ids,
    )
    envelope = MemoryCardEnvelopeV3(
        entry_id=entry.entry_id,
        baseline="dynamic_cheatsheet_rs_optional",
        semantic_kind=entry.semantic_kind,
        schema_version=MEMORY_CARD_V3,
        writer_id="dc_strategy_writer",
        writer_event_id=f"{trial.trial_id}:dc-rs-synthesize",
        writer_stage="dc_rs_synthesize",
        created_trial_id=trial.trial_id,
        source_trial_ids=(trial.trial_id,),
        source_outcome=None,
        trial_support_ids=(trial.trial_id,),
        memory_support_ids=entry.direct_parent_ids,
        direct_parent_ids=entry.direct_parent_ids,
        version_predecessor_id=None if prior_strategy is None else prior_strategy.entry_id,
        order_key=_strategy_order_key(trial.order_key),
        native_component=entry.native_component,
        content=entry.content,
        content_hash=entry.content_hash,
    )
    if state.filter_state is None:
        if lineage_status != "exact":
            raise DcRsContractError("EXPLICIT_PARENT_NOT_ACTIVE")
        assert state.strategies is not None
        state.strategies.append(entry)
        return entry, envelope, AdmissionDecision(entry.entry_id, True, "ADMITTED"), None

    assert state.strategies is not None
    state.strategies.append(entry)
    transition = _route_write(state, entry, envelope, trial)
    assert transition is not None
    return entry, envelope, transition.decision, transition


def _route_write(
    state: DcRsStateV3,
    entry: NativeEntry,
    envelope: MemoryCardEnvelopeV3,
    trial: DcRsTrialContextV3,
) -> FilterTransition | None:
    if state.filter_state is None:
        return None
    assert state.admission_context is not None
    context = replace(
        state.admission_context,
        writer_event_ids=state.admission_context.writer_event_ids | {envelope.writer_event_id},
        trial_record_ids=state.admission_context.trial_record_ids | {trial.trial_id},
        evidence_envelopes=(*state.admission_context.evidence_envelopes, envelope),
    )
    try:
        transition = route_candidate_write(state.filter_state, CandidateWrite(entry, envelope), context)
    except AdmissionError as error:
        raise DcRsContractError(error.code) from error
    state.filter_state = transition.state
    state.admission_context = context
    return transition


def _archive_write(
    raw_output: str,
    canonical_task: str,
    trial: DcRsTrialContextV3,
    *,
    tool_trace: str | None = None,
) -> MemoryEntry:
    metadata: dict[str, Any] = {"generated_output": raw_output, "parsed_answer": None}
    if tool_trace is not None:
        metadata["tool_trace"] = tool_trace
    return MemoryEntry(
        entry_id=f"dc_rs_pair:{trial.trial_id}",
        content=canonical_task,
        memory_type="dc_rs_io_pair",
        source_trial_id=trial.trial_id,
        metadata=metadata,
    )


def _archive_native(entry: MemoryEntry) -> NativeEntry:
    payload = {"input": entry.content, "raw_output": entry.metadata["generated_output"]}
    tool_trace = entry.metadata.get("tool_trace")
    if tool_trace is not None:
        payload["tool_trace"] = tool_trace
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return NativeEntry(
        entry_id=entry.entry_id,
        semantic_kind="dc_rs_io_pair",
        schema_version=NATIVE_ENTRY_V1,
        native_component="archive",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def _archive_envelope(entry: MemoryEntry, trial: DcRsTrialContextV3) -> MemoryCardEnvelopeV3:
    native = _archive_native(entry)
    return MemoryCardEnvelopeV3(
        entry_id=native.entry_id,
        baseline="dynamic_cheatsheet_rs_optional",
        semantic_kind=native.semantic_kind,
        schema_version=MEMORY_CARD_V3,
        writer_id="dc_archive_writer",
        writer_event_id=f"{trial.trial_id}:dc-rs-generate",
        writer_stage="dc_rs_generate",
        created_trial_id=trial.trial_id,
        source_trial_ids=(trial.trial_id,),
        source_outcome=None,
        trial_support_ids=(trial.trial_id,),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=_archive_order_key(trial.order_key),
        native_component=native.native_component,
        content=native.content,
        content_hash=native.content_hash,
    )


def _outcome(
    status: Literal["succeeded", "failed"],
    response: str,
    parsed_answer: str | None,
    verifier_result: Any,
    recorder: MethodCallRecorder,
    state: DcRsStateV3,
    retrieved_records: Sequence[Any],
    trial: DcRsTrialContextV3,
    memory_before: Sequence[dict[str, Any]],
    error_type: Any = None,
    failure_disposition: Any = None,
    scientific_ineligibility_reason: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> BaselineExecutionOutcome:
    calls = recorder.get_records()
    return BaselineExecutionOutcome(
        status=status,
        final_response=response,
        parsed_answer=parsed_answer,
        verifier_result=verifier_result,
        answer_call_id=calls[-1].call_id if calls else None,
        method_calls=tuple(calls),
        memory_before=tuple(memory_before),
        memory_after=tuple(_memory_snapshots(_active_strategies(state), _active_archive(state))),
        retrieved_memory=tuple(
            {**record.model_dump(), "entry_id": record.document_id} for record in retrieved_records
        ),
        retrieved_scores=tuple(record.score for record in retrieved_records),
        memory_write_event={
            "type": "dynamic_cheatsheet_rs_update",
            "source_trial_id": trial.trial_id,
            "status": "accepted",
        },
        error_type=error_type,
        failure_disposition=failure_disposition,
        scientific_ineligibility_reason=scientific_ineligibility_reason,
        metadata={} if metadata is None else dict(metadata),
    )


def _memory_snapshots(strategies: Sequence[NativeEntry], archive: Sequence[MemoryEntry]) -> list[dict[str, Any]]:
    return [
        *(_strategy_memory(entry).model_dump() for entry in strategies),
        *(entry.model_dump() for entry in archive),
    ]


def _archive_entry(entry: MemoryEntry | NativeEntry) -> MemoryEntry:
    if isinstance(entry, MemoryEntry):
        if entry.memory_type == "dynamic_cheatsheet":
            raise DcRsContractError("DIRECT_STRATEGY_INJECTION")
        if entry.memory_type != "dc_rs_io_pair" or not isinstance(entry.metadata.get("generated_output"), str):
            raise DcRsContractError("INVALID_ARCHIVE_COMPONENT")
        return entry
    if not isinstance(entry, NativeEntry):
        raise DcRsContractError("INVALID_ARCHIVE_COMPONENT")
    if (entry.semantic_kind, entry.native_component, entry.schema_version) == (
        "dynamic_cheatsheet",
        "strategy",
        NATIVE_ENTRY_V1,
    ):
        raise DcRsContractError("DIRECT_STRATEGY_INJECTION")
    if (entry.semantic_kind, entry.native_component, entry.schema_version) != (
        "dc_rs_io_pair",
        "archive",
        NATIVE_ENTRY_V1,
    ):
        raise DcRsContractError("INVALID_ARCHIVE_COMPONENT")
    input_text, raw_output, tool_trace = _native_archive_values(entry.content)
    metadata = {"generated_output": raw_output}
    if tool_trace is not None:
        metadata["tool_trace"] = tool_trace
    return MemoryEntry(
        entry_id=entry.entry_id,
        content=input_text,
        memory_type="dc_rs_io_pair",
        metadata=metadata,
    )


def _strategy_entry(entry: MemoryEntry | NativeEntry) -> NativeEntry:
    if isinstance(entry, NativeEntry):
        native = entry
    elif isinstance(entry, MemoryEntry) and entry.memory_type == "dynamic_cheatsheet":
        if not entry.metadata.get("direct_parent_ids"):
            raise DcRsContractError("DIRECT_STRATEGY_INJECTION")
        parents = _metadata_ids(entry, "direct_parent_ids")
        native = NativeEntry(
            entry_id=entry.entry_id,
            semantic_kind="dynamic_cheatsheet",
            schema_version=NATIVE_ENTRY_V1,
            native_component="strategy",
            content=entry.content,
            content_hash=canonical_content_hash(entry.content),
            direct_parent_ids=parents,
        )
    else:
        raise DcRsContractError("INVALID_STRATEGY_COMPONENT")
    if (native.semantic_kind, native.native_component, native.schema_version) != (
        "dynamic_cheatsheet",
        "strategy",
        NATIVE_ENTRY_V1,
    ):
        raise DcRsContractError("INVALID_STRATEGY_COMPONENT")
    if not native.direct_parent_ids:
        raise DcRsContractError("DIRECT_STRATEGY_INJECTION")
    return native


def _strategy_memory(entry: NativeEntry) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry.entry_id,
        content=entry.content,
        memory_type="dynamic_cheatsheet",
        metadata={"direct_parent_ids": list(entry.direct_parent_ids)},
    )


def _native_archive_values(content: str) -> tuple[str, str, str | None]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return content, content, None
    if not isinstance(value, dict) or not all(isinstance(value.get(key), str) for key in ("input", "raw_output")):
        raise DcRsContractError("INVALID_ARCHIVE_COMPONENT")
    tool_trace = value.get("tool_trace")
    if tool_trace is not None and not isinstance(tool_trace, str):
        raise DcRsContractError("INVALID_ARCHIVE_COMPONENT")
    return value["input"], value["raw_output"], tool_trace


def _metadata_ids(entry: MemoryEntry, key: str) -> tuple[str, ...]:
    value = entry.metadata.get(key, ())
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise DcRsContractError("INVALID_STRATEGY_COMPONENT")
    return tuple(value)


def _explicit_source_ids(output: str) -> tuple[str, ...]:
    start_tag = "<source_ids>"
    end_tag = "</source_ids>"
    start = output.find(start_tag)
    if start < 0:
        return ()
    end = output.find(end_tag, start + len(start_tag))
    if end < 0:
        raise DcRsContractError("INVALID_EXPLICIT_SOURCE_IDS")
    value = output[start + len(start_tag) : end].strip()
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in value.split(",")]
    if not isinstance(parsed, list) or any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise DcRsContractError("INVALID_EXPLICIT_SOURCE_IDS")
    return tuple(item.strip() for item in parsed)


def _strategy_order_key(order_key: int | str) -> int | str:
    return order_key * 1_000 + 1 if isinstance(order_key, int) else f"{order_key}:strategy"


def _archive_order_key(order_key: int | str) -> int | str:
    return order_key * 1_000 + 2 if isinstance(order_key, int) else f"{order_key}:archive"
