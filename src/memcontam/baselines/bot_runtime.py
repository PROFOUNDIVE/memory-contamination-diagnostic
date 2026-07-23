from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from memcontam.baselines.bot_read import BoTRetrievalDecision, retrieve_top_template
from memcontam.baselines.bot_solve import (
    parse_bot_solve_result,
    render_tool_augmented_bot_solve_messages,
)
from memcontam.baselines.bot_style import BotStylePolicy
from memcontam.baselines.bot_write import (
    BoTTemplatePayload,
    BoTToolContractError,
    build_template_entry,
    distill_thought_template,
    visible_memory_for_retrieval_decision,
)
from memcontam.baselines.common import parse_final_answer
from memcontam.baselines.contracts import (
    BaselineExecutionOutcome,
    ErrorType,
    FailureDisposition,
    ScientificIneligibilityReason,
)
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.provenance import phase11_lineage_metadata
from memcontam.logging.schema import VerifierResult
from memcontam.memory.bot_buffer import (
    BotBufferIdentity,
    NativeNoveltyDecision,
    evaluate_native_novelty,
)
from memcontam.memory.embeddings import EmbeddingProvider
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json
from memcontam.tools.base import (
    ToolExecutionError,
    ToolInfrastructureError,
    ToolPolicyError,
    ToolRuntimeContract,
)
from memcontam.tools.execution_loop import LlmCall, ToolProtocolError, run_tool_loop


Verifier = Callable[[str], VerifierResult | bool]


@dataclass(frozen=True)
class FrozenNativeTransition:
    candidate: MemoryEntry
    decision: NativeNoveltyDecision
    memory_before: tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class MaterializedNativeTransition:
    memory_after: tuple[MemoryEntry, ...]
    memory_write_event: dict[str, Any]


class BotRuntime:
    def __init__(self, policy: BotStylePolicy | None = None) -> None:
        self.policy = policy or BotStylePolicy()

    def run(
        self,
        *,
        identity: BotBufferIdentity,
        task: TaskInstance,
        buffer_snapshot: list[MemoryEntry],
        client: LLMClient,
        model: str,
        config: dict[str, Any],
        verifier: Verifier | None = None,
    ) -> BaselineExecutionOutcome:
        call_config = {**config, "sample_id": config.get("sample_id", task.sample_id)}
        embedding_provider = call_config.get("embedding_provider")
        if embedding_provider is None:
            raise ValueError("BoT runtime requires an explicit embedding_provider")
        recorder = MethodCallRecorder(client)
        memory_before = tuple(entry.model_dump() for entry in buffer_snapshot)
        tool_mode = _tool_mode(call_config)
        metadata: dict[str, Any] = {
            "bot_buffer_identity": asdict(identity),
            "tool_mode": tool_mode,
            "tool_events": (),
            "executed_trajectory": [],
        }
        text_call_config = _text_call_config(call_config)

        try:
            distilled = self.policy.problem_distillation(task, recorder, model, text_call_config)
        except ValueError:
            return _failure_outcome(
                recorder, memory_before, metadata, "bot_invalid_problem_distillation", None
            )

        metadata["distilled_problem"] = distilled.model_dump()
        retrieval_decision = retrieve_top_template(distilled, buffer_snapshot, embedding_provider)
        metadata["retrieval_decision"] = _retrieval_decision_metadata(retrieval_decision)
        if tool_mode == "text_only":
            raw_solve = self.policy.template_instantiation_solve(
                task,
                distilled,
                recorder,
                model,
                text_call_config,
                retrieval_decision=retrieval_decision,
            )
            answer_call_id = recorder.get_records()[-1].call_id
        else:
            try:
                raw_solve, answer_call_id, tool_events, trajectory = _tool_augmented_solve(
                    identity=identity,
                    task=task,
                    distilled_problem=distilled,
                    retrieval_decision=retrieval_decision,
                    recorder=recorder,
                    model=model,
                    config=call_config,
                )
            except (
                ToolExecutionError,
                ToolInfrastructureError,
                ToolPolicyError,
                ToolProtocolError,
            ) as error:
                metadata["tool_error_code"] = error.code
                return _failure_outcome(
                    recorder,
                    memory_before,
                    metadata,
                    "bot_invalid_solve_result",
                    recorder.get_records()[-1].call_id if recorder.get_records() else None,
                )
            metadata["tool_events"] = tool_events
            metadata["executed_trajectory"] = trajectory
        try:
            solve_result = parse_bot_solve_result(raw_solve, retrieval_decision)
        except ValueError:
            return _failure_outcome(
                recorder,
                memory_before,
                metadata,
                "bot_invalid_solve_result",
                answer_call_id,
                final_response=raw_solve,
            )

        metadata["selected_structure"] = solve_result.selected_structure
        try:
            parsed_answer = parse_final_answer(solve_result.final_answer)
        except ValueError:
            return _failure_outcome(
                recorder,
                memory_before,
                metadata,
                "bot_invalid_solve_result",
                answer_call_id,
                final_response=raw_solve,
            )
        metadata["solution_trace"] = solve_result.solution_trace
        visible_memory = visible_memory_for_retrieval_decision(retrieval_decision)
        visible_entry_ids = [entry.entry_id for entry in visible_memory]
        call_config["visible_memory_ids"] = visible_entry_ids
        text_call_config["visible_memory_ids"] = visible_entry_ids
        try:
            payload = distill_thought_template(
                canonical_task=canonical_task_json(task),
                distilled_problem=distilled,
                retrieval_decision=retrieval_decision,
                selected_structure=solve_result.selected_structure,
                solution_trace=solve_result.solution_trace,
                final_answer=solve_result.final_answer,
                visible_memory=visible_memory,
                client=recorder,
                model=model,
                config=text_call_config,
                executed_trajectory=metadata["executed_trajectory"],
                require_executed_programming=tool_mode == "python_sandbox",
            )
        except BoTToolContractError as error:
            metadata["tool_contract_error"] = error.code
            return invalid_distillation_failure_outcome(
                recorder=recorder,
                memory_before=memory_before,
                metadata=metadata,
                answer_call_id=answer_call_id,
                final_response=solve_result.final_answer,
                parsed_answer=parsed_answer,
                retrieval_decision=retrieval_decision,
            )
        except ValueError:
            return invalid_distillation_failure_outcome(
                recorder=recorder,
                memory_before=memory_before,
                metadata=metadata,
                answer_call_id=answer_call_id,
                final_response=solve_result.final_answer,
                parsed_answer=parsed_answer,
                retrieval_decision=retrieval_decision,
            )

        metadata["thought_template"] = {
            "description": payload.description,
            "template": payload.template,
            "category": payload.category,
            "explicitly_used_memory_ids": list(payload.explicitly_used_memory_ids),
        }
        frozen_transition = freeze_native_transition(
            payload=payload,
            buffer_snapshot=buffer_snapshot,
            source_trial_id=_trial_id(identity, task),
            embedding_provider=embedding_provider,
            visible_entry_ids=visible_entry_ids,
            config=call_config,
        )
        try:
            verifier_result = _verify(verifier, parsed_answer)
        except Exception:
            materialized = materialize_frozen_transition(frozen_transition, source_outcome=None)
            return _failure_outcome(
                recorder,
                memory_before,
                metadata,
                "verifier_contract_failed",
                answer_call_id,
                final_response=solve_result.final_answer,
                parsed_answer=parsed_answer,
                memory_after=materialized.memory_after,
                memory_write_event=materialized.memory_write_event,
                retrieval_decision=retrieval_decision,
            )

        materialized = materialize_frozen_transition(
            frozen_transition, source_outcome=verifier_result
        )
        return BaselineExecutionOutcome(
            status="succeeded",
            final_response=solve_result.final_answer,
            parsed_answer=parsed_answer,
            verifier_result=verifier_result,
            answer_call_id=answer_call_id,
            method_calls=tuple(recorder.get_records()),
            memory_before=memory_before,
            memory_after=tuple(entry.model_dump() for entry in materialized.memory_after),
            retrieved_memory=_retrieved_memory(retrieval_decision),
            retrieved_scores=_retrieved_scores(retrieval_decision),
            memory_write_event=materialized.memory_write_event,
            metadata=metadata,
        )


def freeze_native_transition(
    *,
    payload: BoTTemplatePayload,
    buffer_snapshot: list[MemoryEntry],
    source_trial_id: str,
    embedding_provider: EmbeddingProvider,
    visible_entry_ids: list[str],
    config: dict[str, Any],
) -> FrozenNativeTransition:
    used_entry_ids = set(payload.explicitly_used_memory_ids)
    is_contaminated = any(
        entry.entry_id in used_entry_ids and entry.clean_or_contaminated == "contaminated"
        for entry in buffer_snapshot
    )
    candidate = build_template_entry(
        payload=payload,
        source_trial_id=source_trial_id,
        visible_entry_ids=visible_entry_ids,
        clean_or_contaminated="contaminated" if is_contaminated else "clean",
    )
    target_set = config.get("_logging_target_contamination_set") or config.get(
        "_logging_target_set_id"
    )
    if target_set:
        candidate.metadata.update(
            phase11_lineage_metadata(candidate, [*buffer_snapshot, candidate], target_set)
        )
    return FrozenNativeTransition(
        candidate=candidate,
        decision=evaluate_native_novelty(payload, buffer_snapshot, embedding_provider),
        memory_before=tuple(buffer_snapshot),
    )


def materialize_frozen_transition(
    frozen: FrozenNativeTransition, *, source_outcome: bool | None
) -> MaterializedNativeTransition:
    candidate = frozen.candidate.model_copy(
        update={"metadata": {**frozen.candidate.metadata, "source_outcome": source_outcome}}
    )
    admitted = frozen.decision.admitted
    memory_after = (*frozen.memory_before, candidate) if admitted else frozen.memory_before
    return MaterializedNativeTransition(
        memory_after=memory_after,
        memory_write_event={
            "event_type": "bot_write" if admitted else "bot_write_rejected",
            "baseline": "bot_style",
            "status": "accepted" if admitted else "rejected_novelty",
            "accepted": admitted,
            "parent_trial_id": candidate.source_trial_id,
            "source_trial_id": candidate.source_trial_id,
            "source_entry_ids": list(candidate.metadata["source_entry_ids"]),
            "direct_parent_ids": list(candidate.metadata["direct_parent_ids"]),
            "memory_support_ids": list(candidate.metadata["memory_support_ids"]),
            "candidate_entry_id": candidate.entry_id,
            "candidate_content": candidate.content,
            "top_existing_entry_id": frozen.decision.compared_entry_id,
            "top_similarity": frozen.decision.top_similarity,
            "new_entry_id": candidate.entry_id if admitted else None,
            "source_outcome": source_outcome,
        },
    )


def invalid_distillation_failure_outcome(
    *,
    recorder: MethodCallRecorder,
    memory_before: tuple[dict[str, Any], ...],
    metadata: dict[str, Any],
    answer_call_id: str | None,
    final_response: str,
    parsed_answer: str,
    retrieval_decision: BoTRetrievalDecision,
) -> BaselineExecutionOutcome:
    return _failure_outcome(
        recorder,
        memory_before,
        metadata,
        "bot_invalid_thought_distillation",
        answer_call_id,
        final_response=final_response,
        parsed_answer=parsed_answer,
        memory_write_event={
            "event_type": "bot_write_rejected",
            "baseline": "bot_style",
            "status": "rejected_invalid_distillation",
            "accepted": False,
            "new_entry_id": None,
            "source_outcome": None,
        },
        retrieval_decision=retrieval_decision,
    )


def _failure_outcome(
    recorder: MethodCallRecorder,
    memory_before: tuple[dict[str, Any], ...],
    metadata: dict[str, Any],
    failure_disposition: Literal[
        "bot_invalid_problem_distillation",
        "bot_invalid_solve_result",
        "bot_invalid_thought_distillation",
        "verifier_contract_failed",
    ],
    answer_call_id: str | None,
    *,
    final_response: str | None = None,
    parsed_answer: str | None = None,
    memory_after: tuple[MemoryEntry, ...] | None = None,
    memory_write_event: dict[str, Any] | None = None,
    retrieval_decision: BoTRetrievalDecision | None = None,
) -> BaselineExecutionOutcome:
    failure_triples: dict[FailureDisposition, tuple[ErrorType, ScientificIneligibilityReason]] = {
        "bot_invalid_problem_distillation": (
            "BaselineOutputError",
            "invalid_problem_distillation",
        ),
        "bot_invalid_solve_result": ("BaselineOutputError", "invalid_solve_result"),
        "bot_invalid_thought_distillation": (
            "BaselineOutputError",
            "invalid_thought_distillation",
        ),
        "verifier_contract_failed": (
            "VerifierContractError",
            "verifier_contract_failed",
        ),
    }
    error_type, scientific_ineligibility_reason = failure_triples[failure_disposition]
    return BaselineExecutionOutcome(
        status="failed",
        final_response=final_response,
        parsed_answer=parsed_answer,
        answer_call_id=answer_call_id,
        method_calls=tuple(recorder.get_records()),
        memory_before=memory_before,
        memory_after=(
            tuple(entry.model_dump() for entry in memory_after)
            if memory_after is not None
            else memory_before
        ),
        retrieved_memory=_retrieved_memory(retrieval_decision),
        retrieved_scores=_retrieved_scores(retrieval_decision),
        memory_write_event=memory_write_event,
        error_type=error_type,
        failure_disposition=failure_disposition,
        scientific_ineligibility_reason=scientific_ineligibility_reason,
        metadata=metadata,
    )


def _verify(verifier: Verifier | None, parsed_answer: str) -> bool:
    if verifier is None:
        return True
    result = verifier(parsed_answer)
    if isinstance(result, VerifierResult):
        return result.is_correct
    if isinstance(result, bool):
        return result
    raise TypeError("BoT verifier must return VerifierResult or bool")


def _tool_mode(config: dict[str, Any]) -> Literal["text_only", "python_sandbox"]:
    tool_mode = config.get("tool_mode", "text_only")
    if tool_mode not in {"text_only", "python_sandbox"}:
        raise ValueError("unsupported BoT tool mode")
    return tool_mode


def _text_call_config(config: dict[str, Any]) -> dict[str, Any]:
    forbidden_keys = {
        "tool_executor",
        "tool_runtime_contract",
        "max_tool_rounds",
        "_tool_event_writer",
    }
    return {key: value for key, value in config.items() if key not in forbidden_keys} | {
        "tool_mode": "text_only"
    }


def _tool_augmented_solve(
    *,
    identity: BotBufferIdentity,
    task: TaskInstance,
    distilled_problem: Any,
    retrieval_decision: BoTRetrievalDecision,
    recorder: MethodCallRecorder,
    model: str,
    config: dict[str, Any],
) -> tuple[str, str, tuple[Any, ...], list[dict[str, Any]]]:
    executor = config.get("tool_executor")
    policy = config.get("tool_runtime_contract")
    if executor is None or policy is None:
        raise ToolPolicyError("BOT_TOOL_CONTRACT_REQUIRED")
    if not isinstance(policy, ToolRuntimeContract):
        raise ToolPolicyError("BOT_TOOL_CONTRACT_REQUIRED")
    messages, source_spans = render_tool_augmented_bot_solve_messages(
        task, distilled_problem, retrieval_decision
    )
    solve_config = {
        **config,
        "sample_id": config.get("sample_id", task.sample_id),
        "method_stage": "bot_instantiate_solve",
        "_bot_retrieval_decision": retrieval_decision.decision,
        "source_spans": source_spans,
    }
    response = recorder.chat(messages, model, solve_config)
    initial_record = recorder.get_records()[-1]
    if initial_record.call_id is None:
        raise ToolProtocolError("MISSING_INITIAL_CALL")
    tool_result = run_tool_loop(
        LlmCall(
            call_id=initial_record.call_id,
            content=response.content,
            messages=messages,
            model=model,
            config=solve_config,
            run_id=identity.run_id,
            trial_id=_trial_id(identity, task),
            max_rounds=int(config.get("max_tool_rounds", 3)),
        ),
        recorder,
        executor,
        policy,
        writer=config.get("_tool_event_writer"),
    )
    return (
        tool_result.answer,
        tool_result.answer_call_id,
        tool_result.tool_events,
        _executed_trajectory(recorder, tool_result.tool_events),
    )


def _executed_trajectory(
    recorder: MethodCallRecorder, tool_events: tuple[Any, ...]
) -> list[dict[str, Any]]:
    calls = {call.call_id: call for call in recorder.get_records()}
    trajectory: list[dict[str, Any]] = []
    for event in tool_events:
        parent = calls.get(event.parent_call_id)
        if parent is None:
            raise ToolProtocolError("MISSING_EXECUTED_CODE")
        try:
            action = json.loads(parent.raw_response or "")
        except json.JSONDecodeError as error:
            raise ToolProtocolError("MISSING_EXECUTED_CODE") from error
        code = action.get("code") if action.get("action") == "execute_python" else None
        if not isinstance(code, str):
            raise ToolProtocolError("MISSING_EXECUTED_CODE")
        trajectory.append(
            {
                "code": code,
                "code_hash": event.code_hash,
                "exit_code": event.exit_code,
                "stderr": event.stderr,
                "stdout": event.output,
            }
        )
    return trajectory


def _trial_id(identity: BotBufferIdentity, task: TaskInstance) -> str:
    return ":".join(
        [
            identity.run_id,
            task.task_name,
            task.sample_id,
            identity.baseline,
            identity.arm,
            identity.backbone,
        ]
    )


def _retrieval_decision_metadata(decision: BoTRetrievalDecision) -> dict[str, Any]:
    return {
        "decision": decision.decision,
        "matched_entry_id": decision.matched_entry.entry_id if decision.matched_entry else None,
        "top_similarity": decision.top_similarity,
        "threshold": decision.threshold,
    }


def _retrieved_memory(decision: BoTRetrievalDecision | None) -> tuple[dict[str, Any], ...]:
    if decision is None or decision.matched_entry is None:
        return ()
    return (decision.matched_entry.model_dump(),)


def _retrieved_scores(decision: BoTRetrievalDecision | None) -> tuple[float, ...]:
    if decision is None or decision.decision != "matched" or decision.top_similarity is None:
        return ()
    return (decision.top_similarity,)
