from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Literal

from .bot_read import retrieve_top_template
from .bot_solve import parse_bot_solve_result
from .bot_style import BotStylePolicy
from .common import parse_final_answer
from .contracts import BaselineExecutionOutcome
from memcontam.clients.base import LLMClient
from memcontam.clients.recording import MethodCallRecorder
from memcontam.logging.schema import VerifierResult
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.embeddings import FakeEmbeddingProvider
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


Verifier = Callable[[str], VerifierResult]


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
        del verifier
        call_config = {**config, "sample_id": config.get("sample_id", task.sample_id)}
        recorder = MethodCallRecorder(client)
        memory_before = tuple(entry.model_dump() for entry in buffer_snapshot)
        metadata = {"bot_buffer_identity": asdict(identity)}

        try:
            distilled = self.policy.problem_distillation(task, recorder, model, call_config)
        except ValueError:
            return self._failure_outcome(
                recorder, memory_before, metadata, "bot_invalid_problem_distillation", None
            )

        metadata["distilled_problem"] = distilled.model_dump()
        retrieved = retrieve_top_template(
            distilled,
            buffer_snapshot,
            call_config.get("embedding_provider", FakeEmbeddingProvider()),
        )
        raw_solve = self.policy.template_instantiation_solve(
            task, distilled, recorder, model, call_config, retrieved=retrieved
        )
        answer_call_id = recorder.get_records()[-1].call_id
        try:
            solve_result = parse_bot_solve_result(raw_solve)
        except ValueError:
            return self._failure_outcome(
                recorder,
                memory_before,
                metadata,
                "bot_invalid_solve_result",
                answer_call_id,
                raw_solve,
            )

        metadata["solution_trace"] = solve_result.solution_trace
        return BaselineExecutionOutcome(
            status="succeeded",
            final_response=solve_result.final_answer,
            parsed_answer=parse_final_answer(solve_result.final_answer),
            answer_call_id=answer_call_id,
            method_calls=tuple(recorder.get_records()),
            memory_before=memory_before,
            memory_after=memory_before,
            retrieved_memory=(retrieved["memory_entry"].model_dump(),) if retrieved else (),
            retrieved_scores=(retrieved["score"],) if retrieved else (),
            metadata=metadata,
        )

    @staticmethod
    def _failure_outcome(
        recorder: MethodCallRecorder,
        memory_before: tuple[dict[str, Any], ...],
        metadata: dict[str, Any],
        failure_disposition: Literal[
            "bot_invalid_problem_distillation", "bot_invalid_solve_result"
        ],
        answer_call_id: str | None,
        final_response: str | None = None,
    ) -> BaselineExecutionOutcome:
        reason: Literal["invalid_problem_distillation", "invalid_solve_result"] = (
            "invalid_problem_distillation"
            if failure_disposition == "bot_invalid_problem_distillation"
            else "invalid_solve_result"
        )
        return BaselineExecutionOutcome(
            status="failed",
            final_response=final_response,
            answer_call_id=answer_call_id,
            method_calls=tuple(recorder.get_records()),
            memory_before=memory_before,
            memory_after=memory_before,
            error_type="BaselineOutputError",
            failure_disposition=failure_disposition,
            scientific_ineligibility_reason=reason,
            metadata=metadata,
        )
