from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import StringConstraints


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

SEMANTIC_STAGES = (
    "no_memory_generate",
    "full_history_generate",
    "rag_generate",
    "bot_problem_distill",
    "bot_instantiate_solve",
    "bot_thought_distill",
    "reflexion_generate",
    "reflexion_reflect",
)

ErrorType = Literal[
    "parser",
    "retrieval",
    "embedding",
    "corpus",
    "configuration",
    "provider",
    "verifier",
]
FailureDisposition = Literal[
    "no_memory_invalid_final_answer",
    "full_history_invalid_final_answer",
    "rag_invalid_final_answer",
    "rag_retrieval_failed",
    "rag_embedding_failed",
    "rag_manifest_invalid",
    "rag_embedding_dimension_mismatch",
    "rag_embedding_provider_unpinned",
    "bot_invalid_problem_distillation",
    "bot_invalid_solve_result",
    "bot_invalid_thought_distillation",
    "reflexion_invalid_generation",
    "reflexion_invalid_reflection",
    "provider_call_failed",
    "verifier_contract_failed",
]
ScientificIneligibilityReason = Literal[
    "invalid_final_answer",
    "invalid_intermediate_output",
    "retrieval_failure",
    "embedding_failure",
    "corpus_manifest_failure",
    "configuration_failure",
    "provider_failure",
    "verifier_contract_failure",
]

FAILURE_TAXONOMY: dict[FailureDisposition, tuple[ErrorType, ScientificIneligibilityReason]] = {
    "no_memory_invalid_final_answer": ("parser", "invalid_final_answer"),
    "full_history_invalid_final_answer": ("parser", "invalid_final_answer"),
    "rag_invalid_final_answer": ("parser", "invalid_final_answer"),
    "rag_retrieval_failed": ("retrieval", "retrieval_failure"),
    "rag_embedding_failed": ("embedding", "embedding_failure"),
    "rag_manifest_invalid": ("corpus", "corpus_manifest_failure"),
    "rag_embedding_dimension_mismatch": ("embedding", "embedding_failure"),
    "rag_embedding_provider_unpinned": ("configuration", "configuration_failure"),
    "bot_invalid_problem_distillation": ("parser", "invalid_intermediate_output"),
    "bot_invalid_solve_result": ("parser", "invalid_intermediate_output"),
    "bot_invalid_thought_distillation": ("parser", "invalid_intermediate_output"),
    "reflexion_invalid_generation": ("parser", "invalid_intermediate_output"),
    "reflexion_invalid_reflection": ("parser", "invalid_intermediate_output"),
    "provider_call_failed": ("provider", "provider_failure"),
    "verifier_contract_failed": ("verifier", "verifier_contract_failure"),
}
CANONICAL_FAILURE_MAPPING = FAILURE_TAXONOMY


def validate_failure_triple(
    error_type: ErrorType,
    failure_disposition: FailureDisposition,
    scientific_ineligibility_reason: ScientificIneligibilityReason,
) -> None:
    expected = FAILURE_TAXONOMY.get(failure_disposition)
    if expected is None:
        raise ValueError(f"unknown failure disposition: {failure_disposition!r}")
    if expected != (error_type, scientific_ineligibility_reason):
        raise ValueError(f"failure triple does not match {failure_disposition!r}")


@dataclass(frozen=True)
class StreamIdentity:
    run_id: NonEmptyStr
    task_name: NonEmptyStr
    baseline: NonEmptyStr
    arm: NonEmptyStr
    backbone: NonEmptyStr


@dataclass(frozen=True)
class StreamPairKey:
    run_id: NonEmptyStr
    task_name: NonEmptyStr
    baseline: NonEmptyStr
    backbone: NonEmptyStr


def stream_pair_key(identity: StreamIdentity) -> StreamPairKey:
    return StreamPairKey(
        identity.run_id,
        identity.task_name,
        identity.baseline,
        identity.backbone,
    )


@dataclass(frozen=True)
class CorpusIdentity:
    corpus_hash: NonEmptyStr
    embedding_model_id: NonEmptyStr
    embedding_revision: NonEmptyStr
    embedding_library_version: NonEmptyStr


@dataclass
class BaselineExecutionOutcome:
    status: Literal["succeeded", "failed"]
    final_response: str | None = None
    parsed_answer: str | None = None
    verifier_result: Any | None = None
    answer_call_id: str | None = None
    method_calls: list[Any] = field(default_factory=list)
    memory_before: list[dict[str, Any]] = field(default_factory=list)
    memory_after: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memory: list[dict[str, Any]] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    memory_write_event: dict[str, Any] | None = None
    error_type: ErrorType | None = None
    failure_disposition: FailureDisposition | None = None
    scientific_ineligibility_reason: ScientificIneligibilityReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        failure_values = (
            self.error_type,
            self.failure_disposition,
            self.scientific_ineligibility_reason,
        )
        if self.status == "succeeded":
            if any(value is not None for value in failure_values):
                raise ValueError("succeeded outcome cannot carry failure evidence")
            return
        if any(value is not None for value in failure_values) and not all(
            value is not None for value in failure_values
        ):
            raise ValueError("failed outcome requires a complete failure triple")
        if (
            self.error_type is not None
            and self.failure_disposition is not None
            and self.scientific_ineligibility_reason is not None
        ):
            validate_failure_triple(
                self.error_type,
                self.failure_disposition,
                self.scientific_ineligibility_reason,
            )


@dataclass(frozen=True)
class ProviderCallFailure:
    error_type: ErrorType
    message: str | None = None
    failure_disposition: FailureDisposition = "provider_call_failed"
    scientific_ineligibility_reason: ScientificIneligibilityReason = "provider_failure"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_failure_triple(
            self.error_type,
            self.failure_disposition,
            self.scientific_ineligibility_reason,
        )


@dataclass(frozen=True)
class ReflexionAttemptOutcome:
    attempt_id: NonEmptyStr
    attempt_index: int
    answer_call_id: str | None
    outcome: BaselineExecutionOutcome


@dataclass(frozen=True)
class ReflexionReflectionEvent:
    attempt_id: NonEmptyStr
    reflection_call_id: NonEmptyStr
    reflection_entry_id: NonEmptyStr
