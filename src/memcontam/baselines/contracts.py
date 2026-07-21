from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import StringConstraints


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

BASELINE_FIDELITY_V2 = "baseline_fidelity_v2"
BASELINE_EXECUTION_CONTRACT_V2 = BASELINE_FIDELITY_V2
FAILURE_TAXONOMY_V2 = BASELINE_FIDELITY_V2

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
    "BaselineOutputError",
    "RetrievalContractError",
    "EmbeddingContractError",
    "CorpusContractError",
    "ProviderCallFailure",
    "VerifierContractError",
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
    "retrieval_failed",
    "embedding_failed",
    "manifest_invalid",
    "embedding_dimension_mismatch",
    "embedding_provider_unpinned",
    "invalid_problem_distillation",
    "invalid_solve_result",
    "invalid_thought_distillation",
    "invalid_reflexion_generation",
    "invalid_reflection",
    "provider_call_failed",
    "verifier_contract_failed",
]

FAILURE_TAXONOMY: dict[FailureDisposition, tuple[ErrorType, ScientificIneligibilityReason]] = {
    "no_memory_invalid_final_answer": ("BaselineOutputError", "invalid_final_answer"),
    "full_history_invalid_final_answer": ("BaselineOutputError", "invalid_final_answer"),
    "rag_invalid_final_answer": ("BaselineOutputError", "invalid_final_answer"),
    "rag_retrieval_failed": ("RetrievalContractError", "retrieval_failed"),
    "rag_embedding_failed": ("EmbeddingContractError", "embedding_failed"),
    "rag_manifest_invalid": ("CorpusContractError", "manifest_invalid"),
    "rag_embedding_dimension_mismatch": (
        "EmbeddingContractError",
        "embedding_dimension_mismatch",
    ),
    "rag_embedding_provider_unpinned": (
        "EmbeddingContractError",
        "embedding_provider_unpinned",
    ),
    "bot_invalid_problem_distillation": (
        "BaselineOutputError",
        "invalid_problem_distillation",
    ),
    "bot_invalid_solve_result": ("BaselineOutputError", "invalid_solve_result"),
    "bot_invalid_thought_distillation": (
        "BaselineOutputError",
        "invalid_thought_distillation",
    ),
    "reflexion_invalid_generation": (
        "BaselineOutputError",
        "invalid_reflexion_generation",
    ),
    "reflexion_invalid_reflection": ("BaselineOutputError", "invalid_reflection"),
    "provider_call_failed": ("ProviderCallFailure", "provider_call_failed"),
    "verifier_contract_failed": ("VerifierContractError", "verifier_contract_failed"),
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


@dataclass(frozen=True, order=True)
class StreamIdentity:
    run_id: str
    task_family: str
    baseline: Literal["full_history", "bot_style", "reflexion_style"]
    arm: Literal["clean", "contaminated", "contaminated_filter"]
    backbone: str


@dataclass(frozen=True, order=True)
class StreamPairKey:
    run_id: str
    task_family: str
    baseline: Literal["full_history", "bot_style", "reflexion_style"]
    backbone: str


def stream_pair_key(identity: StreamIdentity) -> StreamPairKey:
    return StreamPairKey(
        identity.run_id,
        identity.task_family,
        identity.baseline,
        identity.backbone,
    )


@dataclass(frozen=True, order=True)
class CorpusIdentity:
    manifest_id: str
    corpus_version: str
    task_family: str
    embedding_provider_identity: str


@dataclass(frozen=True)
class BaselineExecutionOutcome:
    status: Literal["succeeded", "failed"]
    final_response: str | None = None
    parsed_answer: str | None = None
    verifier_result: Any | None = None
    answer_call_id: str | None = None
    method_calls: tuple[Any, ...] = ()
    memory_before: tuple[dict[str, Any], ...] = ()
    memory_after: tuple[dict[str, Any], ...] = ()
    retrieved_memory: tuple[dict[str, Any], ...] = ()
    retrieved_scores: tuple[float, ...] = ()
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
        if not all(value is not None for value in failure_values):
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
    error_type: Literal["ProviderCallFailure"] = "ProviderCallFailure"
    message: str | None = None
    failure_disposition: FailureDisposition = "provider_call_failed"
    scientific_ineligibility_reason: ScientificIneligibilityReason = "provider_call_failed"
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
