from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.logging.schema import VerifierResult


ExecutionStatus = Literal["completed", "invalidated"]
FailureClass = Literal[
    "none",
    "model_behavior",
    "provider_api",
    "infrastructure",
    "verifier",
    "protocol",
    "implementation",
]
AnalysisInclusion = Literal["included", "excluded_prespecified"]
ParseStatus = Literal["parsed", "invalid", "not_produced"]


class OutcomeClassificationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class TrialOutcomeV3:
    parse_status: ParseStatus
    execution_status: ExecutionStatus
    failure_class: FailureClass
    analysis_inclusion: AnalysisInclusion
    verified_score: Literal[0, 1] | None
    verifier_result: VerifierResult | None
    raw_outcome: BaselineExecutionOutcome
    inclusion_reason: str

    def __post_init__(self) -> None:
        completed = (
            self.execution_status == "completed"
            and self.failure_class in {"none", "model_behavior"}
            and self.analysis_inclusion == "included"
            and self.verifier_result is not None
            and self.verified_score == int(self.verifier_result.is_correct)
        )
        invalidated = (
            self.execution_status == "invalidated"
            and self.failure_class
            in {"provider_api", "infrastructure", "verifier", "protocol", "implementation"}
            and self.analysis_inclusion == "excluded_prespecified"
            and self.verified_score is None
            and self.verifier_result is None
        )
        if not completed and not invalidated:
            raise OutcomeClassificationError("ILLEGAL_FAILURE_INCLUSION_PAIR")


def _classify_failure_stage(raw: BaselineExecutionOutcome) -> FailureClass:
    explicit_class = raw.metadata.get("phase12_failure_class", raw.metadata.get("failure_class"))
    if explicit_class in {"protocol", "implementation"}:
        return explicit_class
    if raw.error_type == "BaselineOutputError":
        return "model_behavior"
    if raw.error_type == "ProviderCallFailure":
        return "provider_api"
    if raw.error_type in {"RetrievalContractError", "EmbeddingContractError", "CorpusContractError"}:
        return "infrastructure"
    if raw.error_type == "VerifierContractError":
        return "verifier"
    raise OutcomeClassificationError("ILLEGAL_FAILURE_INCLUSION_PAIR")


def _parse_status(raw: BaselineExecutionOutcome) -> ParseStatus:
    if raw.parsed_answer:
        return "parsed"
    if raw.final_response is None:
        return "not_produced"
    return "invalid"


def _included(
    raw: BaselineExecutionOutcome,
    failure_class: Literal["none", "model_behavior"],
    verifier: VerifierResult | None,
) -> TrialOutcomeV3:
    if raw.parsed_answer:
        if verifier is None:
            raise OutcomeClassificationError("MISSING_VERIFIER_FOR_PARSED_OUTPUT")
        result = verifier
    else:
        result = VerifierResult(is_correct=False, reason="no_valid_answer")
    return TrialOutcomeV3(
        parse_status=_parse_status(raw),
        execution_status="completed",
        failure_class=failure_class,
        analysis_inclusion="included",
        verified_score=1 if result.is_correct else 0,
        verifier_result=result,
        raw_outcome=raw,
        inclusion_reason="verified_answer" if failure_class == "none" else "model_behavior",
    )


def classify_baseline_outcome(
    raw: BaselineExecutionOutcome, verifier: VerifierResult | None
) -> TrialOutcomeV3:
    if raw.status == "succeeded":
        return _included(raw, "none" if raw.parsed_answer else "model_behavior", verifier)

    failure_class = _classify_failure_stage(raw)
    if failure_class == "model_behavior":
        return _included(raw, failure_class, verifier)
    return TrialOutcomeV3(
        parse_status=_parse_status(raw),
        execution_status="invalidated",
        failure_class=failure_class,
        analysis_inclusion="excluded_prespecified",
        verified_score=None,
        verifier_result=None,
        raw_outcome=raw,
        inclusion_reason=f"prespecified_{failure_class}_invalidation",
    )
