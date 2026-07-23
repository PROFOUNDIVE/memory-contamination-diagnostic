from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.experiment.phase12.outcomes import (
    OutcomeClassificationError,
    TrialOutcomeV3,
    classify_baseline_outcome,
)
from memcontam.logging.schema import VerifierResult


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-OUTCOME-001.json"


def _outcome_for_case(case: dict[str, object]) -> tuple[BaselineExecutionOutcome, VerifierResult | None]:
    case_id = case["id"]
    if case_id == "correct" or case_id == "valid_incorrect":
        verifier = VerifierResult(is_correct=bool(case["verifier"]), parsed_answer="answer")
        return BaselineExecutionOutcome(
            status="succeeded",
            final_response="final: answer",
            parsed_answer="answer",
        ), verifier
    if case_id == "malformed_final":
        return (
            BaselineExecutionOutcome(
                status="failed",
                final_response="not a final answer",
                error_type="BaselineOutputError",
                failure_disposition="no_memory_invalid_final_answer",
                scientific_ineligibility_reason="invalid_final_answer",
            ),
            None,
        )
    if case_id == "malformed_updater":
        return (
            BaselineExecutionOutcome(
                status="failed",
                final_response="final: answer",
                parsed_answer="answer",
                memory_write_event={"status": "rejected_invalid_distillation"},
                error_type="BaselineOutputError",
                failure_disposition="bot_invalid_thought_distillation",
                scientific_ineligibility_reason="invalid_thought_distillation",
            ),
            VerifierResult(is_correct=True, parsed_answer="answer"),
        )
    if case_id == "provider_loss":
        return (
            BaselineExecutionOutcome(
                status="failed",
                error_type="ProviderCallFailure",
                failure_disposition="provider_call_failed",
                scientific_ineligibility_reason="provider_call_failed",
            ),
            None,
        )
    raise AssertionError(f"unhandled fixture case: {case_id}")


def test_classifies_registered_outcome_examples() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in fixture["cases"]:
        raw, verifier = _outcome_for_case(case)

        classified = classify_baseline_outcome(raw, verifier)

        assert (
            classified.execution_status,
            classified.failure_class,
            classified.analysis_inclusion,
            classified.verified_score,
        ) == tuple(case["expected"])
        assert classified.raw_outcome is raw


def test_rejects_illegal_failure_inclusion_pairs() -> None:
    with pytest.raises(OutcomeClassificationError, match="ILLEGAL_FAILURE_INCLUSION_PAIR"):
        TrialOutcomeV3(
            parse_status="not_produced",
            execution_status="invalidated",
            failure_class="provider_api",
            analysis_inclusion="included",
            verified_score=None,
            verifier_result=None,
            raw_outcome=BaselineExecutionOutcome(
                status="failed",
                error_type="ProviderCallFailure",
                failure_disposition="provider_call_failed",
                scientific_ineligibility_reason="provider_call_failed",
            ),
            inclusion_reason="prespecified_provider_api_invalidation",
        )

    with pytest.raises(OutcomeClassificationError, match="ILLEGAL_FAILURE_INCLUSION_PAIR"):
        TrialOutcomeV3(
            parse_status="invalid",
            execution_status="completed",
            failure_class="model_behavior",
            analysis_inclusion="excluded_prespecified",
            verified_score=0,
            verifier_result=VerifierResult(is_correct=False),
            raw_outcome=BaselineExecutionOutcome(
                status="failed",
                error_type="BaselineOutputError",
                failure_disposition="no_memory_invalid_final_answer",
                scientific_ineligibility_reason="invalid_final_answer",
            ),
            inclusion_reason="model_behavior",
        )
