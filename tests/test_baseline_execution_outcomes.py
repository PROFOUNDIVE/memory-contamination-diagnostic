from __future__ import annotations

import importlib
import importlib.util

import pytest

from memcontam.baselines.contracts import BaselineExecutionOutcome


def test_baseline_execution_outcome_retains_failed_evidence_and_valid_incorrect_success() -> None:
    assert importlib.util.find_spec("memcontam.baselines.contracts"), (
        "BaselineExecutionOutcome must live in the shared contracts module"
    )
    contracts = importlib.import_module("memcontam.baselines.contracts")

    outcome = getattr(contracts, "BaselineExecutionOutcome", None)
    assert outcome is not None
    assert getattr(outcome, "__dataclass_fields__", {}).keys() >= {
        "status",
        "final_response",
        "parsed_answer",
        "verifier_result",
        "answer_call_id",
        "method_calls",
        "memory_before",
        "memory_after",
        "retrieved_memory",
        "retrieved_scores",
        "memory_write_event",
        "error_type",
        "failure_disposition",
        "metadata",
    }

    incorrect = outcome(status="succeeded", verifier_result=False)
    assert incorrect.verifier_result is False
    assert incorrect.error_type is None
    assert incorrect.failure_disposition is None

    with pytest.raises(ValueError, match="succeeded"):
        outcome(status="succeeded", error_type="ProviderCallFailure")

    with pytest.raises(ValueError, match="complete failure triple"):
        outcome(status="failed")

    failed = outcome(
        status="failed",
        method_calls=(),
        memory_before=(),
        memory_after=(),
        retrieved_memory=(),
        retrieved_scores=(),
        error_type="ProviderCallFailure",
        failure_disposition="provider_call_failed",
        scientific_ineligibility_reason="provider_call_failed",
    )
    assert failed.method_calls == ()
    assert failed.memory_before == ()
    with pytest.raises(ValueError, match="failure triple"):
        outcome(
            status="failed",
            error_type="BaselineOutputError",
            failure_disposition="provider_call_failed",
            scientific_ineligibility_reason="provider_call_failed",
        )

    provider_failure = contracts.ProviderCallFailure(error_type="ProviderCallFailure")
    assert provider_failure.failure_disposition == "provider_call_failed"
    with pytest.raises(ValueError, match="failure triple"):
        contracts.ProviderCallFailure(error_type="BaselineOutputError")


def test_failed_outcome_requires_scientific_ineligibility_metadata() -> None:
    validation = importlib.import_module("memcontam.logging.validation")
    validate = getattr(validation, "validate_outcome_metadata", None)
    assert callable(validate)

    failed = BaselineExecutionOutcome(
        status="failed",
        error_type="ProviderCallFailure",
        failure_disposition="provider_call_failed",
        scientific_ineligibility_reason="provider_call_failed",
    )
    with pytest.raises(ValueError, match="scientific_ineligibility_reason"):
        validate(failed, {})
