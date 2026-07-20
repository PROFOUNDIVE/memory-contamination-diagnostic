from __future__ import annotations

import importlib
import importlib.util

import pytest


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
        outcome(status="succeeded", error_type="provider")
