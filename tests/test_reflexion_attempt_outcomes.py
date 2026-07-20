from __future__ import annotations

import importlib

import pytest

from memcontam.baselines.contracts import BaselineExecutionOutcome, ReflexionAttemptOutcome


def test_reflexion_attempt_outcomes_join_calls_without_using_transport_retry_count() -> None:
    validation = importlib.import_module("memcontam.logging.validation")
    validate = getattr(validation, "validate_reflexion_attempt_records", None)
    assert callable(validate)

    attempt = ReflexionAttemptOutcome(
        attempt_id="attempt-1",
        attempt_index=1,
        answer_call_id="answer-1",
        outcome=BaselineExecutionOutcome(status="succeeded", verifier_result=False),
    )
    validate([attempt], [{"call_id": "answer-1", "retry_count": 9}])


def test_reflexion_attempt_indices_are_never_inferred_from_transport_retries() -> None:
    validation = importlib.import_module("memcontam.logging.validation")
    validate = getattr(validation, "validate_reflexion_attempt_records", None)
    assert callable(validate)

    attempt = ReflexionAttemptOutcome(
        attempt_id="attempt-1",
        attempt_index=1,
        answer_call_id="answer-1",
        outcome=BaselineExecutionOutcome(status="succeeded", verifier_result=False),
    )
    with pytest.raises(ValueError, match="attempt_index"):
        validate([attempt], [{"call_id": "answer-1"}], retry_count=1)
