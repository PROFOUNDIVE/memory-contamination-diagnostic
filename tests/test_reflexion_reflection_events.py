from __future__ import annotations

import importlib

import pytest

from memcontam.baselines.contracts import (
    BaselineExecutionOutcome,
    ReflexionAttemptOutcome,
    ReflexionReflectionEvent,
)


def test_reflexion_reflection_events_join_only_authenticated_incorrect_attempts() -> None:
    validation = importlib.import_module("memcontam.logging.validation")
    validate = getattr(validation, "validate_reflexion_reflection_events", None)
    assert callable(validate)

    attempt = ReflexionAttemptOutcome(
        attempt_id="attempt-1",
        attempt_index=1,
        answer_call_id="answer-1",
        outcome=BaselineExecutionOutcome(status="succeeded", verifier_result=False),
    )
    event = ReflexionReflectionEvent("attempt-1", "reflect-1", "memory-1")

    validate([attempt], [event], [{"call_id": "reflect-1"}], [{"entry_id": "memory-1"}])


def test_reflexion_reflection_rejects_unknown_attempt_reference() -> None:
    validation = importlib.import_module("memcontam.logging.validation")
    validate = getattr(validation, "validate_reflexion_reflection_events", None)
    assert callable(validate)

    with pytest.raises(ValueError, match="unknown attempt"):
        validate(
            [],
            [ReflexionReflectionEvent("missing", "reflect-1", "memory-1")],
            [{"call_id": "reflect-1"}],
            [{"entry_id": "memory-1"}],
        )
