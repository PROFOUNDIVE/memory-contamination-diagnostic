from __future__ import annotations

import pytest

from memcontam.logging.schema import VerifierResult
from memcontam.readiness.phase13_main_live_runtime_support import verifier
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.math_equation_balancer import verify_answer


def _task(input_text: str, target: str, target_value: int) -> TaskInstance:
    return TaskInstance(
        sample_id="pre-authority-meb",
        task_name="math_equation_balancer",
        input={"input": input_text},
        verifier_spec={"target": target, "target_value": target_value},
    )


@pytest.mark.parametrize(
    ("answer", "expected_correct", "expected_reason"),
    (
        ("2 + 3 * 4 = 14", True, "ok"),
        ("2 * 2 - 2 = 2", True, "ok"),
        ("14", False, "malformed_answer"),
        ("2 + 3 * 4 = 20", False, "wrong_answer"),
        ("2 ** 3 + 4 = 12", False, "malformed_answer"),
        ("4 * 3 + 2 = 14", False, "wrong_answer"),
    ),
)
def test_authority_compatible_meb_verifier_covers_fixed_answer_classes(
    answer: str,
    expected_correct: bool,
    expected_reason: str,
) -> None:
    task = (
        _task("2 ? 2 ? 2 = 2", "2 + 2 - 2 = 2", 2)
        if answer == "2 * 2 - 2 = 2"
        else _task("2 ? 3 ? 4 = 14", "2 + 3 * 4 = 14", 14)
    )

    result = verify_answer(answer, task)

    assert result == VerifierResult(
        is_correct=expected_correct,
        parsed_answer=answer,
        reason=expected_reason,
        metadata={"target": task.verifier_spec["target"], "target_value": task.verifier_spec["target_value"]},
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="A-02 expected current defect: production dispatch uses the historical RHS verifier",
)
def test_production_meb_dispatch_accepts_alternative_valid_operator_assignment() -> None:
    task = _task("2 ? 2 ? 2 = 2", "2 + 2 - 2 = 2", 2)

    assert verifier("math_equation_balancer")("2 * 2 - 2 = 2", task) is True


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="A-02 expected current defect: production dispatch accepts the bare target value",
)
def test_production_meb_dispatch_rejects_bare_numeric_target() -> None:
    task = _task("2 ? 3 ? 4 = 14", "2 + 3 * 4 = 14", 14)

    assert verifier("math_equation_balancer")("14", task) is False
