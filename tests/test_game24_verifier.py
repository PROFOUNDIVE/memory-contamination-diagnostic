from memcontam.readiness.phase13_main_live_runtime_support import verifier
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.game24 import verify_expression


def test_verify_expression_accepts_valid_game24_solution() -> None:
    result = verify_expression("6 / (1 - 3 / 4)", [1, 3, 4, 6])

    assert result.is_correct is True
    assert result.parsed_answer == "6 / (1 - 3 / 4)"
    assert result.metadata["value"] == 24


def test_verify_expression_rejects_reusing_a_number() -> None:
    result = verify_expression("6 / (1 - 3 / 3)", [1, 3, 4, 6])

    assert result.is_correct is False
    assert result.reason == "numbers_used_do_not_match"


def test_verify_expression_rejects_wrong_target_value() -> None:
    result = verify_expression("1 + 3 + 4 + 6", [1, 3, 4, 6])

    assert result.is_correct is False
    assert result.reason == "value_does_not_match_target"


def test_verify_expression_rejects_unsafe_or_unparseable_input() -> None:
    result = verify_expression("__import__('os').system('true')", [1, 3, 4, 6])

    assert result.is_correct is False
    assert result.reason == "unsupported_expression"


def test_verify_expression_rejects_boolean_constants() -> None:
    result = verify_expression("True + 3 + 4 + 16", [1, 3, 4, 16])

    assert result.is_correct is False
    assert result.reason == "unsupported_expression"


def test_verify_expression_rejects_overlong_expression() -> None:
    result = verify_expression("1" * 501, [1, 3, 4, 6])

    assert result.is_correct is False
    assert result.reason == "unsupported_expression"


def test_decimal_occurrence_cannot_masquerade_as_supplied_number() -> None:
    expression = "6/(1.0000000000000001-3/4)"
    assert verify_expression(expression, [1, 3, 4, 6]).is_correct is False
    assert verify_expression("6/(1.0-3/4)", [1, 3, 4, 6]).is_correct is False


def test_exact_rational_target_is_required() -> None:
    assert verify_expression("6/(1-3/4)", [1, 3, 4, 6]).is_correct is True
    assert verify_expression("6/(1-3/4)+0", [1, 3, 4, 6]).is_correct is False


def test_main_verifier_uses_exact_game24_semantics() -> None:
    instance = TaskInstance(sample_id="synthetic", task_name="game24", input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24})
    main_verify = verifier("game24")
    assert main_verify("6/(1.0000000000000001-3/4)", instance) is False
    assert main_verify("6/(1-3/4)", instance) is True
