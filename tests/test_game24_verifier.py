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
