from importlib import import_module

from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.word_sorting import verify_words

_meb_verifier = import_module("memcontam.verifiers.math_equation_balancer")
verify_answer = getattr(_meb_verifier, "verify_answer")
verify_rhs_completion_answer = getattr(_meb_verifier, "verify_rhs_completion_answer")


def _meb_task(input_text: str, target: str, target_value: int) -> TaskInstance:
    return TaskInstance(
        sample_id="meb-test",
        task_name="math_equation_balancer",
        input={"input": input_text},
        verifier_spec={"target": target, "target_value": target_value},
    )


def test_verify_answer_accepts_correct_equation_string() -> None:
    task = _meb_task("2 ? 5 = 7", "2 + 5 = 7", 7)

    result = verify_answer("2 + 5 = 7", task)

    assert result.is_correct is True
    assert result.parsed_answer == "2 + 5 = 7"
    assert result.reason == "ok"
    assert result.metadata == {"target": "2 + 5 = 7", "target_value": 7}


def test_verify_answer_rejects_bare_target_value() -> None:
    task = _meb_task("2 ? 5 = 7", "2 + 5 = 7", 7)

    result = verify_answer("7", task)

    assert result.is_correct is False
    assert result.parsed_answer == "7"
    assert result.reason == "malformed_answer"
    assert result.metadata == {"target": "2 + 5 = 7", "target_value": 7}


def test_verify_answer_accepts_alternate_valid_operator_assignment() -> None:
    task = _meb_task("2 ? 2 ? 2 = 2", "2 + 2 - 2 = 2", 2)

    result = verify_answer("2 * 2 - 2 = 2", task)

    assert result.is_correct is True
    assert result.parsed_answer == "2 * 2 - 2 = 2"
    assert result.reason == "ok"


def test_verify_answer_rejects_reordered_operands() -> None:
    task = _meb_task("2 ? 3 ? 4 = 14", "2 + 3 * 4 = 14", 14)

    result = verify_answer("4 * 3 + 2 = 14", task)

    assert result.is_correct is False
    assert result.reason == "wrong_answer"


def test_verify_answer_rejects_illegal_operator() -> None:
    task = _meb_task("2 ? 3 ? 4 = 10", "2 * 3 + 4 = 10", 10)

    result = verify_answer("2 ** 3 + 4 = 12", task)

    assert result.is_correct is False
    assert result.reason == "malformed_answer"


def test_verify_answer_uses_exact_precedence_arithmetic() -> None:
    task = _meb_task("2 ? 3 ? 4 = 14", "2 + 3 * 4 = 14", 14)

    correct = verify_answer("2 + 3 * 4 = 14", task)
    left_to_right = verify_answer("2 + 3 * 4 = 20", task)

    assert correct.is_correct is True
    assert left_to_right.is_correct is False
    assert left_to_right.reason == "wrong_answer"


def test_verify_answer_rejects_wrong_equation() -> None:
    task = _meb_task("2 ? 5 = 7", "2 + 5 = 7", 7)

    result = verify_answer("2 + 5 = 8", task)

    assert result.is_correct is False
    assert result.reason == "wrong_answer"
    assert result.metadata == {"target": "2 + 5 = 7", "target_value": 7}


def test_verify_answer_rejects_malformed_or_empty_answer() -> None:
    task = _meb_task("2 ? 5 = 7", "2 + 5 = 7", 7)

    empty_result = verify_answer("   ", task)
    non_string_result = verify_answer(None, task)  # type: ignore[arg-type]

    assert empty_result.is_correct is False
    assert empty_result.reason == "malformed_answer"
    assert empty_result.metadata == {"detail": "answer is empty"}

    assert non_string_result.is_correct is False
    assert non_string_result.reason == "malformed_answer"
    assert non_string_result.metadata == {"detail": "answer is not a string"}


def test_verify_answer_rejects_input_and_verifier_spec_mismatch() -> None:
    task = _meb_task("2 ? 3 = 5", "7 + 8 = 15", 15)

    result = verify_answer("7 + 8 = 15", task)

    assert result.is_correct is False
    assert result.reason == "wrong_answer"


def test_verify_answer_is_left_associative_for_equal_precedence() -> None:
    task = _meb_task("8 ? 4 ? 2 = 1", "8 / 4 / 2 = 1", 1)

    result = verify_answer("8 / 4 / 2 = 1", task)

    assert result.is_correct is True


def test_verify_answer_rejects_division_by_zero() -> None:
    task = _meb_task("1 ? 0 = 0", "1 * 0 = 0", 0)

    result = verify_answer("1 / 0 = 0", task)

    assert result.is_correct is False


def test_verify_answer_accepts_exact_fractional_intermediates() -> None:
    task = _meb_task("3 ? 2 ? 1 ? 2 = 2", "3 / 2 + 1 / 2 = 2", 2)

    result = verify_answer("3 / 2 + 1 / 2 = 2", task)

    assert result.is_correct is True


def test_historical_rhs_completion_verifier_remains_separate() -> None:
    spec = {"target": "2 + 5 = 7", "target_value": 7}

    result = verify_rhs_completion_answer("7", spec)

    assert result.is_correct is True


def test_verify_words_accepts_correct_word_list() -> None:
    result = verify_words(["apple", "banana", "pear"], ["apple", "banana", "pear"])

    assert result.is_correct is True
    assert result.parsed_answer == "apple banana pear"
    assert result.reason == "ok"
    assert result.metadata == {}


def test_verify_words_rejects_wrong_order() -> None:
    result = verify_words(["pear", "banana", "apple"], ["apple", "banana", "pear"])

    assert result.is_correct is False
    assert result.reason == "wrong_order"
    assert result.metadata == {
        "expected": ["apple", "banana", "pear"],
        "actual": ["pear", "banana", "apple"],
    }


def test_verify_words_rejects_malformed_non_list() -> None:
    result = verify_words("apple banana pear", ["apple", "banana", "pear"])  # type: ignore[arg-type]

    assert result.is_correct is False
    assert result.reason == "malformed_answer"
    assert result.metadata == {"detail": "answer_words is not a non-empty list"}


def test_verify_words_rejects_empty_list() -> None:
    result = verify_words([], ["apple", "banana", "pear"])

    assert result.is_correct is False
    assert result.reason == "malformed_answer"
    assert result.metadata == {"detail": "answer_words is not a non-empty list"}


def test_verify_words_rejects_non_string_element() -> None:
    result = verify_words(["apple", None, "pear"], ["apple", "banana", "pear"])  # type: ignore[list-item]

    assert result.is_correct is False
    assert result.reason == "malformed_answer"
    assert result.metadata == {"detail": "answer_words contains non-string tokens"}
