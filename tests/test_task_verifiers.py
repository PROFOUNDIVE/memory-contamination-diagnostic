from memcontam.verifiers.math_equation_balancer import verify_answer
from memcontam.verifiers.word_sorting import verify_words


def test_verify_answer_accepts_correct_equation_string() -> None:
    spec = {"target": "2 + 5 = 7", "target_value": 7}

    result = verify_answer("2 + 5 = 7", spec)

    assert result.is_correct is True
    assert result.parsed_answer == "2 + 5 = 7"
    assert result.reason == "ok"
    assert result.metadata == {"target": "2 + 5 = 7", "target_value": 7}


def test_verify_answer_accepts_correct_numeric_only_answer() -> None:
    spec = {"target": "2 + 5 = 7", "target_value": 7}

    result = verify_answer("7", spec)

    assert result.is_correct is True
    assert result.parsed_answer == "7"
    assert result.reason == "ok"
    assert result.metadata == {"target": "2 + 5 = 7", "target_value": 7}


def test_verify_answer_rejects_wrong_equation() -> None:
    spec = {"target": "2 + 5 = 7", "target_value": 7}

    result = verify_answer("2 + 5 = 8", spec)

    assert result.is_correct is False
    assert result.reason == "wrong_answer"
    assert result.metadata == {"target": "2 + 5 = 7", "target_value": 7}


def test_verify_answer_rejects_malformed_or_empty_answer() -> None:
    spec = {"target": "2 + 5 = 7", "target_value": 7}

    empty_result = verify_answer("   ", spec)
    non_string_result = verify_answer(None, spec)  # type: ignore[arg-type]

    assert empty_result.is_correct is False
    assert empty_result.reason == "malformed_answer"
    assert empty_result.metadata == {"detail": "answer is empty"}

    assert non_string_result.is_correct is False
    assert non_string_result.reason == "malformed_answer"
    assert non_string_result.metadata == {"detail": "answer is not a string"}


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
