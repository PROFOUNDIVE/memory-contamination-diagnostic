from __future__ import annotations

from memcontam.logging.schema import VerifierResult


def verify_words(answer_words: list[str], sorted_words: list[str]) -> VerifierResult:
    if not isinstance(answer_words, list) or not answer_words:
        return VerifierResult(
            is_correct=False,
            parsed_answer=None,
            reason="malformed_answer",
            metadata={"detail": "answer_words is not a non-empty list"},
        )

    if any(not isinstance(word, str) for word in answer_words):
        return VerifierResult(
            is_correct=False,
            parsed_answer=None,
            reason="malformed_answer",
            metadata={"detail": "answer_words contains non-string tokens"},
        )

    normalized_answer = [word.strip() for word in answer_words]
    normalized_gold = [word.strip() for word in sorted_words]
    parsed = " ".join(normalized_answer)

    if normalized_answer == normalized_gold:
        return VerifierResult(
            is_correct=True,
            parsed_answer=parsed,
            reason="ok",
            metadata={},
        )

    return VerifierResult(
        is_correct=False,
        parsed_answer=parsed,
        reason="wrong_order",
        metadata={"expected": normalized_gold, "actual": normalized_answer},
    )
