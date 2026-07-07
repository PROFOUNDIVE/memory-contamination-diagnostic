from __future__ import annotations

from memcontam.logging.schema import VerifierResult



def verify_words(answer_words: list[str], sorted_words: list[str]) -> VerifierResult:
    normalized_answer = [word.strip() for word in answer_words]
    normalized_gold = [word.strip() for word in sorted_words]
    return VerifierResult(
        is_correct=normalized_answer == normalized_gold,
        parsed_answer=" ".join(normalized_answer),
    )
