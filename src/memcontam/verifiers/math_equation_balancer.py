from __future__ import annotations

from memcontam.logging.schema import VerifierResult



def verify_answer(answer: str, spec: dict) -> VerifierResult:
    if not isinstance(answer, str):
        return VerifierResult(
            is_correct=False,
            parsed_answer=None,
            reason="malformed_answer",
            metadata={"detail": "answer is not a string"},
        )

    normalized = " ".join(answer.split())
    if not normalized:
        return VerifierResult(
            is_correct=False,
            parsed_answer=None,
            reason="malformed_answer",
            metadata={"detail": "answer is empty"},
        )

    target = spec.get("target")
    target_value = spec.get("target_value")

    normalized_target = " ".join(str(target).split()) if target is not None else ""
    normalized_value = " ".join(str(target_value).split()) if target_value is not None else ""

    if normalized == normalized_target or normalized == normalized_value:
        return VerifierResult(
            is_correct=True,
            parsed_answer=normalized,
            reason="ok",
            metadata={"target": target, "target_value": target_value},
        )

    return VerifierResult(
        is_correct=False,
        parsed_answer=normalized,
        reason="wrong_answer",
        metadata={"target": target, "target_value": target_value},
    )
