from __future__ import annotations

from memcontam.logging.schema import VerifierResult



def verify_answer(answer: str, spec: dict) -> VerifierResult:
    return VerifierResult(is_correct=False, parsed_answer=answer, reason="verifier skeleton")
