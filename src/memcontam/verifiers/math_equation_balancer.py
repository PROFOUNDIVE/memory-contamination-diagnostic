from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import re
from typing import TypeAlias

from memcontam.logging.schema import VerifierResult
from memcontam.tasks.base import TaskInstance


ParsedEquation: TypeAlias = tuple[tuple[int, ...], tuple[str, ...], int]
_INTEGER = re.compile(r"[+-]?\d+")
_OPERATORS = frozenset({"+", "-", "*", "/"})


def verify_answer(answer: str, task: TaskInstance) -> VerifierResult:
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

    target = task.verifier_spec.get("target")
    target_value = task.verifier_spec.get("target_value")
    registered_input = task.input.get("input")
    metadata = {"target": target, "target_value": target_value}
    parsed = _parse_equation(normalized)
    expected = _parse_equation(target) if isinstance(target, str) else None
    registered = _parse_input(registered_input) if isinstance(registered_input, str) else None
    if parsed is None:
        return VerifierResult(
            is_correct=False,
            parsed_answer=normalized,
            reason="malformed_answer",
            metadata=metadata,
        )
    if (
        expected is not None
        and registered is not None
        and type(target_value) is int
        and expected[0] == registered[0]
        and expected[2] == registered[1] == target_value
        and parsed[0] == registered[0]
        and parsed[2] == target_value
        and _evaluate(parsed[0], parsed[1]) == Fraction(target_value)
    ):
        return VerifierResult(
            is_correct=True,
            parsed_answer=normalized,
            reason="ok",
            metadata=metadata,
        )

    return VerifierResult(
        is_correct=False,
        parsed_answer=normalized,
        reason="wrong_answer",
        metadata=metadata,
    )


def verify_rhs_completion_answer(
    answer: str,
    spec: Mapping[str, int | str],
) -> VerifierResult:
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
    metadata = {"target": target, "target_value": target_value}
    accepted = {
        " ".join(str(value).split())
        for value in (target, target_value)
        if value is not None
    }
    return VerifierResult(
        is_correct=normalized in accepted,
        parsed_answer=normalized,
        reason="ok" if normalized in accepted else "wrong_answer",
        metadata=metadata,
    )


def _parse_equation(value: str) -> ParsedEquation | None:
    tokens = " ".join(value.split()).split(" ")
    if len(tokens) < 5 or len(tokens) % 2 == 0 or tokens[-2] != "=":
        return None
    number_tokens = tokens[:-2:2]
    operator_tokens = tokens[1:-2:2]
    if (
        len(operator_tokens) != len(number_tokens) - 1
        or any(_INTEGER.fullmatch(token) is None for token in (*number_tokens, tokens[-1]))
        or any(operator not in _OPERATORS for operator in operator_tokens)
    ):
        return None
    return (
        tuple(int(token) for token in number_tokens),
        tuple(operator_tokens),
        int(tokens[-1]),
    )


def _parse_input(value: str) -> tuple[tuple[int, ...], int] | None:
    tokens = " ".join(value.split()).split(" ")
    if len(tokens) < 5 or len(tokens) % 2 == 0 or tokens[-2] != "=":
        return None
    number_tokens = tokens[:-2:2]
    slot_tokens = tokens[1:-2:2]
    if (
        len(slot_tokens) != len(number_tokens) - 1
        or set(slot_tokens) != {"?"}
        or any(_INTEGER.fullmatch(token) is None for token in (*number_tokens, tokens[-1]))
    ):
        return None
    return tuple(int(token) for token in number_tokens), int(tokens[-1])


def _evaluate(operands: tuple[int, ...], operators: tuple[str, ...]) -> Fraction | None:
    terms = [Fraction(operands[0])]
    additive_operators: list[str] = []
    for operator, operand in zip(operators, operands[1:], strict=True):
        value = Fraction(operand)
        if operator == "*":
            terms[-1] = terms[-1] * value
        elif operator == "/":
            if value == 0:
                return None
            terms[-1] = terms[-1] / value
        else:
            additive_operators.append(operator)
            terms.append(value)
    result = terms[0]
    for operator, term in zip(additive_operators, terms[1:], strict=True):
        result = result + term if operator == "+" else result - term
    return result
