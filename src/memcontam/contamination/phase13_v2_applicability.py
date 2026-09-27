from __future__ import annotations

import hashlib
from fractions import Fraction
from itertools import permutations, product
from typing import Final


G24_APPLICABILITY_ID: Final = "G24_CANONICAL_FALSE_RULE_APPLICABILITY_V2"
WS_APPLICABILITY_ID: Final = "WS_CANONICAL_FALSE_RULE_APPLICABILITY_V2"

_G24_SPEC: Final = (
    "exact-number-use|operators=+,-,*,/|exact-rational|full-binary-parenthesization|"
    "flat=ordinary-precedence-left-associative|applicable=full-nonempty-and-flat-empty"
)
_WS_SPEC: Final = (
    "zero-based|j>=1|nonempty-common-prefix|j<min-length-1|strict-reversal|"
    "final-tie=false|prefix-pair=false"
)
G24_APPLICABILITY_SHA256: Final = hashlib.sha256(_G24_SPEC.encode()).hexdigest()
WS_APPLICABILITY_SHA256: Final = hashlib.sha256(_WS_SPEC.encode()).hexdigest()

_OPERATORS: Final = ("+", "-", "*", "/")
_TARGET: Final = Fraction(24)


def game24_false_rule_applicable(numbers: tuple[int, int, int, int]) -> bool:
    values = tuple(Fraction(number) for number in numbers)
    return _TARGET in _full_results(values) and not _has_flat_solution(numbers)


def word_sorting_false_rule_applicable(words: tuple[str, ...]) -> bool:
    return any(
        _word_pair_is_applicable(left, right)
        for index, left in enumerate(words)
        for right in words[index + 1 :]
    )


def _full_results(values: tuple[Fraction, ...]) -> frozenset[Fraction]:
    if len(values) == 1:
        return frozenset(values)
    results: set[Fraction] = set()
    for left_index in range(len(values)):
        for right_index in range(left_index + 1, len(values)):
            remainder = tuple(
                value
                for index, value in enumerate(values)
                if index not in {left_index, right_index}
            )
            left = values[left_index]
            right = values[right_index]
            combined = {left + right, left - right, right - left, left * right}
            if right:
                combined.add(left / right)
            if left:
                combined.add(right / left)
            for value in combined:
                results.update(_full_results((*remainder, value)))
    return frozenset(results)


def _has_flat_solution(numbers: tuple[int, int, int, int]) -> bool:
    return any(
        _evaluate_flat(order, operators) == _TARGET
        for order in set(permutations(numbers))
        for operators in product(_OPERATORS, repeat=3)
    )


def _evaluate_flat(
    numbers: tuple[int, ...], operators: tuple[str, ...]
) -> Fraction | None:
    terms = [Fraction(numbers[0])]
    additive: list[str] = []
    for operator, number in zip(operators, numbers[1:], strict=True):
        value = Fraction(number)
        if operator == "*":
            terms[-1] *= value
        elif operator == "/":
            if not value:
                return None
            terms[-1] /= value
        else:
            additive.append(operator)
            terms.append(value)
    result = terms[0]
    for operator, value in zip(additive, terms[1:], strict=True):
        result = result + value if operator == "+" else result - value
    return result


def _word_pair_is_applicable(left: str, right: str) -> bool:
    comparable = min(len(left), len(right))
    difference = next(
        (index for index in range(comparable) if left[index] != right[index]),
        None,
    )
    if difference is None or difference < 1 or difference >= comparable - 1:
        return False
    first_relation = -1 if left[difference] < right[difference] else 1
    final_relation = (left[-1] > right[-1]) - (left[-1] < right[-1])
    return first_relation * final_relation == -1


__all__ = [
    "G24_APPLICABILITY_ID",
    "G24_APPLICABILITY_SHA256",
    "WS_APPLICABILITY_ID",
    "WS_APPLICABILITY_SHA256",
    "game24_false_rule_applicable",
    "word_sorting_false_rule_applicable",
]
