from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Final


OPERATORS: Final = ("+", "-", "*", "/")


@dataclass(frozen=True, slots=True)
class _Game24Node:
    value: Fraction
    expression: str
    operator: str | None = None
    left: _Game24Node | None = None
    right: _Game24Node | None = None


def _fraction(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _game24_nodes(numbers: tuple[int, ...]) -> tuple[_Game24Node, ...]:
    initial = tuple(_Game24Node(Fraction(number), str(number)) for number in numbers)

    def solve(nodes: tuple[_Game24Node, ...]) -> tuple[_Game24Node, ...]:
        if len(nodes) == 1:
            return nodes
        values: dict[str, _Game24Node] = {}
        for first, second in combinations(range(len(nodes)), 2):
            left, right = nodes[first], nodes[second]
            remainder = tuple(node for index, node in enumerate(nodes) if index not in {first, second})
            candidates: list[tuple[str, _Game24Node, _Game24Node, Fraction]] = [
                ("+", left, right, left.value + right.value),
                ("*", left, right, left.value * right.value),
                ("-", left, right, left.value - right.value),
                ("-", right, left, right.value - left.value),
            ]
            if right.value:
                candidates.append(("/", left, right, left.value / right.value))
            if left.value:
                candidates.append(("/", right, left, right.value / left.value))
            for operator, candidate_left, candidate_right, value in candidates:
                if operator in {"+", "*"} and candidate_right.expression < candidate_left.expression:
                    candidate_left, candidate_right = candidate_right, candidate_left
                node = _Game24Node(
                    value,
                    f"({candidate_left.expression}{operator}{candidate_right.expression})",
                    operator,
                    candidate_left,
                    candidate_right,
                )
                for result in solve((*remainder, node)):
                    values[result.expression] = result
        return tuple(values.values())

    return solve(initial)


def _game24_trace(node: _Game24Node) -> list[dict[str, object]]:
    trace: list[dict[str, object]] = []

    def visit(current: _Game24Node) -> None:
        if current.operator is None:
            return
        assert current.left is not None and current.right is not None
        visit(current.left)
        visit(current.right)
        trace.append(
            {
                "node_index": len(trace) + 1,
                "operator": current.operator,
                "left": _fraction(current.left.value),
                "right": _fraction(current.right.value),
                "result": _fraction(current.value),
            }
        )

    visit(node)
    return trace


def game24_certificate(numbers: tuple[int, int, int, int]) -> dict[str, object] | None:
    matches = [node for node in _game24_nodes(numbers) if node.value == 24]
    integer_matches = [node for node in matches if _all_integer(node)]
    if not matches or integer_matches:
        return None
    selected = min(matches, key=lambda node: node.expression)
    return {
        "schema_version": "phase12_fv5_game24_certificate_v1",
        "task_family": "game24",
        "input_canonical": ",".join(str(number) for number in numbers),
        "numbers": list(numbers),
        "target": 24,
        "expression": selected.expression,
        "postorder_trace": _game24_trace(selected),
    }


def _all_integer(node: _Game24Node) -> bool:
    if node.operator is None:
        return True
    assert node.left is not None and node.right is not None
    return node.value.denominator == 1 and _all_integer(node.left) and _all_integer(node.right)


def _apply(left: Fraction, operator: str, right: Fraction) -> Fraction | None:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/" and right:
        return left / right
    return None


def _standard(a: int, first: str, b: int, second: str, c: int) -> Fraction | None:
    if second in {"*", "/"}:
        right = _apply(Fraction(b), second, Fraction(c))
        return None if right is None else _apply(Fraction(a), first, right)
    left = _apply(Fraction(a), first, Fraction(b))
    return None if left is None else _apply(left, second, Fraction(c))


def _left_to_right(a: int, first: str, b: int, second: str, c: int) -> Fraction | None:
    left = _apply(Fraction(a), first, Fraction(b))
    return None if left is None else _apply(left, second, Fraction(c))


def meb_certificate(a: int, b: int, c: int, target: int) -> dict[str, object] | None:
    pairs = tuple((first, second) for first in OPERATORS for second in OPERATORS)
    standard = [(_standard(a, first, b, second, c), first, second) for first, second in pairs]
    selected = next((item for item in standard if item[0] == target), None)
    left_results = [(_left_to_right(a, first, b, second, c), first, second) for first, second in pairs]
    if selected is None or any(result == target for result, _, _ in left_results):
        return None
    result, first, second = selected
    assert result is not None
    return {
        "schema_version": "phase12_fv5_meb_certificate_v1",
        "task_family": "math_equation_balancer",
        "input_canonical": f"{a},{b},{c},{target}",
        "numbers": [a, b, c],
        "target": target,
        "operator_pair": [first, second],
        "expression": f"{a}{first}{b}{second}{c}",
        "standard_result": _fraction(result),
        "left_to_right_results": [
            {"operator_pair": [left, right], "result": None if value is None else _fraction(value)}
            for value, left, right in left_results
        ],
    }


def word_sorting_certificate(words: tuple[str, str, str]) -> dict[str, object] | None:
    ordered = tuple(sorted(words))
    final_order = tuple(sorted(words, key=lambda word: word[-1]))
    witnesses: list[dict[str, object]] = []
    for left, right in combinations(ordered, 2):
        index = next((position for position, pair in enumerate(zip(left, right, strict=True)) if pair[0] != pair[1]), None)
        if index != 1 or left[0] != right[0]:
            continue
        first_relation = "lt" if left[index] < right[index] else "gt"
        final_relation = "lt" if left[-1] < right[-1] else "gt"
        if first_relation != final_relation:
            witnesses.append(
                {
                    "left": left,
                    "right": right,
                    "common_prefix_length": 1,
                    "first_difference_index": 1,
                    "left_first_difference_char": left[index],
                    "right_first_difference_char": right[index],
                    "left_final_char": left[-1],
                    "right_final_char": right[-1],
                    "first_difference_relation": first_relation,
                    "final_character_relation": final_relation,
                }
            )
    if not witnesses or ordered == final_order:
        return None
    return {
        "schema_version": "phase12_fv5_word_sorting_certificate_v1",
        "task_family": "word_sorting",
        "input_canonical": "|".join(words),
        "input_words": list(words),
        "correct_order": list(ordered),
        "final_character_order": list(final_order),
        "witness": min(witnesses, key=lambda witness: (str(witness["left"]), str(witness["right"]))),
    }
