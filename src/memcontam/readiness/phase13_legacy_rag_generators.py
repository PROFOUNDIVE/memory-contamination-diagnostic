from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, product
from typing import Final, Literal, TypeAlias, assert_never

from .phase13_legacy_rag_bytes import JsonValue, canonical_json_bytes

_GAME24_GENERATOR: Final = "legacy_game24_build_generator_v1"
_MEB_GENERATOR: Final = "legacy_meb_build_generator_v1"
MebOperator: TypeAlias = Literal["+", "-", "*", "/"]
_MEB_OPERATORS: Final[tuple[MebOperator, ...]] = ("+", "-", "*", "/")
_WORD_SORTING_GENERATOR: Final = "legacy_word_sorting_build_generator_v1"
WORD_SORTING_VOCABULARY: Final = (
    "acorn", "alder", "amber", "basil", "birch", "cedar", "coral", "delta",
    "ember", "fern", "flint", "grove", "hazel", "ivory", "juniper", "kelp",
    "linen", "maple", "moss", "nickel", "olive", "pearl", "quartz", "reed",
    "sable", "thyme", "umber", "violet", "willow", "xenon", "yarrow", "zinc",
)


@dataclass(frozen=True, slots=True)
class Game24Candidate:
    numbers: tuple[int, int, int, int]
    target: int
    response: str
    candidate_bytes: bytes
    digest: str
    canonical_signature: str


@dataclass(frozen=True, slots=True)
class WordSortingCandidate:
    input_words: tuple[str, str, str, str, str]
    sorted_words: tuple[str, str, str, str, str]
    response: str
    candidate_bytes: bytes
    digest: str
    canonical_signature: str


@dataclass(frozen=True, slots=True)
class MebCandidate:
    ordered_operands: tuple[int, ...]
    target_value: int
    canonical_operator_tuple: tuple[MebOperator, ...]
    response: str
    candidate_bytes: bytes
    digest: str
    canonical_signature: str


def game24_candidates(
    excluded_signatures: frozenset[str], *, limit: int
) -> tuple[Game24Candidate, ...]:
    selected = heapq.nsmallest(limit, _game24_stream(excluded_signatures), key=lambda row: row[:2])
    return tuple(
        Game24Candidate(
            numbers=numbers,
            target=24,
            response=response.decode("ascii"),
            candidate_bytes=candidate_bytes,
            digest=digest.hex(),
            canonical_signature=signature,
        )
        for digest, candidate_bytes, numbers, response, signature in selected
    )


def word_sorting_candidates(
    excluded_signatures: frozenset[str], *, limit: int
) -> tuple[WordSortingCandidate, ...]:
    selected = heapq.nsmallest(
        limit, _word_sorting_stream(excluded_signatures), key=lambda row: row[:2]
    )
    return tuple(
        WordSortingCandidate(
            input_words=input_words,
            sorted_words=sorted_words,
            response=" ".join(sorted_words),
            candidate_bytes=candidate_bytes,
            digest=digest.hex(),
            canonical_signature=signature,
        )
        for digest, candidate_bytes, input_words, sorted_words, signature in selected
    )


def meb_candidates(
    excluded_signatures: frozenset[str], *, limit: int
) -> tuple[MebCandidate, ...]:
    selected = heapq.nsmallest(limit, _meb_stream(excluded_signatures), key=lambda row: row[:2])
    return tuple(
        MebCandidate(
            ordered_operands=operands,
            target_value=target,
            canonical_operator_tuple=operators,
            response=response,
            candidate_bytes=candidate_bytes,
            digest=digest.hex(),
            canonical_signature=signature,
        )
        for digest, candidate_bytes, operands, target, operators, response, signature in selected
    )


def _meb_stream(
    excluded_signatures: frozenset[str],
) -> Iterator[tuple[bytes, bytes, tuple[int, ...], int, tuple[MebOperator, ...], str, str]]:
    for operand_count in (3, 4):
        for operands in product(range(1, 10), repeat=operand_count):
            if len(set(operands)) < 2:
                continue
            canonical: dict[int, tuple[MebOperator, ...]] = {}
            for operators in product(_MEB_OPERATORS, repeat=operand_count - 1):
                value = _evaluate_meb(operands, operators)
                if value.denominator == 1 and -512 <= value <= 512:
                    canonical.setdefault(int(value), operators)
            for target, operators in canonical.items():
                operand_values: list[JsonValue] = list(operands)
                signature_key: dict[str, JsonValue] = {
                    "ordered_operands": operand_values,
                    "target_value": target,
                }
                signature = hashlib.sha256(canonical_json_bytes(signature_key)).hexdigest()
                if signature in excluded_signatures:
                    continue
                key: dict[str, JsonValue] = {
                    "generator": _MEB_GENERATOR,
                    **signature_key,
                    "canonical_operator_tuple": list(operators),
                }
                candidate_bytes = canonical_json_bytes(key)
                expression = " ".join(
                    value
                    for pair in zip(map(str, operands), (*operators, ""), strict=True)
                    for value in pair
                    if value
                )
                yield (
                    hashlib.sha256(candidate_bytes).digest(),
                    candidate_bytes,
                    operands,
                    target,
                    operators,
                    f"{expression} = {target}",
                    signature,
                )


def _evaluate_meb(operands: tuple[int, ...], operators: tuple[MebOperator, ...]) -> Fraction:
    total = Fraction(0)
    term = Fraction(operands[0])
    additive = "+"
    for operator, operand in zip(operators, operands[1:], strict=True):
        match operator:
            case "*":
                term *= operand
            case "/":
                term /= operand
            case "+" | "-":
                total = total + term if additive == "+" else total - term
                term = Fraction(operand)
                additive = operator
            case unreachable:
                assert_never(unreachable)
    return total + term if additive == "+" else total - term


def _game24_stream(
    excluded_signatures: frozenset[str],
) -> Iterator[tuple[bytes, bytes, tuple[int, int, int, int], bytes, str]]:
    for numbers in combinations_with_replacement(range(1, 14), 4):
        response = _game24_response(numbers)
        if response is None:
            continue
        number_values: list[JsonValue] = list(numbers)
        key: dict[str, JsonValue] = {
            "generator": _GAME24_GENERATOR,
            "numbers": number_values,
            "target": 24,
        }
        candidate_bytes = canonical_json_bytes(key)
        digest = hashlib.sha256(candidate_bytes).digest()
        signature_key: dict[str, JsonValue] = {"numbers": number_values, "target": 24}
        signature = hashlib.sha256(canonical_json_bytes(signature_key)).hexdigest()
        if signature not in excluded_signatures:
            yield digest, candidate_bytes, numbers, response, signature


def _game24_response(numbers: tuple[int, int, int, int]) -> bytes | None:
    states: dict[int, dict[Fraction, bytes]] = {
        1 << index: {Fraction(number): str(number).encode("ascii")}
        for index, number in enumerate(numbers)
    }
    for size in range(2, 5):
        for indices in combinations(range(4), size):
            mask = sum(1 << index for index in indices)
            results: dict[Fraction, bytes] = {}
            left_mask = (mask - 1) & mask
            while left_mask:
                right_mask = mask ^ left_mask
                if right_mask:
                    for left_value, left_expression in states[left_mask].items():
                        for right_value, right_expression in states[right_mask].items():
                            for value, operator in _arithmetic_results(left_value, right_value):
                                expression = (
                                    b"(" + left_expression + operator + right_expression + b")"
                                )
                                current = results.get(value)
                                if current is None or expression < current:
                                    results[value] = expression
                left_mask = (left_mask - 1) & mask
            states[mask] = results
    return states[15].get(Fraction(24))


def _arithmetic_results(
    left: Fraction, right: Fraction
) -> tuple[tuple[Fraction, bytes], ...]:
    results = [(left + right, b"+"), (left - right, b"-"), (left * right, b"*")]
    if right:
        results.append((left / right, b"/"))
    return tuple(results)


def _word_sorting_stream(
    excluded_signatures: frozenset[str],
) -> Iterator[
    tuple[
        bytes,
        bytes,
        tuple[str, str, str, str, str],
        tuple[str, str, str, str, str],
        str,
    ]
]:
    for sorted_words in combinations(WORD_SORTING_VOCABULARY, 5):
        sorted_values: list[JsonValue] = list(sorted_words)
        keyed_words = []
        for word in sorted_words:
            permutation_key: dict[str, JsonValue] = {
                "domain": "legacy_word_sorting_input_v1",
                "sorted_subset": sorted_values,
                "word": word,
            }
            keyed_words.append(
                (hashlib.sha256(canonical_json_bytes(permutation_key)).digest(), word.encode(), word)
            )
        input_words = tuple(item[2] for item in sorted(keyed_words))
        key: dict[str, JsonValue] = {
            "generator": _WORD_SORTING_GENERATOR,
            "input_words": list(input_words),
            "sorted_words": sorted_values,
        }
        candidate_bytes = canonical_json_bytes(key)
        digest = hashlib.sha256(candidate_bytes).digest()
        signature_key: dict[str, JsonValue] = {"tokens": sorted_values}
        signature = hashlib.sha256(canonical_json_bytes(signature_key)).hexdigest()
        if signature not in excluded_signatures:
            yield digest, candidate_bytes, input_words, sorted_words, signature


__all__ = [
    "Game24Candidate",
    "MebCandidate",
    "WordSortingCandidate",
    "WORD_SORTING_VOCABULARY",
    "game24_candidates",
    "meb_candidates",
    "word_sorting_candidates",
]
