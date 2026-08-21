from __future__ import annotations

import importlib
import math
from typing import Final, assert_never

from memcontam.readiness.phase13_new_mcq_candidate import mcq_normalize
from memcontam.readiness.phase13_new_mcq_leakage_models import (
    LeakageArtifactError,
    McqContent,
)

_STEM_BOUNDARY: Final = "\u241e"
_OPTION_BOUNDARY: Final = "\u241f"
_unicode_data = importlib.import_module("unicodedata2")


def answer_free_item_text(stem: str, options: tuple[str, ...]) -> str:
    return f"{mcq_normalize(stem)}\n{_STEM_BOUNDARY}\n" + f"\n{_OPTION_BOUNDARY}\n".join(
        mcq_normalize(option) for option in options
    )


def structural_representation(value: McqContent | str) -> str:
    match value:
        case McqContent(stem=stem, options=options):
            text = answer_free_item_text(stem, options)
        case str():
            text = mcq_normalize(value)
        case unreachable:
            assert_never(unreachable)
    output: list[str] = []
    in_token = False
    for character in text:
        category = _category(character)[0]
        token_character = category in {"L", "N"} or (category == "M" and in_token)
        if token_character and not in_token:
            output.append("#")
        if not token_character:
            output.append(character)
        in_token = token_character
    return "".join(output)


def mcq_identity(value: McqContent) -> tuple[str, tuple[str, ...]]:
    return mcq_normalize(value.stem), tuple(sorted(mcq_normalize(option) for option in value.options))


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_EMBEDDING_INVALID")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_EMBEDDING_INVALID")
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def levenshtein(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)
    masks: dict[str, int] = {}
    for index, character in enumerate(left):
        masks[character] = masks.get(character, 0) | (1 << index)
    score = len(left)
    positive, negative, final = ~0, 0, 1 << (len(left) - 1)
    for character in right:
        equal = masks.get(character, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        score += bool(positive_horizontal & final) - bool(negative_horizontal & final)
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = negative_horizontal | ~(vertical | positive_horizontal)
        negative = positive_horizontal & vertical
    return score


def _category(character: str) -> str:
    category_function = getattr(_unicode_data, "category", None)
    if not callable(category_function):
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_UNICODE_DATA_UNAVAILABLE")
    category = category_function(character)
    if not isinstance(category, str):
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_UNICODE_DATA_UNAVAILABLE")
    return category


__all__ = [
    "answer_free_item_text",
    "cosine",
    "levenshtein",
    "mcq_identity",
    "structural_representation",
]
