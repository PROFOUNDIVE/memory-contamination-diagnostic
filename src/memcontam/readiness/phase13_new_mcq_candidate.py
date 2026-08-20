from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, TypeVar

H1_ID: Final = "MCQ-H1-LEXICAL-OVERLAP-v1"
H2_ID: Final = "MCQ-H2-DETAIL-LENGTH-v1"
I1_ID: Final = "MCQ-I1-SINGLETON-OPTION-v1"
UNICODE_VERSION: Final = "15.1.0"
_WHITE_SPACE: Final = frozenset(
    "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_Comparable = TypeVar("_Comparable")
_unicode_data = importlib.import_module("unicodedata2")


class CandidateContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class DisplayedMcq:
    query_id: str
    stem: str
    options: tuple[str, ...]
    gold_index: int
    display_identity: str

    def __post_init__(self) -> None:
        if (
            not self.query_id
            or not self.options
            or not 0 <= self.gold_index < len(self.options)
            or not self.display_identity
        ):
            raise CandidateContractError("NEW_MCQ_DISPLAYED_INPUT_INVALID")


@dataclass(frozen=True, slots=True)
class CandidateCertification:
    task_id: str
    source_identity: str
    candidate_id: str
    correct_twin_id: str
    irrelevant_control_id: str
    applicable_query_ids: tuple[str, ...]
    counterexample_query_ids: tuple[str, ...]
    display_identities: tuple[str, ...]
    unicode_data_manifest_hash: str


@dataclass(frozen=True, slots=True)
class InterventionRelevance:
    contam: bool
    correct: bool
    irrelevant: bool


def mcq_normalize(text: str) -> str:
    version = getattr(_unicode_data, "unidata_version", None)
    if version != UNICODE_VERSION:
        raise CandidateContractError("NEW_MCQ_UNICODE_VERSION_MISMATCH")
    normalize = getattr(_unicode_data, "normalize", None)
    if not callable(normalize):
        raise CandidateContractError("NEW_MCQ_UNICODE_DATA_UNAVAILABLE")
    normalized_value = normalize("NFKC", text)
    if not isinstance(normalized_value, str):
        raise CandidateContractError("NEW_MCQ_UNICODE_DATA_UNAVAILABLE")
    normalized = normalized_value.casefold()
    spaced = "".join(" " if character in _WHITE_SPACE else character for character in normalized)
    return " ".join(spaced.split())


def mcq_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for character in mcq_normalize(text):
        category_function = getattr(_unicode_data, "category", None)
        if not callable(category_function):
            raise CandidateContractError("NEW_MCQ_UNICODE_DATA_UNAVAILABLE")
        category = category_function(character)
        if not isinstance(category, str):
            raise CandidateContractError("NEW_MCQ_UNICODE_DATA_UNAVAILABLE")
        if category[0] in {"L", "N"} or (category[0] == "M" and current):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def h1_selection(item: DisplayedMcq) -> int | None:
    stem = frozenset(mcq_tokens(item.stem))
    scores = tuple(_jaccard(stem, frozenset(mcq_tokens(option))) for option in item.options)
    return _unique_max_index(scores, max(scores))


def h2_selection(item: DisplayedMcq) -> int | None:
    scores = tuple(
        (len(mcq_tokens(option)), sum(character != " " for character in mcq_normalize(option)))
        for option in item.options
    )
    return _unique_max_index(scores, max(scores))


def certify_task_candidate(
    task_id: str,
    source_identity: str,
    rows: tuple[DisplayedMcq, ...],
) -> CandidateCertification:
    if not task_id or not source_identity or not rows or any(len(row.options) <= 1 for row in rows):
        raise CandidateContractError("NEW_MCQ_CANDIDATE_SOURCE_INVALID")
    for candidate_id, correct_id, selector in (
        (H1_ID, "MCQ-H1-CORRECT-SUBSTANTIVE-CONTENT-v1", h1_selection),
        (H2_ID, "MCQ-H2-CORRECT-SUBSTANTIVE-CONTENT-v1", h2_selection),
    ):
        selections = tuple((row, selector(row)) for row in rows)
        applicable = tuple(row.query_id for row, selected in selections if selected is not None)
        counterexamples = tuple(
            row.query_id
            for row, selected in selections
            if selected is not None and selected != row.gold_index
        )
        if applicable and counterexamples:
            return CandidateCertification(
                task_id=task_id,
                source_identity=source_identity,
                candidate_id=candidate_id,
                correct_twin_id=correct_id,
                irrelevant_control_id=I1_ID,
                applicable_query_ids=applicable,
                counterexample_query_ids=counterexamples,
                display_identities=tuple(row.display_identity for row in rows),
                unicode_data_manifest_hash=_unicode_manifest_hash(),
            )
    raise CandidateContractError("NEW_MCQ_CANDIDATE_FAMILY_NOT_READY")


def build_intervention_relevance(
    certification: CandidateCertification,
    rows: tuple[DisplayedMcq, ...],
) -> dict[str, InterventionRelevance]:
    applicable = set(certification.applicable_query_ids)
    return {
        row.query_id: InterventionRelevance(
            contam=row.query_id in applicable,
            correct=row.query_id in applicable,
            irrelevant=False,
        )
        for row in rows
    }


def _jaccard(left: frozenset[str], right: frozenset[str]) -> Fraction:
    union = left | right
    return Fraction(len(left & right), len(union)) if union else Fraction(0)


def _unique_max_index(values: tuple[_Comparable, ...], maximum: _Comparable) -> int | None:
    positions = tuple(index for index, value in enumerate(values) if value == maximum)
    return positions[0] if len(positions) == 1 else None


def _unicode_manifest_hash() -> str:
    module_file = getattr(_unicode_data, "__file__", None)
    if not isinstance(module_file, str):
        raise CandidateContractError("NEW_MCQ_UNICODE_DATA_UNAVAILABLE")
    module_path = Path(module_file)
    return hashlib.sha256(module_path.read_bytes()).hexdigest()


__all__ = [
    "CandidateCertification",
    "CandidateContractError",
    "DisplayedMcq",
    "InterventionRelevance",
    "build_intervention_relevance",
    "certify_task_candidate",
    "h1_selection",
    "h2_selection",
    "mcq_normalize",
    "mcq_tokens",
]
