from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final, TypeVar

from memcontam.readiness.phase13_unicode_15_1 import (
    UnicodeProvenance,
    mcq_normalize,
    mcq_tokens,
    unicode_provenance,
)

H1_ID: Final = "MCQ-H1-LEXICAL-OVERLAP-v1"
H2_ID: Final = "MCQ-H2-DETAIL-LENGTH-v1"
I1_ID: Final = "MCQ-I1-SINGLETON-OPTION-v1"
_Comparable = TypeVar("_Comparable")


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
    return unicode_provenance().unicode_data_manifest_hash


__all__ = [
    "CandidateCertification",
    "CandidateContractError",
    "DisplayedMcq",
    "InterventionRelevance",
    "UnicodeProvenance",
    "build_intervention_relevance",
    "certify_task_candidate",
    "h1_selection",
    "h2_selection",
    "mcq_normalize",
    "mcq_tokens",
    "unicode_provenance",
]
