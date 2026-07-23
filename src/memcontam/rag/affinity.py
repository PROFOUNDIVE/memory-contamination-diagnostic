from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from memcontam.memory.embeddings import normalized_dot_top_k


class AffinityError(ValueError):
    pass


@dataclass(frozen=True)
class AffinityAssignment:
    query_id: str
    candidate_id: str
    score: float
    band: str


@dataclass(frozen=True)
class AffinityResult:
    assignments: tuple[AffinityAssignment, ...]


def calibrate_affinity(
    queries: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    embedder: Any,
    bands: Mapping[str, tuple[float, float]],
) -> AffinityResult:
    _validate_bands(bands)
    if any(_has_main_outcome(value) for value in (*queries, *candidates)):
        raise AffinityError("OUTCOME_TUNED_AFFINITY")
    assignments = []
    for query in queries:
        query_id, query_text = _identity_and_text(query)
        query_vector = embedder.encode_query(query_text)
        for candidate in candidates:
            candidate_id, candidate_text = _identity_and_text(candidate)
            score = normalized_dot_top_k(
                query_vector, [embedder.encode_document(candidate_text)], [candidate_id], 1
            )[0][1]
            assignments.append(
                AffinityAssignment(query_id, candidate_id, score, _band_for(score, bands))
            )
    return AffinityResult(assignments=tuple(assignments))


def _validate_bands(bands: Mapping[str, tuple[float, float]]) -> None:
    ordered = sorted(bands.items(), key=lambda item: item[1][0])
    if not ordered:
        raise AffinityError("AFFINITY_BAND_REQUIRED")
    previous_end = -1.0
    for name, (start, end) in ordered:
        if not name or start < -1.0 or end > 1.0 or start >= end:
            raise AffinityError("AFFINITY_BAND_FORBIDDEN")
        if start < previous_end:
            raise AffinityError("AFFINITY_BANDS_OVERLAP")
        previous_end = end


def _band_for(score: float, bands: Mapping[str, tuple[float, float]]) -> str:
    for name, (start, end) in sorted(bands.items(), key=lambda item: item[1][0]):
        if start <= score <= end:
            return name
    raise AffinityError("AFFINITY_BAND_FORBIDDEN")


def _identity_and_text(value: Mapping[str, Any]) -> tuple[str, str]:
    identifier = value.get("id")
    text = value.get("text", value.get("content"))
    if not isinstance(identifier, str) or not isinstance(text, str):
        raise AffinityError("INVALID_AFFINITY_INPUT")
    return identifier, text


def _has_main_outcome(value: Mapping[str, Any]) -> bool:
    return any(
        key in value for key in ("main_outcome", "outcome", "verified_success", "final_inclusion")
    )
