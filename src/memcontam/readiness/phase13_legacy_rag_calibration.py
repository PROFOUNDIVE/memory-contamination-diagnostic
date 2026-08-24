from __future__ import annotations

import hashlib
import unicodedata
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .phase13_legacy_rag_bytes import canonical_json_bytes
from .phase13_legacy_rag_generators import MebCandidate, WORD_SORTING_VOCABULARY
from .phase13_legacy_rag_models import ArtifactReference, BuildCandidate, Sha256


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class HistoricalMebPilot(_FrozenModel):
    path: Literal["data/tasks/math_equation_balancer_pilot.jsonl"]
    sha256: Sha256
    status: Literal["HISTORICAL_EVIDENCE_ONLY"]


class MebCalibrationRegistry(_FrozenModel):
    schema_version: Literal["legacy_meb_current_calibration_registry_v1"]
    registry_id: Literal["legacy_meb_current_calibration_registry_v1"]
    current_calibration_count: Literal[16]
    selection_law: Literal["first_16_eligible_operator_slot_candidates"]
    partition_law: Literal["first_16_D_cal_next_64_D_build"]
    historical_rhs_completion_pilot: HistoricalMebPilot
    candidates: tuple[BuildCandidate, ...]


class _WordSortingCalibrationRow(_FrozenModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    sample_id: str
    words: tuple[str, ...]


class WordSortingLeakageCalibration(_FrozenModel):
    schema_version: Literal["word_sorting_leakage_threshold_calibration_v1"]
    threshold_calibration_contract_id: Literal["word_sorting_leakage_threshold_calibration_v1"]
    token_overlap_metric_id: Literal["word_sorting_token_overlap_metric_v1"]
    lexical_signature_metric_id: Literal["word_sorting_lexical_signature_metric_v1"]
    canonicalization_id: Literal["NFC_exact_token_v1"]
    comparator_id: Literal["registered_word_sorting_lexical_comparator_v1"]
    calibration_registry: ArtifactReference
    positive_control_ids: tuple[str, ...]
    negative_control_ids: tuple[str, ...]
    positive_similarities: dict[str, tuple[str, ...]]
    negative_similarities: dict[str, tuple[str, ...]]
    extrema: dict[str, dict[str, str]]
    thresholds: dict[str, str]
    boundary_rule: Literal["similarity_greater_than_or_equal_to_threshold_rejects"]
    separability_result: Literal["PASS"]


class LegacyRagCalibrationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


def build_meb_calibration_registry(
    candidates: tuple[MebCandidate, ...],
) -> MebCalibrationRegistry:
    if len(candidates) != 16:
        raise LegacyRagCalibrationError("MEB_CURRENT_CAL_BUILD_PARTITION_UNDERSUPPLIED")
    return MebCalibrationRegistry(
        schema_version="legacy_meb_current_calibration_registry_v1",
        registry_id="legacy_meb_current_calibration_registry_v1",
        current_calibration_count=16,
        selection_law="first_16_eligible_operator_slot_candidates",
        partition_law="first_16_D_cal_next_64_D_build",
        historical_rhs_completion_pilot=HistoricalMebPilot(
            path="data/tasks/math_equation_balancer_pilot.jsonl",
            sha256="6fa5a5d3be52853f8d9da93a9a9c0ea5399f67c9c08acc64fdbdd4821f68bb41",
            status="HISTORICAL_EVIDENCE_ONLY",
        ),
        candidates=tuple(
            BuildCandidate(
                candidate_id=row.digest,
                canonical_signature=row.canonical_signature,
                candidate_bytes=row.candidate_bytes.decode("utf-8"),
                response=row.response,
            )
            for row in candidates
        ),
    )


def build_word_sorting_leakage_calibration(path: Path) -> WordSortingLeakageCalibration:
    rows = tuple(
        _WordSortingCalibrationRow.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    positives = tuple(_positive_control(row) for row in rows)
    negatives = tuple(combinations(rows, 2))
    positive_values = tuple(
        word_sorting_similarities(row.words, control) for row, control in positives
    )
    negative_values = tuple(
        word_sorting_similarities(left.words, right.words) for left, right in negatives
    )
    token_positive = tuple(value[0] for value in positive_values)
    lexical_positive = tuple(value[1] for value in positive_values)
    token_negative = tuple(value[0] for value in negative_values)
    lexical_negative = tuple(value[1] for value in negative_values)
    token_extrema = (min(token_positive), max(token_negative))
    lexical_extrema = (min(lexical_positive), max(lexical_negative))
    if token_extrema[1] >= token_extrema[0] or lexical_extrema[1] >= lexical_extrema[0]:
        raise LegacyRagCalibrationError("WORD_SORTING_LEAKAGE_CALIBRATION_NOT_SEPARABLE")
    token_threshold = sum(token_extrema, Fraction()) / 2
    lexical_threshold = sum(lexical_extrema, Fraction()) / 2
    if token_threshold != Fraction(1, 4) or lexical_threshold != Fraction(1, 6):
        raise LegacyRagCalibrationError("WORD_SORTING_LEAKAGE_CALIBRATION_AUTHORITY_MISMATCH")
    return WordSortingLeakageCalibration(
        schema_version="word_sorting_leakage_threshold_calibration_v1",
        threshold_calibration_contract_id="word_sorting_leakage_threshold_calibration_v1",
        token_overlap_metric_id="word_sorting_token_overlap_metric_v1",
        lexical_signature_metric_id="word_sorting_lexical_signature_metric_v1",
        canonicalization_id="NFC_exact_token_v1",
        comparator_id="registered_word_sorting_lexical_comparator_v1",
        calibration_registry=ArtifactReference(
            path=str(path.relative_to(path.parents[2])),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            row_count=len(rows),
        ),
        positive_control_ids=tuple(f"{row.sample_id}:single-token-substitution" for row in rows),
        negative_control_ids=tuple(
            f"{left.sample_id}:{right.sample_id}" for left, right in negatives
        ),
        positive_similarities={
            "token_overlap": tuple(map(str, token_positive)),
            "lexical_signature": tuple(map(str, lexical_positive)),
        },
        negative_similarities={
            "token_overlap": tuple(map(str, token_negative)),
            "lexical_signature": tuple(map(str, lexical_negative)),
        },
        extrema={
            "token_overlap": {"s_plus": str(token_extrema[0]), "s_minus": str(token_extrema[1])},
            "lexical_signature": {
                "s_plus": str(lexical_extrema[0]),
                "s_minus": str(lexical_extrema[1]),
            },
        },
        thresholds={"token_overlap": str(token_threshold), "lexical_signature": str(lexical_threshold)},
        boundary_rule="similarity_greater_than_or_equal_to_threshold_rejects",
        separability_result="PASS",
    )


def _positive_control(row: _WordSortingCalibrationRow) -> tuple[_WordSortingCalibrationRow, tuple[str, ...]]:
    replacement = next(word for word in WORD_SORTING_VOCABULARY if word not in row.words)
    return row, (replacement, *row.words[1:])


def word_sorting_similarities(
    left: tuple[str, ...], right: tuple[str, ...]
) -> tuple[Fraction, Fraction]:
    left_tokens = tuple(sorted(unicodedata.normalize("NFC", token) for token in left))
    right_tokens = tuple(sorted(unicodedata.normalize("NFC", token) for token in right))
    left_set, right_set = set(left_tokens), set(right_tokens)
    token = Fraction(len(left_set & right_set), len(left_set | right_set))
    lexical = Fraction(1) - Fraction(_levenshtein(left_tokens, right_tokens), max(len(left), len(right)))
    return token, lexical


def _levenshtein(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def calibration_artifact_sha256(model: _FrozenModel) -> str:
    return hashlib.sha256(canonical_json_bytes(model.model_dump(mode="json"))).hexdigest()


__all__ = [
    "LegacyRagCalibrationError",
    "MebCalibrationRegistry",
    "WordSortingLeakageCalibration",
    "build_meb_calibration_registry",
    "build_word_sorting_leakage_calibration",
    "calibration_artifact_sha256",
    "word_sorting_similarities",
]
