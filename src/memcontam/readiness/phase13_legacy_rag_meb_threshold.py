from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from fractions import Fraction
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from .phase13_legacy_rag_bytes import canonical_json_bytes
from .phase13_legacy_rag_generators import MebCandidate
from .phase13_legacy_rag_models import ArtifactReference, Sha256


BOOTSTRAP_PANEL_ID: Final = "D_structcal^MEB::meb_structural_threshold_bootstrap_panel_v2"
PRE_THRESHOLD_EXCLUSION_ID: Final = "meb_structcal_exact_canonical_eval_exclusion_v1"
STRUCTURAL_REPRESENTATION_ID: Final = (
    "proposed_sigma_meb_operand_count_ordered_operands_target_v1"
)
STRUCTURAL_METRIC_ID: Final = "proposed_meb_whole_token_normalized_levenshtein_v1"
THRESHOLD_MATERIALIZATION_ID: Final = "meb_structural_threshold_materialization_v1"
HISTORICAL_BOOTSTRAP_SHA256: Final = (
    "5a1a69dee03aba5ca785db3749ff48804944d3ad10880341ec17b660de7e60a1"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@dataclass(frozen=True, slots=True)
class MebStructuralEndpoint:
    ordered_operands: tuple[int, ...]
    target_value: int
    signature: Sha256

    @property
    def tokens(self) -> tuple[int, ...]:
        return (len(self.ordered_operands), *self.ordered_operands, self.target_value)


class MebStructuralControl(_FrozenModel):
    anchor_signature: Sha256
    control_signature: Sha256
    similarity: str


class MebStructuralThreshold(_FrozenModel):
    schema_version: Literal["meb_structural_threshold_materialization_v1"]
    threshold_materialization_id: Literal["meb_structural_threshold_materialization_v1"]
    evaluation_registry: ArtifactReference
    canonicalizer_id: Literal["Phase13-Protocol-v8::PROTO-19::canon_MEB"]
    pre_threshold_eval_exclusion_contract_id: Literal[
        "meb_structcal_exact_canonical_eval_exclusion_v1"
    ]
    bootstrap_panel_id: Literal[
        "D_structcal^MEB::meb_structural_threshold_bootstrap_panel_v2"
    ]
    bootstrap_panel_sha256: Sha256
    structural_representation_id: Literal[
        "proposed_sigma_meb_operand_count_ordered_operands_target_v1"
    ]
    structural_similarity_metric_id: Literal[
        "proposed_meb_whole_token_normalized_levenshtein_v1"
    ]
    anchor_signatures: tuple[Sha256, ...]
    positive_controls: tuple[MebStructuralControl, ...]
    negative_controls: tuple[MebStructuralControl, ...]
    endpoint_exclusion_statuses: dict[Sha256, Literal["PASS"]]
    s_plus: str
    s_minus: str
    tau_meb: str
    boundary_rule: Literal["similarity_greater_than_or_equal_to_threshold_rejects"]
    separability_result: Literal["PASS"]
    historical_pre_exclusion_bootstrap_evidence_sha256: Literal[
        "5a1a69dee03aba5ca785db3749ff48804944d3ad10880341ec17b660de7e60a1"
    ]


class MebStructuralThresholdError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def structural_similarity(left: MebStructuralEndpoint, right: MebStructuralEndpoint) -> Fraction:
    denominator = max(len(left.tokens), len(right.tokens))
    return Fraction(1) - Fraction(_levenshtein(left.tokens, right.tokens), denominator)


def build_meb_structural_threshold(
    candidates: tuple[MebCandidate, ...], evaluation_registry: ArtifactReference
) -> MebStructuralThreshold:
    anchors = tuple(heapq.nsmallest(16, candidates, key=_anchor_selector_key))
    if len(anchors) != 16:
        raise MebStructuralThresholdError("MEB_STRUCTURAL_BOOTSTRAP_PANEL_UNDERSUPPLIED")
    reserved = {anchor.canonical_signature for anchor in anchors}
    positives = _positive_controls(anchors, candidates, reserved)
    negatives = _negative_controls(anchors, candidates, reserved)
    positive_records = tuple(_control(anchor, control) for anchor, control in zip(anchors, positives))
    negative_records = tuple(_control(anchor, control) for anchor, control in zip(anchors, negatives))
    s_plus = min(Fraction(record.similarity) for record in positive_records)
    s_minus = max(Fraction(record.similarity) for record in negative_records)
    if s_minus >= s_plus:
        raise MebStructuralThresholdError("MEB_STRUCTURAL_BOOTSTRAP_PANEL_NOT_SEPARABLE")
    panel = {
        "anchors": [candidate.canonical_signature for candidate in anchors],
        "positive_controls": [record.model_dump(mode="json") for record in positive_records],
        "negative_controls": [record.model_dump(mode="json") for record in negative_records],
    }
    endpoints = (*anchors, *positives, *negatives)
    return MebStructuralThreshold(
        schema_version=THRESHOLD_MATERIALIZATION_ID,
        threshold_materialization_id=THRESHOLD_MATERIALIZATION_ID,
        evaluation_registry=evaluation_registry,
        canonicalizer_id="Phase13-Protocol-v8::PROTO-19::canon_MEB",
        pre_threshold_eval_exclusion_contract_id=PRE_THRESHOLD_EXCLUSION_ID,
        bootstrap_panel_id=BOOTSTRAP_PANEL_ID,
        bootstrap_panel_sha256=hashlib.sha256(canonical_json_bytes(panel)).hexdigest(),
        structural_representation_id=STRUCTURAL_REPRESENTATION_ID,
        structural_similarity_metric_id=STRUCTURAL_METRIC_ID,
        anchor_signatures=tuple(candidate.canonical_signature for candidate in anchors),
        positive_controls=positive_records,
        negative_controls=negative_records,
        endpoint_exclusion_statuses={candidate.canonical_signature: "PASS" for candidate in endpoints},
        s_plus=str(s_plus),
        s_minus=str(s_minus),
        tau_meb=str((s_minus + s_plus) / 2),
        boundary_rule="similarity_greater_than_or_equal_to_threshold_rejects",
        separability_result="PASS",
        historical_pre_exclusion_bootstrap_evidence_sha256=HISTORICAL_BOOTSTRAP_SHA256,
    )


def near_duplicate_exclusions(
    candidates: tuple[MebCandidate, ...],
    evaluation_endpoints: tuple[MebStructuralEndpoint, ...],
    threshold: Fraction,
) -> tuple[Sha256, ...]:
    rejected: list[Sha256] = []
    accepted = 0
    for candidate in sorted(candidates, key=lambda row: (row.digest, row.candidate_bytes)):
        endpoint = _endpoint(candidate)
        if any(structural_similarity(endpoint, evaluation) >= threshold for evaluation in evaluation_endpoints):
            rejected.append(candidate.canonical_signature)
        else:
            accepted += 1
            if accepted == 80:
                return tuple(sorted(rejected))
    raise MebStructuralThresholdError("MEB_CURRENT_CAL_BUILD_PARTITION_UNDERSUPPLIED")


def _anchor_selector_key(candidate: MebCandidate) -> tuple[bytes, bytes]:
    selector = {
        "domain": "meb_structcal_anchor_selection_v1",
        "anchor_signature": "",
        "ordered_operands": list(candidate.ordered_operands),
        "target_value": candidate.target_value,
    }
    return hashlib.sha256(canonical_json_bytes(selector)).digest(), candidate.candidate_bytes


def _positive_controls(
    anchors: tuple[MebCandidate, ...],
    candidates: tuple[MebCandidate, ...],
    reserved: set[str],
) -> tuple[MebCandidate, ...]:
    by_operands: dict[tuple[int, ...], list[MebCandidate]] = {}
    for candidate in candidates:
        by_operands.setdefault(candidate.ordered_operands, []).append(candidate)
    selected: list[MebCandidate] = []
    for anchor in anchors:
        control = next(
            (
                candidate
                for candidate in by_operands[anchor.ordered_operands]
                if candidate.target_value != anchor.target_value
                and candidate.canonical_signature not in reserved
            ),
            None,
        )
        if control is None:
            raise MebStructuralThresholdError("MEB_STRUCTURAL_BOOTSTRAP_PANEL_UNDERSUPPLIED")
        reserved.add(control.canonical_signature)
        selected.append(control)
    return tuple(selected)


def _negative_controls(
    anchors: tuple[MebCandidate, ...],
    candidates: tuple[MebCandidate, ...],
    reserved: set[str],
) -> tuple[MebCandidate, ...]:
    selected: list[MebCandidate] = []
    for anchor in anchors:
        eligible = (
            candidate
            for candidate in candidates
            if len(candidate.ordered_operands) == len(anchor.ordered_operands)
            and all(
                left != right
                for left, right in zip(anchor.ordered_operands, candidate.ordered_operands, strict=True)
            )
            and (anchor.target_value < 0) != (candidate.target_value < 0)
            and candidate.canonical_signature not in reserved
        )
        try:
            control = min(eligible, key=lambda candidate: _negative_selector_key(anchor, candidate))
        except ValueError as error:
            raise MebStructuralThresholdError(
                "MEB_STRUCTURAL_BOOTSTRAP_PANEL_UNDERSUPPLIED"
            ) from error
        reserved.add(control.canonical_signature)
        selected.append(control)
    return tuple(selected)


def _negative_selector_key(
    anchor: MebCandidate, candidate: MebCandidate
) -> tuple[bytes, bytes]:
    selector = {
        "domain": "meb_structcal_negative_selection_v1",
        "anchor_signature": anchor.canonical_signature,
        "ordered_operands": list(candidate.ordered_operands),
        "target_value": candidate.target_value,
    }
    return hashlib.sha256(canonical_json_bytes(selector)).digest(), candidate.candidate_bytes


def _control(anchor: MebCandidate, control: MebCandidate) -> MebStructuralControl:
    return MebStructuralControl(
        anchor_signature=anchor.canonical_signature,
        control_signature=control.canonical_signature,
        similarity=str(structural_similarity(_endpoint(anchor), _endpoint(control))),
    )


def _endpoint(candidate: MebCandidate) -> MebStructuralEndpoint:
    return MebStructuralEndpoint(
        ordered_operands=candidate.ordered_operands,
        target_value=candidate.target_value,
        signature=candidate.canonical_signature,
    )


def _levenshtein(left: tuple[int, ...], right: tuple[int, ...]) -> int:
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


__all__ = [
    "MebStructuralEndpoint",
    "MebStructuralThreshold",
    "MebStructuralThresholdError",
    "build_meb_structural_threshold",
    "near_duplicate_exclusions",
    "structural_similarity",
]
