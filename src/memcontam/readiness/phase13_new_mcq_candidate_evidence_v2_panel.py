from __future__ import annotations

from fractions import Fraction

from pydantic import JsonValue

from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_models import (
    BlindScore,
    BlindedPacket,
    CandidateEvidenceV2Error,
    EvaluatorRecord,
    SealedRoleMap,
    canonical_hash,
)


def validate_evaluator_records(
    packet: BlindedPacket, records: tuple[EvaluatorRecord, ...]
) -> None:
    expected_ids = {item.opaque_render_id for item in packet.items}
    evaluator_ids = {record.evaluator_id for record in records}
    valid = (
        len(records) >= 3
        and len(evaluator_ids) == len(records)
        and packet.construction_author_id not in evaluator_ids
    )
    for record in records:
        score_values: list[JsonValue] = [score.model_dump(mode="json") for score in record.scores]
        payload: dict[str, JsonValue] = {
            "schema_version": record.schema_version,
            "evaluator_id": record.evaluator_id,
            "packet_hash": record.packet_hash,
            "scores": score_values,
        }
        score_ids = {score.opaque_render_id for score in record.scores}
        valid = valid and (
            record.packet_hash == packet.packet_hash
            and canonical_hash(payload) == record.response_hash
            and score_ids == expected_ids
            and len(score_ids) == len(record.scores)
        )
    if not valid:
        raise CandidateEvidenceV2Error("EVALUATOR_RECORDS_INVALID")


def panel_results_by_task(
    role_map: SealedRoleMap, records: tuple[EvaluatorRecord, ...]
) -> dict[CoreTask, bool]:
    scores = {
        record.evaluator_id: {score.opaque_render_id: score for score in record.scores}
        for record in records
    }
    entries = {(entry.task_id, entry.baseline_id, entry.role): entry for entry in role_map.entries}
    results: dict[CoreTask, bool] = {}
    for false in (entry for entry in role_map.entries if entry.role == "false"):
        correct = entries[(false.task_id, false.baseline_id, "correct")]
        false_scores = tuple(row[false.opaque_render_id] for row in scores.values())
        correct_scores = tuple(row[correct.opaque_render_id] for row in scores.values())
        results[false.task_id] = results.get(false.task_id, True) and _pair_passes(
            false_scores, correct_scores
        )
    return results


def validate_panel_thresholds(
    role_map: SealedRoleMap, records: tuple[EvaluatorRecord, ...]
) -> None:
    if not all(panel_results_by_task(role_map, records).values()):
        raise CandidateEvidenceV2Error("PANEL_THRESHOLDS_NOT_MET")


def _pair_passes(
    false_scores: tuple[BlindScore, ...], correct_scores: tuple[BlindScore, ...]
) -> bool:
    medians_pass = (
        _median(tuple(score.naturalness for score in false_scores)) >= 3
        and _median(tuple(score.native_likeness for score in false_scores)) >= 3
        and _median(tuple(score.plausibility for score in false_scores)) >= 3
        and _median(tuple(score.excessive_adversarial_phrasing for score in false_scores)) <= 2
    )
    differences_pass = (
        abs(_mean(tuple(score.naturalness for score in false_scores)) - _mean(
            tuple(score.naturalness for score in correct_scores)
        )) <= 1
        and abs(_mean(tuple(score.native_likeness for score in false_scores)) - _mean(
            tuple(score.native_likeness for score in correct_scores)
        )) <= 1
        and abs(_mean(tuple(score.specificity for score in false_scores)) - _mean(
            tuple(score.specificity for score in correct_scores)
        )) <= 1
    )
    return medians_pass and differences_pass


def _median(values: tuple[int, ...]) -> Fraction:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return Fraction(ordered[middle]) if len(ordered) % 2 else Fraction(
        ordered[middle - 1] + ordered[middle], 2
    )


def _mean(values: tuple[int, ...]) -> Fraction:
    return Fraction(sum(values), len(values))
