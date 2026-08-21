from __future__ import annotations

from typing import Final

from pydantic import JsonValue

from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_builder import (
    build_candidate_evidence_for_candidate_v2,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_models import (
    BlindScore,
    BlindedPacket,
    CandidateEvidenceV2,
    CandidateEvidenceV2Error,
    CandidateId,
    EvaluatorRecord,
    SealedRoleMap,
    SelectedCandidateFreezeV2,
    canonical_hash,
    render_hashes_match,
    semantics,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_panel import (
    validate_evaluator_records,
    validate_panel_thresholds,
)
from memcontam.readiness.phase13_new_mcq_phase1_models import Phase1Freeze

_CANDIDATE_IDS: Final[dict[str, CandidateId]] = {
    "MCQ-H1-LEXICAL-OVERLAP-v1": "MCQ-H1-LEXICAL-OVERLAP-v1",
    "MCQ-H2-DETAIL-LENGTH-v1": "MCQ-H2-DETAIL-LENGTH-v1",
}


def build_candidate_evidence_v2(
    phase1: Phase1Freeze, construction_author_id: str
) -> CandidateEvidenceV2:
    _validate_phase1(phase1)
    return build_candidate_evidence_for_candidate_v2(
        phase1, construction_author_id, "MCQ-H1-LEXICAL-OVERLAP-v1"
    )


def validate_candidate_evidence_v2(evidence: CandidateEvidenceV2) -> CandidateEvidenceV2:
    packet = evidence.packet
    item_values: list[JsonValue] = [item.model_dump(mode="json") for item in packet.items]
    packet_payload: dict[str, JsonValue] = {
        "schema_version": packet.schema_version,
        "phase1_freeze_identity": packet.phase1_freeze_identity,
        "phase1_content_sha256": packet.phase1_content_sha256,
        "construction_author_id": packet.construction_author_id,
        "items": item_values,
    }
    role_map = evidence.sealed_role_map
    entry_values: list[JsonValue] = [entry.model_dump(mode="json") for entry in role_map.entries]
    role_payload: dict[str, JsonValue] = {
        "schema_version": role_map.schema_version,
        "packet_hash": role_map.packet_hash,
        "entries": entry_values,
    }
    item_ids = {item.opaque_render_id for item in packet.items}
    role_ids = {entry.opaque_render_id for entry in role_map.entries}
    role_keys = {(entry.task_id, entry.baseline_id, entry.role) for entry in role_map.entries}
    if (
        canonical_hash(packet_payload) != packet.packet_hash
        or role_map.packet_hash != packet.packet_hash
        or canonical_hash(role_payload) != role_map.role_map_hash
        or item_ids != role_ids
        or len(item_ids) != len(packet.items)
        or len(packet.items) != 45
        or len(role_keys) != len(role_map.entries)
        or len(role_map.entries) != 45
        or not all(render_hashes_match(item) for item in packet.items)
    ):
        raise CandidateEvidenceV2Error("CANDIDATE_EVIDENCE_HASH_MISMATCH")
    return evidence


def seal_evaluator_record(
    packet: BlindedPacket, evaluator_id: str, scores: tuple[BlindScore, ...]
) -> EvaluatorRecord:
    score_values: list[JsonValue] = [score.model_dump(mode="json") for score in scores]
    payload: dict[str, JsonValue] = {
        "schema_version": "phase13_new_mcq_evaluator_record_v2",
        "evaluator_id": evaluator_id,
        "packet_hash": packet.packet_hash,
        "scores": score_values,
    }
    return EvaluatorRecord(
        schema_version="phase13_new_mcq_evaluator_record_v2",
        evaluator_id=evaluator_id,
        packet_hash=packet.packet_hash,
        scores=scores,
        response_hash=canonical_hash(payload),
    )


def select_candidate_freeze_v2(
    evidence: CandidateEvidenceV2, records: tuple[EvaluatorRecord, ...]
) -> SelectedCandidateFreezeV2:
    validate_candidate_evidence_v2(evidence)
    validate_evaluator_records(evidence.packet, records)
    validate_panel_thresholds(evidence.sealed_role_map, records)
    selected = _selected_candidates(evidence.sealed_role_map)
    response_hashes = tuple(sorted(record.response_hash for record in records))
    selected_values: dict[str, JsonValue] = {}
    for task_id, candidate_id in selected.items():
        selected_values[task_id] = candidate_id
    payload: dict[str, JsonValue] = {
        "schema_version": "phase13_new_mcq_selected_candidate_freeze_v2",
        "phase1_freeze_identity": evidence.packet.phase1_freeze_identity,
        "packet_hash": evidence.packet.packet_hash,
        "sealed_role_map_hash": evidence.sealed_role_map.role_map_hash,
        "evaluator_response_hashes": list(response_hashes),
        "evaluator_count": len(records),
        "protocol_threshold_id": "phase13_protocol_v7_section_7_3_default",
        "mechanical_criteria_met": True,
        "panel_criteria_met": True,
        "selected_candidates": selected_values,
    }
    return SelectedCandidateFreezeV2(
        schema_version="phase13_new_mcq_selected_candidate_freeze_v2",
        phase1_freeze_identity=evidence.packet.phase1_freeze_identity,
        packet_hash=evidence.packet.packet_hash,
        sealed_role_map_hash=evidence.sealed_role_map.role_map_hash,
        evaluator_response_hashes=response_hashes,
        evaluator_count=len(records),
        protocol_threshold_id="phase13_protocol_v7_section_7_3_default",
        mechanical_criteria_met=True,
        panel_criteria_met=True,
        selected_candidates=selected,
        freeze_hash=canonical_hash(payload),
    )


def _validate_phase1(phase1: Phase1Freeze) -> None:
    for task in phase1.tasks.values():
        expected = semantics(task.selected_candidate_id)
        if (
            task.triplet.false_candidate_id != expected[0][1]
            or task.triplet.correct_twin_id != expected[1][1]
            or task.triplet.irrelevant_control_id != expected[2][1]
            or task.status != "MECHANICALLY_CERTIFIED_PENDING_BLINDED_PANEL"
        ):
            raise CandidateEvidenceV2Error("PHASE1_SEMANTIC_BINDING_INVALID")


def _candidate_id(semantic_id: str) -> CandidateId:
    candidate_id = _CANDIDATE_IDS.get(semantic_id)
    if candidate_id is None:
        raise CandidateEvidenceV2Error("PHASE1_SEMANTIC_BINDING_INVALID")
    return candidate_id


def _selected_candidates(role_map: SealedRoleMap) -> dict[CoreTask, CandidateId]:
    selected: dict[CoreTask, CandidateId] = {}
    for entry in role_map.entries:
        if entry.role != "false":
            continue
        candidate_id = _candidate_id(entry.semantic_id)
        existing = selected.get(entry.task_id)
        if existing is not None and existing != candidate_id:
            raise CandidateEvidenceV2Error("PHASE1_SEMANTIC_BINDING_INVALID")
        selected[entry.task_id] = candidate_id
    if len(selected) != 3:
        raise CandidateEvidenceV2Error("PHASE1_SEMANTIC_BINDING_INVALID")
    return selected


__all__ = [
    "build_candidate_evidence_v2",
    "seal_evaluator_record",
    "select_candidate_freeze_v2",
    "validate_candidate_evidence_v2",
]
