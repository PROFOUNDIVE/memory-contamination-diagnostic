from __future__ import annotations

from typing import Final, Literal

from pydantic import JsonValue

from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2 import (
    validate_candidate_evidence_v2,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_builder import (
    build_candidate_evidence_for_candidate_v2,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_fallback_models import (
    CandidateFallbackFreezeV2,
    PanelRejectionSummaryV2,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_models import (
    CandidateEvidenceV2,
    CandidateEvidenceV2Error,
    CandidateId,
    EvaluatorRecord,
    SealedRoleMap,
    canonical_hash,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_panel import (
    panel_results_by_task,
    validate_evaluator_records,
)
from memcontam.readiness.phase13_new_mcq_phase1_models import Phase1Freeze

_H1: Final[CandidateId] = "MCQ-H1-LEXICAL-OVERLAP-v1"
_H2: Final[CandidateId] = "MCQ-H2-DETAIL-LENGTH-v1"
_TASKS: Final[frozenset[CoreTask]] = frozenset(
    {"mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond"}
)


def seal_panel_rejection_v2(
    evidence: CandidateEvidenceV2, records: tuple[EvaluatorRecord, ...]
) -> PanelRejectionSummaryV2:
    validate_candidate_evidence_v2(evidence)
    validate_evaluator_records(evidence.packet, records)
    rejected = _candidate_ids_by_task(evidence.sealed_role_map)
    panel_results = panel_results_by_task(evidence.sealed_role_map, records)
    if (
        set(rejected) != _TASKS
        or set(rejected.values()) != {_H1}
        or set(panel_results) != _TASKS
        or any(panel_results.values())
    ):
        raise CandidateEvidenceV2Error("H1_PANEL_NOT_REJECTED_FOR_ALL_TASKS")
    response_hashes = tuple(sorted(record.response_hash for record in records))
    payload = _rejection_payload(
        evidence.packet.phase1_freeze_identity,
        evidence.packet.phase1_content_sha256,
        evidence.packet.packet_hash,
        evidence.sealed_role_map.role_map_hash,
        response_hashes,
        len(records),
        rejected,
    )
    return PanelRejectionSummaryV2(
        schema_version="phase13_new_mcq_panel_rejection_summary_v2",
        phase1_freeze_identity=evidence.packet.phase1_freeze_identity,
        phase1_content_sha256=evidence.packet.phase1_content_sha256,
        packet_hash=evidence.packet.packet_hash,
        sealed_role_map_hash=evidence.sealed_role_map.role_map_hash,
        evaluator_response_hashes=response_hashes,
        evaluator_count=len(records),
        protocol_threshold_id="phase13_protocol_v7_section_7_3_default",
        rejected_candidates=rejected,
        summary_hash=canonical_hash(payload),
    )


def build_h2_candidate_evidence_v2(
    phase1: Phase1Freeze,
    construction_author_id: str,
    h1_rejection: PanelRejectionSummaryV2,
) -> CandidateEvidenceV2:
    _validate_rejection(h1_rejection)
    if (
        h1_rejection.phase1_freeze_identity != phase1.freeze_identity
        or h1_rejection.phase1_content_sha256
        != canonical_hash(phase1.model_dump(mode="json"))
    ):
        raise CandidateEvidenceV2Error("H1_REJECTION_SOURCE_MISMATCH")
    return build_candidate_evidence_for_candidate_v2(phase1, construction_author_id, _H2)


def select_h2_fallback_freeze_v2(
    evidence: CandidateEvidenceV2,
    records: tuple[EvaluatorRecord, ...],
    h1_rejection: PanelRejectionSummaryV2,
) -> CandidateFallbackFreezeV2:
    _validate_rejection(h1_rejection)
    validate_candidate_evidence_v2(evidence)
    validate_evaluator_records(evidence.packet, records)
    candidates = _candidate_ids_by_task(evidence.sealed_role_map)
    if (
        evidence.packet.phase1_freeze_identity != h1_rejection.phase1_freeze_identity
        or evidence.packet.phase1_content_sha256 != h1_rejection.phase1_content_sha256
        or set(candidates) != _TASKS
        or set(candidates.values()) != {_H2}
    ):
        raise CandidateEvidenceV2Error("H2_EVIDENCE_BINDING_INVALID")
    panel_results = panel_results_by_task(evidence.sealed_role_map, records)
    if set(panel_results) != _TASKS:
        raise CandidateEvidenceV2Error("H2_EVIDENCE_BINDING_INVALID")
    decisions: dict[CoreTask, CandidateId | Literal["NOT_READY"]] = {
        task_id: _H2 if passed else "NOT_READY"
        for task_id, passed in panel_results.items()
    }
    response_hashes = tuple(sorted(record.response_hash for record in records))
    decision_values: dict[str, JsonValue] = {
        task_id: decision for task_id, decision in decisions.items()
    }
    payload: dict[str, JsonValue] = {
        "schema_version": "phase13_new_mcq_candidate_fallback_freeze_v2",
        "phase1_freeze_identity": evidence.packet.phase1_freeze_identity,
        "h1_rejection_summary_hash": h1_rejection.summary_hash,
        "h1_packet_hash": h1_rejection.packet_hash,
        "h2_packet_hash": evidence.packet.packet_hash,
        "h2_sealed_role_map_hash": evidence.sealed_role_map.role_map_hash,
        "h2_evaluator_response_hashes": list(response_hashes),
        "evaluator_count": len(records),
        "protocol_threshold_id": "phase13_protocol_v7_section_7_3_default",
        "task_decisions": decision_values,
    }
    return CandidateFallbackFreezeV2(
        schema_version="phase13_new_mcq_candidate_fallback_freeze_v2",
        phase1_freeze_identity=evidence.packet.phase1_freeze_identity,
        h1_rejection_summary_hash=h1_rejection.summary_hash,
        h1_packet_hash=h1_rejection.packet_hash,
        h2_packet_hash=evidence.packet.packet_hash,
        h2_sealed_role_map_hash=evidence.sealed_role_map.role_map_hash,
        h2_evaluator_response_hashes=response_hashes,
        evaluator_count=len(records),
        protocol_threshold_id="phase13_protocol_v7_section_7_3_default",
        task_decisions=decisions,
        freeze_hash=canonical_hash(payload),
    )


def _candidate_ids_by_task(role_map: SealedRoleMap) -> dict[CoreTask, CandidateId]:
    candidates: dict[CoreTask, CandidateId] = {}
    known: Final[dict[str, CandidateId]] = {_H1: _H1, _H2: _H2}
    for entry in role_map.entries:
        if entry.role != "false":
            continue
        candidate = known.get(entry.semantic_id)
        if candidate is None:
            raise CandidateEvidenceV2Error("CANDIDATE_SEMANTIC_BINDING_INVALID")
        existing = candidates.get(entry.task_id)
        if existing is not None and existing != candidate:
            raise CandidateEvidenceV2Error("CANDIDATE_SEMANTIC_BINDING_INVALID")
        candidates[entry.task_id] = candidate
    return candidates


def _rejection_payload(
    phase1_freeze_identity: str,
    phase1_content_sha256: str,
    packet_hash: str,
    role_map_hash: str,
    response_hashes: tuple[str, ...],
    evaluator_count: int,
    rejected: dict[CoreTask, CandidateId],
) -> dict[str, JsonValue]:
    rejected_values: dict[str, JsonValue] = {
        task_id: candidate_id for task_id, candidate_id in rejected.items()
    }
    return {
        "schema_version": "phase13_new_mcq_panel_rejection_summary_v2",
        "phase1_freeze_identity": phase1_freeze_identity,
        "phase1_content_sha256": phase1_content_sha256,
        "packet_hash": packet_hash,
        "sealed_role_map_hash": role_map_hash,
        "evaluator_response_hashes": list(response_hashes),
        "evaluator_count": evaluator_count,
        "protocol_threshold_id": "phase13_protocol_v7_section_7_3_default",
        "rejected_candidates": rejected_values,
    }


def _validate_rejection(summary: PanelRejectionSummaryV2) -> None:
    payload = _rejection_payload(
        summary.phase1_freeze_identity,
        summary.phase1_content_sha256,
        summary.packet_hash,
        summary.sealed_role_map_hash,
        summary.evaluator_response_hashes,
        summary.evaluator_count,
        summary.rejected_candidates,
    )
    if (
        canonical_hash(payload) != summary.summary_hash
        or summary.evaluator_count != len(summary.evaluator_response_hashes)
        or set(summary.rejected_candidates) != _TASKS
        or set(summary.rejected_candidates.values()) != {_H1}
    ):
        raise CandidateEvidenceV2Error("H1_REJECTION_SUMMARY_INVALID")


__all__ = [
    "build_h2_candidate_evidence_v2",
    "seal_panel_rejection_v2",
    "select_h2_fallback_freeze_v2",
]
