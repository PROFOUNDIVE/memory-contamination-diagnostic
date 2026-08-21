from __future__ import annotations

from pydantic import JsonValue

from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_models import (
    BlindedPacket,
    CandidateEvidenceV2,
    CandidateId,
    SealedRoleMap,
    canonical_hash,
    render_candidate_triplets_for_candidate,
)
from memcontam.readiness.phase13_new_mcq_phase1_models import Phase1Freeze


def build_candidate_evidence_for_candidate_v2(
    phase1: Phase1Freeze, construction_author_id: str, candidate_id: CandidateId
) -> CandidateEvidenceV2:
    items, entries = render_candidate_triplets_for_candidate(phase1, candidate_id)
    phase1_content_sha256 = canonical_hash(phase1.model_dump(mode="json"))
    item_values: list[JsonValue] = [item.model_dump(mode="json") for item in items]
    packet_payload: dict[str, JsonValue] = {
        "schema_version": "phase13_new_mcq_blinded_packet_v2",
        "phase1_freeze_identity": phase1.freeze_identity,
        "phase1_content_sha256": phase1_content_sha256,
        "construction_author_id": construction_author_id,
        "items": item_values,
    }
    packet = BlindedPacket(
        schema_version="phase13_new_mcq_blinded_packet_v2",
        phase1_freeze_identity=phase1.freeze_identity,
        phase1_content_sha256=phase1_content_sha256,
        construction_author_id=construction_author_id,
        items=items,
        packet_hash=canonical_hash(packet_payload),
    )
    entry_values: list[JsonValue] = [entry.model_dump(mode="json") for entry in entries]
    role_payload: dict[str, JsonValue] = {
        "schema_version": "phase13_new_mcq_sealed_role_map_v2",
        "packet_hash": packet.packet_hash,
        "entries": entry_values,
    }
    return CandidateEvidenceV2(
        schema_version="phase13_new_mcq_candidate_evidence_v2",
        packet=packet,
        sealed_role_map=SealedRoleMap(
            schema_version="phase13_new_mcq_sealed_role_map_v2",
            packet_hash=packet.packet_hash,
            entries=entries,
            role_map_hash=canonical_hash(role_payload),
        ),
    )
