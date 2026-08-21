from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_models import (
    CandidateId,
    Sha256,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class PanelRejectionSummaryV2(_FrozenModel):
    schema_version: Literal["phase13_new_mcq_panel_rejection_summary_v2"]
    phase1_freeze_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    phase1_content_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    packet_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    sealed_role_map_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_response_hashes: tuple[Sha256, ...]
    evaluator_count: int = Field(ge=3)
    protocol_threshold_id: Literal["phase13_protocol_v7_section_7_3_default"]
    rejected_candidates: dict[CoreTask, CandidateId]
    summary_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateFallbackFreezeV2(_FrozenModel):
    schema_version: Literal["phase13_new_mcq_candidate_fallback_freeze_v2"]
    phase1_freeze_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    h1_rejection_summary_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    h1_packet_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    h2_packet_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    h2_sealed_role_map_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    h2_evaluator_response_hashes: tuple[Sha256, ...]
    evaluator_count: int = Field(ge=3)
    protocol_threshold_id: Literal["phase13_protocol_v7_section_7_3_default"]
    task_decisions: dict[CoreTask, CandidateId | Literal["NOT_READY"]]
    freeze_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
