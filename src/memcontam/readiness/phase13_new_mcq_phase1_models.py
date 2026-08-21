from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from memcontam.readiness.phase13_core_bundle import CORE_TASKS, CoreTask

Sha256 = str
EvidenceSplit = Literal["build", "calibration"]
BaselineId = Literal[
    "FH-bounded",
    "RAG-Frozen",
    "BoT-style",
    "Reflexion-style",
    "DC-RS adapted",
]
NativeKind = Literal["raw_interaction", "retrieved_document", "thought_template", "reflection"]
CorrectId = Literal[
    "MCQ-H1-CORRECT-SUBSTANTIVE-CONTENT-v1",
    "MCQ-H2-CORRECT-SUBSTANTIVE-CONTENT-v1",
]
ApplicabilityId = Literal[
    "MCQ-H1-UNIQUE-MAX-APP-v1", "MCQ-H2-UNIQUE-MAX-APP-v1"
]
CertificationId = Literal[
    "MCQ-H1-GOLD-DISAGREEMENT-CERT-v1", "MCQ-H2-GOLD-DISAGREEMENT-CERT-v1"
]
_RENDERERS: Final[tuple[tuple[BaselineId, NativeKind], ...]] = (
    ("FH-bounded", "raw_interaction"),
    ("RAG-Frozen", "retrieved_document"),
    ("BoT-style", "thought_template"),
    ("Reflexion-style", "reflection"),
    ("DC-RS adapted", "raw_interaction"),
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Phase1SourcePaths(_FrozenModel):
    mmlu_source: Path
    gpqa_main_source: Path
    gpqa_diamond_evaluation: Path
    gpqa_extended_source: Path | None = None
    gpqa_extended_sha256: str | None = None
    frozen_at: str


class Phase1SourceIdentity(_FrozenModel):
    upstream_dataset_identity: str
    upstream_revision: str
    selected_source_config: Literal["mmlu_pro_validation", "gpqa_main", "gpqa_extended"]
    source_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_source_sha256: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    excluded_canonical_rows: int = Field(ge=0)
    eligible_rows: int = Field(gt=0)
    split_law_id: Literal["phase13_source_identity_hash_50_50_v1"]
    split_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")


class SplitCertification(_FrozenModel):
    split: EvidenceSplit
    split_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(gt=0)
    applicable_rows: int = Field(gt=0)
    counterexample_rows: int = Field(gt=0)


class OpaqueCertificationRecord(_FrozenModel):
    opaque_item_id: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_local_sample_id: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    permitted_content_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    displayed_option_permutation_id: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    split: EvidenceSplit
    option_count: int = Field(gt=1)
    applicable_audit_only: bool
    counterexample_audit_only: bool


class TripletSemantics(_FrozenModel):
    false_candidate_id: Literal["MCQ-H1-LEXICAL-OVERLAP-v1", "MCQ-H2-DETAIL-LENGTH-v1"]
    correct_twin_id: Literal[
        "MCQ-H1-CORRECT-SUBSTANTIVE-CONTENT-v1",
        "MCQ-H2-CORRECT-SUBSTANTIVE-CONTENT-v1",
    ]
    irrelevant_control_id: Literal["MCQ-I1-SINGLETON-OPTION-v1"]
    correct_shares_false_applicability: Literal[True]
    irrelevant_is_valid: Literal[True]
    irrelevant_is_inapplicable_to_target: Literal[True]


class TaskLocalRelevance(_FrozenModel):
    false_relation: Literal["MCQ-H1-UNIQUE-MAX-APP-v1", "MCQ-H2-UNIQUE-MAX-APP-v1"]
    correct_relation: Literal["MCQ-H1-UNIQUE-MAX-APP-v1", "MCQ-H2-UNIQUE-MAX-APP-v1"]
    irrelevant_relation: Literal["MCQ-I1-SINGLETON-APP-v1"]
    relation_scope: Literal["task_local_frozen_displayed_items"]
    realized_retrieval_excluded: Literal[True]


class RenderPacketIdentity(_FrozenModel):
    baseline_id: BaselineId
    native_kind: NativeKind
    native_schema_version: str
    cell_status: Literal[
        "CONSTRUCTION_IDENTITY_FROZEN",
        "NOT_READY_NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN",
    ]
    false_render_id: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    correct_render_id: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    irrelevant_render_id: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    role_invariant_query_identity: Sha256 | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    packet_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")


class TaskPhase1Certification(_FrozenModel):
    task_id: CoreTask
    status: Literal["MECHANICALLY_CERTIFIED_PENDING_BLINDED_PANEL"]
    source: Phase1SourceIdentity
    pool_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_family_id: Literal["MCQ-SURFACE-CUE-SUFFICIENCY-v1"]
    selected_candidate_id: Literal[
        "MCQ-H1-LEXICAL-OVERLAP-v1", "MCQ-H2-DETAIL-LENGTH-v1"
    ]
    mechanical_spec_id: Literal["MCQ-SURFACE-HEURISTICS-v1.0.0"]
    normalizer_id: Literal["MCQ-NORM-NFKC-CASEFOLD-WS-v1"]
    tokenizer_id: Literal["MCQ-TOK-UNICODE-LNM-RUN-v1"]
    overlap_metric_id: Literal["MCQ-H1-JACCARD-DISTINCT-TOKENS-v1"]
    length_detail_rule_id: Literal["MCQ-DETAIL-TOKEN-CODEPOINT-LEX-v1"]
    applicability_implementation_id: Literal["MCQ-SURFACE-HEURISTICS-v1.0.0"]
    counterexample_certification_id: Literal[
        "MCQ-H1-GOLD-DISAGREEMENT-CERT-v1", "MCQ-H2-GOLD-DISAGREEMENT-CERT-v1"
    ]
    certification_suite_id: Literal["MCQ-HEURISTIC-CERT-v1"]
    unicode_data_manifest_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_source_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    test_vector_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    build: SplitCertification
    calibration: SplitCertification
    records: tuple[OpaqueCertificationRecord, ...]
    triplet: TripletSemantics
    relevance: TaskLocalRelevance
    render_packets: tuple[RenderPacketIdentity, ...]
    challenge_suite_key: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    certification_record_hash: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_freeze_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at: str


class Phase1Freeze(_FrozenModel):
    schema_version: Literal["phase13_new_mcq_phase1_freeze_v1"]
    authority_stack: tuple[str, str, str, str]
    candidate_pool_frozen_before_certification: Literal[True]
    candidate_pool_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    tasks: dict[CoreTask, TaskPhase1Certification]
    frozen_at: str
    freeze_identity: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_tasks(self) -> Phase1Freeze:
        if set(self.tasks) != set(CORE_TASKS):
            msg = "Phase-1 freeze must contain exactly the three new MCQ tasks"
            raise ValueError(msg)
        return self


def render_packet_identities(
    task_id: str, candidate_id: str, correct_id: str
) -> tuple[RenderPacketIdentity, ...]:
    packets: list[RenderPacketIdentity] = []
    for baseline_id, native_kind in _RENDERERS:
        base = {
            "task": task_id,
            "baseline": baseline_id,
            "native_kind": native_kind,
            "native_schema_version": "phase13_baseline_native_render_v1",
        }
        false_id = _hash({**base, "role": "false", "semantic_id": candidate_id})
        correct_render_id = _hash({**base, "role": "correct", "semantic_id": correct_id})
        irrelevant_id = _hash(
            {**base, "role": "irrelevant", "semantic_id": "MCQ-I1-SINGLETON-OPTION-v1"}
        )
        query_identity = (
            _hash({**base, "role_invariant_query": "matched_triplet"})
            if baseline_id == "DC-RS adapted"
            else None
        )
        packet_id = _hash(
            {
                **base,
                "false": false_id,
                "correct": correct_render_id,
                "irrelevant": irrelevant_id,
                "role_invariant_query": query_identity,
            }
        )
        packets.append(
            RenderPacketIdentity(
                baseline_id=baseline_id,
                native_kind=native_kind,
                native_schema_version="phase13_baseline_native_render_v1",
                cell_status=(
                    "NOT_READY_NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"
                    if baseline_id == "RAG-Frozen"
                    else "CONSTRUCTION_IDENTITY_FROZEN"
                ),
                false_render_id=false_id,
                correct_render_id=correct_render_id,
                irrelevant_render_id=irrelevant_id,
                role_invariant_query_identity=query_identity,
                packet_identity=packet_id,
            )
        )
    return tuple(packets)


def _hash(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "OpaqueCertificationRecord",
    "Phase1Freeze",
    "Phase1SourcePaths",
    "Phase1SourceIdentity",
    "RenderPacketIdentity",
    "SplitCertification",
    "TaskLocalRelevance",
    "TaskPhase1Certification",
    "TripletSemantics",
    "render_packet_identities",
]
