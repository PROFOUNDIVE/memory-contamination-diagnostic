from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest

from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2 import (
    build_candidate_evidence_v2,
    seal_evaluator_record,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_fallback import (
    build_h2_candidate_evidence_v2,
    seal_panel_rejection_v2,
    select_h2_fallback_freeze_v2,
)
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_models import (
    BlindScore,
    CandidateEvidenceV2,
    CandidateEvidenceV2Error,
    EvaluatorRecord,
    canonical_hash,
)
from memcontam.readiness.phase13_new_mcq_phase1_freeze import (
    Phase1SourcePaths,
    build_phase1_freeze,
)
from memcontam.readiness.phase13_new_mcq_phase1_models import Phase1Freeze

MMLU_SOURCE = Path("data/phase13/rag/new_mcq/sources/mmlu_pro_validation_475d58ba.parquet")
GPQA_MAIN = Path("data/phase13/core/materialized/gpqa_main_certification_633f5ee8.csv")
GPQA_DIAMOND = Path("data/phase13/core/materialized/gpqa_diamond.jsonl")


@pytest.fixture(scope="module")
def phase1() -> Phase1Freeze:
    return build_phase1_freeze(
        Phase1SourcePaths(
            mmlu_source=MMLU_SOURCE,
            gpqa_main_source=GPQA_MAIN,
            gpqa_diamond_evaluation=GPQA_DIAMOND,
            frozen_at="2026-08-20T00:00:00Z",
        )
    )


@pytest.fixture(scope="module")
def h1_evidence(phase1: Phase1Freeze) -> CandidateEvidenceV2:
    return build_candidate_evidence_v2(phase1, "phase13-construction-author")


def _records(evidence: CandidateEvidenceV2, plausibility: int) -> tuple[EvaluatorRecord, ...]:
    scores = tuple(
        BlindScore(
            opaque_render_id=item.opaque_render_id,
            naturalness=3,
            native_likeness=3,
            plausibility=plausibility,
            specificity=3,
            excessive_adversarial_phrasing=2,
        )
        for item in evidence.packet.items
    )
    return tuple(
        seal_evaluator_record(evidence.packet, f"independent-session-{index}", scores)
        for index in range(3)
    )


def test_h2_requires_a_sealed_all_task_h1_rejection(
    phase1: Phase1Freeze, h1_evidence: CandidateEvidenceV2
) -> None:
    with pytest.raises(CandidateEvidenceV2Error, match="H1_PANEL_NOT_REJECTED_FOR_ALL_TASKS"):
        seal_panel_rejection_v2(h1_evidence, _records(h1_evidence, plausibility=3))


def test_all_task_h1_rejection_builds_source_bound_h2_only_packet(
    phase1: Phase1Freeze, h1_evidence: CandidateEvidenceV2
) -> None:
    rejection = seal_panel_rejection_v2(
        h1_evidence, _records(h1_evidence, plausibility=2)
    )

    h2_evidence = build_h2_candidate_evidence_v2(
        phase1, "phase13-construction-author", rejection
    )
    semantic_ids = {entry.semantic_id for entry in h2_evidence.sealed_role_map.entries}

    assert h2_evidence.packet.phase1_content_sha256 == h1_evidence.packet.phase1_content_sha256
    assert h2_evidence.packet.packet_hash != h1_evidence.packet.packet_hash
    assert set(rejection.rejected_candidates.values()) == {"MCQ-H1-LEXICAL-OVERLAP-v1"}
    assert semantic_ids == {
        "MCQ-H2-DETAIL-LENGTH-v1",
        "MCQ-H2-CORRECT-SUBSTANTIVE-CONTENT-v1",
        "MCQ-I1-SINGLETON-OPTION-v1",
    }


def test_h2_fallback_uses_a_versioned_surface_revision_without_changing_roles(
    phase1: Phase1Freeze, h1_evidence: CandidateEvidenceV2
) -> None:
    rejection = seal_panel_rejection_v2(
        h1_evidence, _records(h1_evidence, plausibility=2)
    )

    h2_evidence = build_h2_candidate_evidence_v2(
        phase1, "phase13-construction-author", rejection
    )
    h1_versions = {
        json.loads(base64.b64decode(item.render_bytes_base64))["config"]["renderer_version"]
        for item in h1_evidence.packet.items
    }
    h2_versions = {
        json.loads(base64.b64decode(item.render_bytes_base64))["config"]["renderer_version"]
        for item in h2_evidence.packet.items
    }
    h2_false_ids = {
        entry.semantic_id for entry in h2_evidence.sealed_role_map.entries if entry.role == "false"
    }
    roles = {
        entry.opaque_render_id: entry.role for entry in h2_evidence.sealed_role_map.entries
    }
    rendered = {
        (item.task_id, item.baseline_id, roles[item.opaque_render_id]): json.loads(
            base64.b64decode(item.render_bytes_base64)
        )["native_entry"]
        for item in h2_evidence.packet.items
    }

    assert h1_versions == {"phase13_new_mcq_native_renderer_v2"}
    assert h1_evidence.packet.packet_hash == (
        "86f4f3a4af92fc35d8ba909331c5aaa72684cff9d9e25ddaae09b27ae6e7c2f3"
    )
    assert h2_versions == {"phase13_new_mcq_native_renderer_h2_surface_v4"}
    assert h2_false_ids == {"MCQ-H2-DETAIL-LENGTH-v1"}
    for task_id, baseline_id, role in rendered:
        if role != "false":
            continue
        false_text = json.dumps(rendered[(task_id, baseline_id, "false")], sort_keys=True)
        correct_text = json.dumps(rendered[(task_id, baseline_id, "correct")], sort_keys=True)
        false_tokens = set(re.findall(r"[a-z]+", false_text.lower()))
        correct_tokens = set(re.findall(r"[a-z]+", correct_text.lower()))
        assert len(false_tokens & correct_tokens) / len(false_tokens | correct_tokens) >= 0.5
        assert abs(len(false_text) - len(correct_text)) <= 24


def test_h2_rejects_changed_phase1_content_with_retained_freeze_identity(
    phase1: Phase1Freeze, h1_evidence: CandidateEvidenceV2
) -> None:
    rejection = seal_panel_rejection_v2(
        h1_evidence, _records(h1_evidence, plausibility=2)
    )
    task = phase1.tasks["mmlu_pro_engineering"]
    changed_task = task.model_copy(update={"challenge_suite_key": "0" * 64})
    changed_phase1 = phase1.model_copy(
        update={"tasks": {**phase1.tasks, "mmlu_pro_engineering": changed_task}}
    )

    with pytest.raises(CandidateEvidenceV2Error, match="H1_REJECTION_SOURCE_MISMATCH"):
        build_h2_candidate_evidence_v2(
            changed_phase1, "phase13-construction-author", rejection
        )


def test_h2_is_selected_only_after_every_task_panel_passes(
    phase1: Phase1Freeze, h1_evidence: CandidateEvidenceV2
) -> None:
    rejection = seal_panel_rejection_v2(
        h1_evidence, _records(h1_evidence, plausibility=2)
    )
    h2_evidence = build_h2_candidate_evidence_v2(
        phase1, "phase13-construction-author", rejection
    )

    selected = select_h2_fallback_freeze_v2(
        h2_evidence, _records(h2_evidence, plausibility=3), rejection
    )

    assert set(selected.task_decisions.values()) == {"MCQ-H2-DETAIL-LENGTH-v1"}
    assert selected.h1_rejection_summary_hash == rejection.summary_hash
    assert selected.h1_packet_hash == rejection.packet_hash


def test_h2_selection_rejects_phase1_content_mismatch(
    phase1: Phase1Freeze, h1_evidence: CandidateEvidenceV2
) -> None:
    rejection = seal_panel_rejection_v2(
        h1_evidence, _records(h1_evidence, plausibility=2)
    )
    h2_evidence = build_h2_candidate_evidence_v2(
        phase1, "phase13-construction-author", rejection
    )
    packet_payload = {
        "schema_version": h2_evidence.packet.schema_version,
        "phase1_freeze_identity": h2_evidence.packet.phase1_freeze_identity,
        "phase1_content_sha256": "0" * 64,
        "construction_author_id": h2_evidence.packet.construction_author_id,
        "items": [item.model_dump(mode="json") for item in h2_evidence.packet.items],
    }
    packet = h2_evidence.packet.model_copy(
        update={"phase1_content_sha256": "0" * 64, "packet_hash": canonical_hash(packet_payload)}
    )
    role_payload = {
        "schema_version": h2_evidence.sealed_role_map.schema_version,
        "packet_hash": packet.packet_hash,
        "entries": [
            entry.model_dump(mode="json") for entry in h2_evidence.sealed_role_map.entries
        ],
    }
    role_map = h2_evidence.sealed_role_map.model_copy(
        update={
            "packet_hash": packet.packet_hash,
            "role_map_hash": canonical_hash(role_payload),
        }
    )
    changed = h2_evidence.model_copy(update={"packet": packet, "sealed_role_map": role_map})

    with pytest.raises(CandidateEvidenceV2Error, match="H2_EVIDENCE_BINDING_INVALID"):
        select_h2_fallback_freeze_v2(changed, _records(changed, plausibility=3), rejection)


def test_failed_h2_task_is_not_ready_without_baseline_mixing(
    phase1: Phase1Freeze, h1_evidence: CandidateEvidenceV2
) -> None:
    rejection = seal_panel_rejection_v2(
        h1_evidence, _records(h1_evidence, plausibility=2)
    )
    h2_evidence = build_h2_candidate_evidence_v2(
        phase1, "phase13-construction-author", rejection
    )
    task_id = "mmlu_pro_engineering"
    task_render_ids = {
        entry.opaque_render_id
        for entry in h2_evidence.sealed_role_map.entries
        if entry.task_id == task_id and entry.role == "false"
    }
    records = tuple(
        record.model_copy(
            update={
                "scores": tuple(
                    score.model_copy(update={"plausibility": 2})
                    if score.opaque_render_id in task_render_ids
                    else score
                    for score in record.scores
                )
            }
        )
        for record in _records(h2_evidence, plausibility=3)
    )
    records = tuple(
        seal_evaluator_record(h2_evidence.packet, record.evaluator_id, record.scores)
        for record in records
    )

    selected = select_h2_fallback_freeze_v2(h2_evidence, records, rejection)

    assert selected.task_decisions[task_id] == "NOT_READY"
    assert {
        decision for task, decision in selected.task_decisions.items() if task != task_id
    } == {"MCQ-H2-DETAIL-LENGTH-v1"}
