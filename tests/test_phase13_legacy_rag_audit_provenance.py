from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.readiness.phase13_legacy_rag_audit import build_opaque_exclusion_registry
from memcontam.readiness.phase13_legacy_rag_calibration import (
    build_meb_calibration_registry,
    calibration_artifact_sha256,
)
from memcontam.readiness.phase13_legacy_rag_construction import (
    BuildRegistrySource,
    build_registry,
)
from memcontam.readiness.phase13_legacy_rag_documents import clean_documents
from memcontam.readiness.phase13_legacy_rag_generators import meb_candidates


ROOT = Path(__file__).resolve().parents[1]


def test_auditor_rejects_unregistered_main_source_identity(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "game24_main_v1.jsonl").write_text(
        json.dumps({"numbers": [3, 3, 8, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )
    (evaluation / "word_sorting_main_v1.jsonl").write_text(
        json.dumps({"words": ["ayz", "aza"]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="LEGACY_RAG_MAIN_SOURCE_IDENTITY_MISMATCH"):
        build_opaque_exclusion_registry(evaluation, tmp_path / "opaque.json")


def test_meb_provenance_binds_authoritative_identities() -> None:
    eligible = meb_candidates(frozenset(), limit=80)
    calibration = build_meb_calibration_registry(eligible[:16])
    registry = build_registry(
        BuildRegistrySource(
            repository_root=ROOT,
            task="math_equation_balancer",
            calibration_hashes=tuple(row.canonical_signature for row in eligible[:16]),
            evaluation_exclusion_hashes=(),
            calibration_registry_path="math_equation_balancer/calibration_registry.json",
            calibration_registry_id=calibration.registry_id,
            calibration_registry_sha256=calibration_artifact_sha256(calibration),
            calibration_selection_law=calibration.selection_law,
            build_partition_law=calibration.partition_law,
            historical_pilot_status=calibration.historical_rhs_completion_pilot.status,
            leakage_calibration_artifact=None,
            opaque_hash="0" * 64,
            candidates=eligible[16:],
        )
    )
    worked = [
        row
        for row in clean_documents("math_equation_balancer", eligible[16:], registry)
        if row.semantic_stratum == "D"
    ]

    assert registry.generator.generator_id == "legacy_meb_build_generator_v1"
    assert len(registry.candidate_audits) == 64
    assert all(row.semantic_validator_status == "PASS" for row in registry.candidate_audits)
    assert all(row.leakage_audit_status == "PASS" for row in registry.candidate_audits)
    assert calibration.historical_rhs_completion_pilot.model_dump(mode="json") == {
        "path": "data/tasks/math_equation_balancer_pilot.jsonl",
        "sha256": "6fa5a5d3be52853f8d9da93a9a9c0ea5399f67c9c08acc64fdbdd4821f68bb41",
        "status": "HISTORICAL_EVIDENCE_ONLY",
    }
    assert all(row.build_registry_sha256 for row in worked)
    assert all(row.generator_implementation_sha256 for row in worked)
