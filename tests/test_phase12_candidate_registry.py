from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "phase12" / "FX-CANDIDATE-001.json"
REGISTRY_PATH = ROOT / "data" / "phase12" / "registries" / "candidate_registry_v1.json"
AUDIT_PATH = ROOT / "data" / "phase12" / "registries" / "hidden_audit_registry_v1.json"


def _registry_module():
    assert importlib.util.find_spec("memcontam.contamination.phase12.registry") is not None
    return importlib.import_module("memcontam.contamination.phase12.registry")


def _certification_module():
    assert importlib.util.find_spec("memcontam.contamination.phase12.certification") is not None
    return importlib.import_module("memcontam.contamination.phase12.certification")


def test_freezes_one_certified_triplet_per_primary_task() -> None:
    registry_module = _registry_module()
    certification = _certification_module()
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    registry = registry_module.load_candidate_registry(REGISTRY_PATH)
    reloaded = registry_module.load_candidate_registry(REGISTRY_PATH)
    frozen = registry_module.freeze_registry(registry, certification.CertificationSuite.primary())
    audit = registry_module.load_hidden_audit_registry(AUDIT_PATH)

    assert registry == reloaded
    assert registry.artifact_hash == reloaded.artifact_hash
    assert registry.registry_id == "phase12-candidate-registry-v1"
    assert {triplet.task for triplet in registry.triplets} == {
        "game24",
        "math_equation_balancer",
        "word_sorting",
    }
    assert len(registry.triplets) == 3
    assert all(result.passed for result in frozen.certification_results)
    assert all(triplet.false_candidate.in_b_star for triplet in registry.triplets)
    assert all(not triplet.correct_twin.in_b_star for triplet in registry.triplets)
    assert all(not triplet.irrelevant_control.in_b_star for triplet in registry.triplets)
    assert all(triplet.frozen_at == registry.frozen_at for triplet in registry.triplets)
    assert {origin.candidate_id for origin in audit.origins} == {
        triplet.false_candidate.candidate_id for triplet in registry.triplets
    }
    assert all(origin.independent_of_outcomes for origin in audit.origins)
    assert all(not hasattr(triplet, "audit_origin") for triplet in registry.triplets)
    assert fixture["expected"] == {"controls_in_b_star": False, "root_count": 1}


def test_rejects_uncertifiable_or_outcome_selected_candidates() -> None:
    registry_module = _registry_module()
    certification = _certification_module()
    models = importlib.import_module("memcontam.contamination.phase12.models")
    registry = registry_module.load_candidate_registry(REGISTRY_PATH)
    triplet = registry.triplets[0]

    invalid_syntax = replace(
        triplet,
        false_candidate=replace(triplet.false_candidate, code_variant="def broken(:\n    pass"),
    )
    empty_applicability = replace(
        triplet,
        false_candidate=replace(triplet.false_candidate, applicability=()),
    )
    main_example_leak = replace(
        triplet,
        false_candidate=replace(triplet.false_candidate, main_example_ids=("main-game24-001",)),
    )
    mismatched_signature = replace(
        triplet,
        false_candidate=replace(
            triplet.false_candidate,
            code_variant=triplet.false_candidate.code_variant.replace(
                "is_integer_intermediate", "wrong_signature"
            ),
        ),
    )
    outcome_selected = replace(
        triplet,
        false_candidate=replace(triplet.false_candidate, outcome_selected=True),
    )

    for invalid in (
        invalid_syntax,
        empty_applicability,
        main_example_leak,
        mismatched_signature,
        outcome_selected,
    ):
        with pytest.raises(models.CandidateCertificationError):
            certification.certify_triplet(invalid, certification.CertificationSuite.primary())
