from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CURRENT_REGISTRY = ROOT / "data/phase12/registries/candidate_registry_v2.json"
HISTORICAL_REGISTRY = ROOT / "data/phase12/registries/candidate_registry_v1.json"


def test_game24_v2_requires_full_solution_and_no_flat_solution() -> None:
    from memcontam.contamination.phase13_v2_applicability import (
        G24_APPLICABILITY_ID,
        game24_false_rule_applicable,
    )

    assert G24_APPLICABILITY_ID == "G24_CANONICAL_FALSE_RULE_APPLICABILITY_V2"
    assert game24_false_rule_applicable((1, 1, 1, 8))
    assert not game24_false_rule_applicable((1, 2, 3, 4))
    assert not game24_false_rule_applicable((1, 1, 1, 1))


def test_word_sorting_v2_accepts_only_strict_pre_final_reversal() -> None:
    from memcontam.contamination.phase13_v2_applicability import (
        WS_APPLICABILITY_ID,
        word_sorting_false_rule_applicable,
    )

    assert WS_APPLICABILITY_ID == "WS_CANONICAL_FALSE_RULE_APPLICABILITY_V2"
    assert word_sorting_false_rule_applicable(("ayz", "aza"))
    assert not word_sorting_false_rule_applicable(("az", "ba"))
    assert not word_sorting_false_rule_applicable(("ab", "ac"))
    assert not word_sorting_false_rule_applicable(("aya", "aza"))
    assert not word_sorting_false_rule_applicable(("ab", "abc"))


def test_v2_applicability_specs_are_hash_bound() -> None:
    from memcontam.contamination.phase13_v2_applicability import (
        G24_APPLICABILITY_SHA256,
        WS_APPLICABILITY_SHA256,
    )

    assert len(G24_APPLICABILITY_SHA256) == 64
    assert len(WS_APPLICABILITY_SHA256) == 64
    assert G24_APPLICABILITY_SHA256 != WS_APPLICABILITY_SHA256


def test_current_registry_uses_only_governed_v2_task_laws() -> None:
    from memcontam.contamination.phase12.registry import load_current_candidate_registry

    registry = load_current_candidate_registry(CURRENT_REGISTRY)
    game24, _, word_sorting = registry.triplets

    assert registry.registry_id == "phase13-candidate-registry-v2"
    assert game24.triplet_id == "game24-parentheses-restriction-v2"
    assert game24.false_candidate.candidate_id == "candidate-game24-flat-precedence-v2"
    assert game24.correct_twin.candidate_id == "control-game24-parenthesization-v2"
    assert (
        game24.irrelevant_control.candidate_id
        == "control-game24-flat-solvable-subfamily-v2"
    )
    assert game24.applicability_id == "G24_CANONICAL_FALSE_RULE_APPLICABILITY_V2"
    assert word_sorting.applicability_id == "WS_CANONICAL_FALSE_RULE_APPLICABILITY_V2"


def test_current_registry_rejects_historical_fraction_rule() -> None:
    from memcontam.contamination.phase12.models import CandidateCertificationError
    from memcontam.contamination.phase12.registry import load_current_candidate_registry

    with pytest.raises(CandidateCertificationError, match="STALE_CANDIDATE_REGISTRY"):
        load_current_candidate_registry(HISTORICAL_REGISTRY)


def test_current_registry_certifies_v2_witnesses() -> None:
    from memcontam.contamination.phase12.certification import (
        CertificationSuite,
        certify_triplet,
    )
    from memcontam.contamination.phase12.registry import load_current_candidate_registry

    registry = load_current_candidate_registry(CURRENT_REGISTRY)
    results = tuple(
        certify_triplet(triplet, CertificationSuite.primary())
        for triplet in registry.triplets
    )

    assert all(result.passed for result in results)
    assert results[0].counterexample == "1,1,1,8"
    assert results[0].false_rule_result is False
    assert results[0].correct_rule_result is True
