from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "phase12" / "registries" / "candidate_registry_v1.json"


def test_certifies_first_difference_counterexample() -> None:
    from memcontam.contamination.phase12.certification import CertificationSuite, certify_triplet
    from memcontam.contamination.phase12.registry import load_candidate_registry

    triplet = next(
        item
        for item in load_candidate_registry(REGISTRY_PATH).triplets
        if item.task == "word_sorting"
    )
    result = certify_triplet(triplet, CertificationSuite.primary())

    assert result.passed
    assert result.counterexample == ("ayz", "aza")
    assert result.false_rule_result is False
    assert result.correct_rule_result is True
