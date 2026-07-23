from __future__ import annotations

import json
from pathlib import Path

from memcontam.behavior.directional import evaluate_direction
from memcontam.behavior.invariance import SeedInterval, evaluate_invariance
from memcontam.behavior.registry import load_behavior_registry_bundle


ROOT = Path(__file__).parents[1]
REGISTRIES = ROOT / "data" / "phase12" / "registries"
FIXTURES = ROOT / "tests" / "fixtures" / "phase12"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_classifies_all_registered_inv_and_dir_families() -> None:
    bundle = load_behavior_registry_bundle(REGISTRIES)
    behavior_fixture = _fixture("FX-BEHAVIOR-001.json")
    rag_fixture = _fixture("FX-RAG-001.json")
    aggregate_fixture = _fixture("FX-AGG-001.json")
    route_fixture = _fixture("FX-ROUTE-001.json")

    rows = {row.test_id: row for row in bundle.behavior_tests.rows}
    assert set(behavior_fixture["expected"]["required_test_ids"]) == {
        "MFT-01-EASY-NOMEM",
        "MFT-02-INVALID-MEMORY",
        "MFT-03-CORRECT-MEMORY",
        "MFT-04-FILTER-CONTRACT",
        "INV-01-IRRELEVANT",
        "INV-02-PARAPHRASE",
        "INV-03-METADATA",
        "DIR-01-CONTAM-EXPOSURE",
        "DIR-02-FILTER-EXPOSURE",
        "DIR-03-CORRECT-UTILITY",
        "DIR-04-FH-CAPACITY",
    }
    assert route_fixture["valid_inv03_equivalence_registry"]["artifact_hash"] == (
        "ffdb247dc187d462208dbe9f7a4ead8bfa27def24a3052baedca77c50aa2e620"
    )

    inv01 = evaluate_invariance(
        {"interval": SeedInterval(-0.05, -0.02, 0.04, resampling_unit="seed")},
        rows["INV-01"],
    )
    inv02 = evaluate_invariance(
        {"interval": SeedInterval(0.0, -0.1, 0.1, resampling_unit="seed")},
        rows["INV-02"],
    )
    inv03 = evaluate_invariance(
        {
            "interval": SeedInterval(0.01, -0.04, 0.05, resampling_unit="seed"),
            "mechanical_gate_status": "pass",
            "canonical_content_id": rag_fixture["expected_branch_index_sha256"]["clean"],
            "id_correspondence": {
                "reference": rag_fixture["expected_branch_index_sha256"]["clean"],
                "variant": rag_fixture["expected_branch_index_sha256"]["clean"],
            },
        },
        rows["INV-03"],
    )
    dir01 = evaluate_direction(
        SeedInterval(0.2, 0.1, 0.3, resampling_unit="seed"),
        rows["DIR-01"],
    )
    dir02 = evaluate_direction(
        SeedInterval(-0.1, -0.2, 0.0, resampling_unit="seed"),
        rows["DIR-02"],
    )
    dir03 = evaluate_direction(
        SeedInterval(0.1, 0.0, 0.2, resampling_unit="seed"),
        rows["DIR-03"],
    )
    dir04 = evaluate_direction(
        SeedInterval(0.1, 0.0, 0.2, resampling_unit="seed"),
        rows["DIR-04"],
    )

    results = (inv01, inv02, inv03, dir01, dir02, dir03, dir04)
    assert {result.test_id: result.classification for result in results} == {
        "INV-01": "supported",
        "INV-02": "supported",
        "INV-03": "supported",
        "DIR-01": "supported",
        "DIR-02": "supported",
        "DIR-03": "supported",
        "DIR-04": "supported",
    }
    assert all(result.evidence_hash for result in results)
    assert aggregate_fixture["expected"]["clean_minus_contam"] == 0.5
