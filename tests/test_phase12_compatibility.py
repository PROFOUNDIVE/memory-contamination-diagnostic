from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from memcontam.logging.schema_v3 import parse_log_record_v3


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-SCHEMA-001.json"
compatibility = importlib.import_module("memcontam.evaluation.compatibility")
CompatibilityError = compatibility.CompatibilityError
build_compatibility_key = compatibility.build_compatibility_key
validate_compatible_runs = compatibility.validate_compatible_runs
_METADATA_CASES = {
    "pre_route": 0,
    "selected_route": 2,
    "non_scientific_exploratory": 3,
    "scientific_exploratory": 5,
}


def _metadata(case: str) -> Any:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return parse_log_record_v3(copy.deepcopy(fixture["valid_run_metadata"][_METADATA_CASES[case]]))


def _changed(run: Any, **changes: Any) -> Any:
    return run.model_copy(update=changes)


def test_pools_identical_phase12_branch_runs() -> None:
    branch = _metadata("selected_route")
    paired_branch = _changed(
        branch,
        trajectory_seed=branch.trajectory_seed + 1,
        abstract_seed_slot_or_none=f"{branch.abstract_seed_slot_or_none}-next",
    )

    key = validate_compatible_runs([branch, paired_branch])

    assert key == build_compatibility_key(branch)
    assert {build_compatibility_key(run) for run in [branch, paired_branch]} == {key}


@pytest.mark.parametrize(
    ("field", "value", "error_field"),
    [
        ("schema_version", "logging_v1", "schema_version"),
        ("schema_version", "logging_v2", "schema_version"),
        ("contract_level", "phase11", "contract_level"),
        ("protocol_version", "phase12_code_exploratory_v1", "protocol_version"),
        ("evidence_layer", "calibration", "evidence_layer"),
        ("execution_key", {"kind": "memory_arm", "arm": "filter"}, "execution_key"),
        ("scientific_admission_ref", {"certificate": "other"}, "scientific_admission"),
        ("route_selection_manifest_id", "route-selection-other", "route_selection_manifest_id"),
        ("seed_allocation_manifest_id", "seed-allocation-other", "seed_allocation_manifest_id"),
        ("tool_contract_hash", "sha256:python-sandbox", "tool_contract_hash"),
        ("embedding_contract_hash", "sha256:other-embedding", "embedding_contract_hash"),
        ("candidate_registry_version", "candidates-v2", "candidate_registry_version"),
        ("split_manifest_version", "split-v2", "split_manifest_version"),
        ("metric_registry_version", "metrics-v2", "metric_registry_version"),
        ("baseline_condition_id", "full-history-bounded", "baseline_condition_id"),
        (
            "sensitivity_cell_ref",
            {"kind": "horizon", "cell_id": "horizon-32", "base_cell_id": "base", "horizon": 32},
            "sensitivity_cell_ref",
        ),
        ("prefix_template_key_or_none", "checkpoint-other", "prefix_template_key_or_none"),
    ],
    ids=(
        "v1-v3",
        "v2-v3",
        "contract-level",
        "protocol-version",
        "evidence-layer",
        "execution-key",
        "scientific-admission",
        "route-selection",
        "seed-allocation",
        "tool-mode",
        "embedding",
        "candidate",
        "split",
        "metric",
        "fidelity-rag-fh",
        "timing-horizon",
        "checkpoint",
    ),
)
def test_rejects_each_registered_incompatible_mixture(
    field: str, value: Any, error_field: str
) -> None:
    branch = _metadata("selected_route")
    incompatible = _changed(branch, **{field: value})

    with pytest.raises(CompatibilityError, match=error_field):
        validate_compatible_runs([branch, incompatible])


def test_enforces_governance_and_seed_applicability() -> None:
    pre_route = _metadata("pre_route")
    with pytest.raises(
        CompatibilityError, match="GOVERNANCE_FORBIDDEN.*route_selection_manifest_id"
    ):
        build_compatibility_key(
            _changed(pre_route, route_selection_manifest_id="route-selection-forbidden")
        )

    selected_route = _metadata("selected_route")
    with pytest.raises(
        CompatibilityError, match="SEED_ASSIGNMENT_MISMATCH.*abstract_seed_slot_or_none"
    ):
        validate_compatible_runs(
            [
                selected_route,
                _changed(selected_route, trajectory_seed=selected_route.trajectory_seed + 1),
            ]
        )

    non_scientific = _metadata("non_scientific_exploratory")
    with pytest.raises(CompatibilityError, match="EXPLORATORY_ACTIVATION_FORBIDDEN"):
        build_compatibility_key(
            _changed(non_scientific, exploratory_activation_manifest_id="activation-forbidden")
        )

    scientific = _metadata("scientific_exploratory")
    with pytest.raises(CompatibilityError, match="exploratory_activation_manifest_id"):
        validate_compatible_runs(
            [
                scientific,
                _changed(scientific, exploratory_activation_manifest_id="activation-other"),
            ]
        )
