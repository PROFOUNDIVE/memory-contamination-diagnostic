from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import json
from collections import Counter

from memcontam.config.phase12 import (
    build_candidate_template_set,
    load_phase12_config,
    resolve_phase12_config,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-CONFIG-001.json"
ROUTE_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-ROUTE-001.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _route_fixture() -> dict[str, Any]:
    return json.loads(ROUTE_FIXTURE_PATH.read_text(encoding="utf-8"))


def test_config_generates_complete_candidate_template_sets() -> None:
    resolved = resolve_phase12_config(load_phase12_config(FIXTURE_PATH))

    candidates: tuple[Literal["3w", "5w"], ...] = ("3w", "5w")
    layers: tuple[Literal["core", "sensitivity", "replication", "extension"], ...] = (
        "core",
        "sensitivity",
        "replication",
        "extension",
    )
    template_sets = [build_candidate_template_set(resolved, candidate) for candidate in candidates]
    expected = _fixture()["template_package"]["expected_template_counts_by_route"]
    expected_hashes = {
        candidate["candidate_route"]: candidate["expected_candidate_template_set_hash"]
        for candidate in _fixture()["route_candidates"]
    }
    frozen_sets = _route_fixture()["valid_candidate_template_sets"]

    assert {template_set.candidate_route for template_set in template_sets} == {"3w", "5w"}
    assert all(
        template_set.artifact_hash == expected_hashes[template_set.candidate_route]
        for template_set in template_sets
    )
    assert all(
        template_set.model_dump(mode="json") == frozen_sets[template_set.candidate_route]
        for template_set in template_sets
    )
    assert all(
        len(template_set.run_templates) == expected[template_set.candidate_route]["total"]
        for template_set in template_sets
    )
    assert all(
        len(template_set.prefix_templates) == expected[template_set.candidate_route]["prefix"]
        for template_set in template_sets
    )
    assert all(
        all(
            Counter(template.layer for template in template_set.run_templates).get(layer, 0)
            == expected[template_set.candidate_route][layer]
            for layer in layers
        )
        for template_set in template_sets
    )
    assert all(template_set.candidate_route in {"3w", "5w"} for template_set in template_sets)
    assert all(
        template.evidence_layer in {"build", "calibration", "main", "extension"}
        for template_set in template_sets
        for template in template_set.run_templates
    )
    assert all(
        template.evidence_layer in {"build", "calibration", "main", "extension"}
        for template_set in template_sets
        for template in template_set.prefix_templates
    )
    assert all(
        template.execution_key.kind in {"memory_arm", "nomem_singleton"}
        for template_set in template_sets
        for template in template_set.run_templates
    )
    assert all(
        "trajectory_seed" not in template.model_dump()
        for template_set in template_sets
        for template in template_set.run_templates
    )
    assert all(
        template.model_snapshot == "gpt-4o-v1"
        for template_set in template_sets
        for template in template_set.run_templates
        if template.run_family in {"main_a", "main_b", "extension"}
    )
    assert all(
        template.model_snapshot == "frontier-model-v1"
        and template.analysis_status == "exploratory_model_specificity"
        for template_set in template_sets
        for template in template_set.run_templates
        if template.run_family == "main_c"
    )
    assert all(
        template.prefix_template_key_or_none is None
        and template.execution_key.kind == "nomem_singleton"
        for template_set in template_sets
        for template in template_set.run_templates
        if template.baseline_condition_id == "nomem"
    )
    rag_branch_ids = {
        template.corpus_index_filter_versions["index_hash"]
        for template in template_sets[0].run_templates
        if template.baseline_condition_id == "rag_frozen"
        and template.execution_key.kind == "memory_arm"
    }
    assert len(rag_branch_ids) > 1
    assert not hasattr(template_sets[0], "route_selection_manifest")
