from __future__ import annotations

import json
from pathlib import Path


def test_reduced_main_readiness_stops_on_one_pre_main_decision() -> None:
    artifact = json.loads(
        Path("data/phase13/main/reduced_main_a_readiness_v1.json").read_text(encoding="utf-8")
    )

    assert artifact["status"] == "BLOCKED_ON_ONE_PRE_MAIN_DECISION"
    assert artifact["provider_calls_issued"] == 0
    assert artifact["filter_calls"] == 0
    assert artifact["panel"]["arms"] == ["clean", "correct", "irrelevant", "contam"]
    assert artifact["panel"]["target_jointly_eligible_seeds_per_task"] == 10
    assert artifact["blocking_rationale"] == {
        "existing_route_is_not_transferable": (
            "The only 3w route binds five arms including Filter and has null route and seed manifests."
        ),
        "attempted_seed_counts_are_undetermined": (
            "Ten jointly eligible seeds require a pre-Main reduced-panel intersection rate across "
            "FH, RAG-Frozen, BoT, and Reflexion; no admissible rate exists."
        ),
        "cost_totals_are_undetermined": (
            "Reduced semantic calls depend on attempted counts and conditional activations; token/USD "
            "totals additionally require frozen stage token estimates."
        ),
        "prohibited_inference": "Main outcomes, synthetic fixtures, and Filter-only reservations cannot supply these inputs.",
    }
    assert artifact["blocking_decision"] == {
        "code": "REDUCED_PANEL_PRE_MAIN_RATE_SOURCE_REQUIRED",
        "required_output": [
            "conditional_call_rate_registry",
            "attempted_seed_counts_by_task",
            "route_selection_manifest",
            "seed_allocation_manifest",
        ],
        "admissible_options": [
            "run_reduced_panel_pilot_b_before_main",
            "register_a_separately_justified_conservative_upper_bound_before_main",
        ],
    }
