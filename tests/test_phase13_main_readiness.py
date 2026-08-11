from __future__ import annotations

import json
from pathlib import Path


def test_reduced_main_readiness_requires_only_generic_clean_prefix_calibration() -> None:
    artifact = json.loads(
        Path("data/phase13/main/reduced_main_a_readiness_v1.json").read_text(encoding="utf-8")
    )

    assert artifact["status"] == "READY_FOR_SEPARATE_PRE_MAIN_CALIBRATION_AUTHORIZATION"
    assert artifact["provider_calls_issued"] == 0
    assert artifact["filter_calls"] == 0
    assert artifact["eligibility_definition"] == {
        "baseline_families": ["fh_bounded", "rag_frozen", "bot_style", "reflexion_style"],
        "horizon": 1,
        "filter_arm_affects_population": False,
        "conditional_call_rates_are_separate": True,
    }
    assert artifact["empirical_joint_eligibility"] == {
        "game24": "missing",
        "math_equation_balancer": "missing",
        "word_sorting": "missing",
    }
    assert artifact["calibration_packet"] == {
        "config_path": "configs/phase13/clean_prefix_calibration_v1.yaml",
        "config_sha256": "3f1ab4ec633e812b922efc3a3d7a2d7f551dbaf65be67b95f097d109d01a41f9",
        "trajectory_seeds_per_task": 4,
        "trajectory_count": 12,
        "prefix_position_count": 44,
        "nominal_semantic_calls": 264,
        "maximum_semantic_calls": 396,
        "maximum_transport_attempts": 1584,
        "maximum_input_tokens": 6488064,
        "maximum_output_tokens": 3244032,
        "hard_ceiling_microusd": 48660480,
    }
    assert artifact["prospective_scope_restriction"] == {
        "filter_challenge_available": False,
        "filter_pilot_b_selection_executed": False,
        "generic_pre_main_calibration_role": "non_filter_joint_eligibility_and_route_budget",
        "reduced_scope_frozen_before_main_unblinding": True,
    }
