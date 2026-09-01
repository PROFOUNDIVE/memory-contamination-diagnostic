from __future__ import annotations

import json
import hashlib
from pathlib import Path


def test_reduced_main_readiness_preserves_calibration_without_using_it_as_main_gate() -> None:
    artifact = json.loads(
        Path("data/phase13/main/reduced_main_a_readiness_v1.json").read_text(encoding="utf-8")
    )

    assert artifact["status"] == "SUPERSEDED_BY_POST_CUTOFF_MAIN_PACKAGE_SELECTION"
    assert artifact["provider_calls_issued"] == 0
    assert artifact["filter_calls"] == 0
    assert artifact["current_main_seed_policy"] == {
        "attempted_seed_count_per_task": 10,
        "replacement": "prohibited",
        "adaptive_augmentation": "prohibited",
        "support_role": "realized_analysis_domain_and_estimability",
    }
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
        "generic_pre_main_calibration_role": "historical_descriptive_build_evidence_only",
        "reduced_scope_frozen_before_main_unblinding": True,
    }
    assert artifact["remaining_track2_prerequisites"] == [
        "legacy_rag_build_cal_eval_materialization",
        "legacy_rag_three_task_local_24_document_corpora",
        "legacy_rag_final_bge_m3_indices",
    ]


def test_track1_checkpoint_records_observed_read_only_router_completion() -> None:
    path = Path("data/phase13/main/track1_authority_state_sync_checkpoint_v1.json")
    assert path.exists()
    checkpoint = json.loads(path.read_text(encoding="utf-8"))

    assert checkpoint["repository_state_sync"] == "COMPLETE"
    assert checkpoint["track1_status"] == "TRACK1_AUTHORITY_AND_STATE_SYNC_COMPLETE"
    assert checkpoint["authority_router"]["current_sha256"] == (
        "76af55a04ce47f5bfe82795a99b0654b509becc7e202b3e05db29550297cf430"
    )
    assert checkpoint["active_authority_hashes"]["post_cutoff_addendum"] == (
        "6ab2db4041333a26e9c76a034266c2d7de64491c5c109634c97ecebef9b2f59b"
    )
    assert checkpoint["active_authority_hashes"]["experiment_design_revised_v10"] == (
        "174002b407c2c4055c42298bd16ca13463e80af5c935f9593f556c3f71d7671c"
    )
    assert "experiment_design_revised_v9" not in checkpoint["active_authority_hashes"]
    assert checkpoint["authority_router"]["mount_options"] == [
        "ro",
        "nosuid",
        "nodev",
        "relatime",
    ]
    assert checkpoint["track2"]["legacy_rag_materialization"] == "PENDING"
    assert checkpoint["main_execution_authorized"] is False
    unsigned = dict(checkpoint)
    checkpoint_hash = unsigned.pop("checkpoint_hash")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    assert checkpoint_hash == hashlib.sha256(canonical).hexdigest()
