from __future__ import annotations

import hashlib
import json

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_execution_contract import CORE_MAIN_REGISTRY
from memcontam.readiness.phase13_main_readiness_models import Phase13MainReadinessError


def validate_track1(raw: bytes) -> None:
    payload = _json_object(raw, "MR_P4_TRACK1_CONTRACT_MISMATCH")
    checkpoint_hash = payload.pop("checkpoint_hash", None)
    router = _object_field(payload, "authority_router", "MR_P4_TRACK1_CONTRACT_MISMATCH")
    authority_hashes = dict(CORE_MAIN_REGISTRY.authority_stack)
    expected_authorities = {
        "theory_revised_v1": authority_hashes["theory"],
        "baseline_revised_v5": authority_hashes["baseline"],
        "protocol_revised_v8": authority_hashes["protocol"],
        "post_cutoff_addendum": authority_hashes["post_cutoff_addendum"],
        "experiment_design_revised_v10": authority_hashes["experiment_design"],
    }
    completed = _object_field(
        payload, "completed_repository_sync", "MR_P4_TRACK1_CONTRACT_MISMATCH"
    )
    if (
        checkpoint_hash != _canonical_hash(payload)
        or set(payload)
        != {
            "schema_version",
            "repository_branch",
            "repository_head_at_start",
            "repository_state_sync",
            "track1_status",
            "authority_router",
            "active_authority_hashes",
            "completed_repository_sync",
            "preserved",
            "track2",
            "main_execution_authorized",
        }
        or payload.get("schema_version")
        != "phase13_track1_authority_state_sync_checkpoint_v1"
        or payload.get("repository_state_sync") != "COMPLETE"
        or payload.get("track1_status") != "TRACK1_AUTHORITY_AND_STATE_SYNC_COMPLETE"
        or set(router)
        != {
            "path",
            "current_sha256",
            "mount_target",
            "mount_options",
            "required_sync_clauses",
        }
        or router.get("current_sha256") != CORE_MAIN_REGISTRY.authority_router_sha256
        or payload.get("active_authority_hashes") != expected_authorities
        or completed.get("attempted_seed_count_per_task")
        != CORE_MAIN_REGISTRY.attempted_seed_count
        or completed.get("seed_replacement") != "prohibited"
        or completed.get("adaptive_seed_augmentation") != "prohibited"
        or completed.get("rag_cutoff_status") != CORE_MAIN_REGISTRY.rag_cutoff_status
        or completed.get("new_mcq_rag_current_main_exclusions")
        != [list(cell) for cell in CORE_MAIN_REGISTRY.current_main_excluded_cells]
        or completed.get("authority_freeze_accepts_post_cutoff_addendum_role") is not True
        or payload.get("main_execution_authorized") is not False
    ):
        raise Phase13MainReadinessError("MR_P4_TRACK1_CONTRACT_MISMATCH")


def validate_package_selection(raw: bytes) -> None:
    payload = _json_object(raw, "MR_P4_PACKAGE_SELECTION_MISMATCH")
    package_hash = payload.pop("package_hash", None)
    authority = _object_field(payload, "authority", "MR_P4_PACKAGE_SELECTION_MISMATCH")
    cutoff = _object_field(payload, "cutoff", "MR_P4_PACKAGE_SELECTION_MISMATCH")
    resolution = _object_field(payload, "resolution", "MR_P4_PACKAGE_SELECTION_MISMATCH")
    selected = _object_field(payload, "selected_current_main", "MR_P4_PACKAGE_SELECTION_MISMATCH")
    track2 = _object_field(payload, "track2", "MR_P4_PACKAGE_SELECTION_MISMATCH")
    authority_hashes = dict(CORE_MAIN_REGISTRY.authority_stack)
    if (
        package_hash != _canonical_hash(payload)
        or set(payload)
        != {
            "schema_version",
            "authority",
            "cutoff",
            "pre_cutoff_package",
            "resolution",
            "selected_current_main",
            "track2",
        }
        or payload.get("schema_version") != "phase13_post_cutoff_package_selection_v1"
        or set(authority) != {"addendum_path", "addendum_sha256"}
        or authority.get("addendum_path") != CORE_MAIN_REGISTRY.post_cutoff_addendum_path
        or authority.get("addendum_sha256") != authority_hashes["post_cutoff_addendum"]
        or set(cutoff) != {"timestamp", "rule_id", "status"}
        or cutoff.get("timestamp") != CORE_MAIN_REGISTRY.rag_deadline
        or cutoff.get("rule_id") != CORE_MAIN_REGISTRY.post_cutoff_rule_id
        or cutoff.get("status") != CORE_MAIN_REGISTRY.rag_cutoff_status
        or set(resolution)
        != {
            "excluded_current_main_cells",
            "prospective_extension_id",
            "late_reentry_to_current_main",
        }
        or resolution.get("excluded_current_main_cells")
        != [list(cell) for cell in CORE_MAIN_REGISTRY.current_main_excluded_cells]
        or resolution.get("prospective_extension_id")
        != CORE_MAIN_REGISTRY.prospective_rag_extension_id
        or resolution.get("late_reentry_to_current_main") != "prohibited"
        or set(selected)
        != {
            "package_id",
            "tasks",
            "memory_baselines",
            "arms",
            "nomem_policy",
            "attempted_seed_count_per_task",
            "seed_replacement",
            "adaptive_seed_augmentation",
            "H_run",
            "H_primary",
            "primary_analysis_window_id",
        }
        or selected.get("package_id") != CORE_MAIN_REGISTRY.current_main_package_id
        or selected.get("tasks") != list(CORE_MAIN_REGISTRY.tasks)
        or selected.get("memory_baselines") != list(CORE_MAIN_REGISTRY.memory_baselines)
        or selected.get("arms") != list(CORE_MAIN_REGISTRY.arms)
        or selected.get("nomem_policy") != CORE_MAIN_REGISTRY.nomem_policy
        or selected.get("attempted_seed_count_per_task")
        != CORE_MAIN_REGISTRY.attempted_seed_count
        or selected.get("seed_replacement") != "prohibited"
        or selected.get("adaptive_seed_augmentation") != "prohibited"
        or selected.get("H_run") != CORE_MAIN_REGISTRY.H_run
        or selected.get("H_primary") != CORE_MAIN_REGISTRY.H_primary
        or selected.get("primary_analysis_window_id")
        != CORE_MAIN_REGISTRY.primary_analysis_window_id
        or track2.get("legacy_rag_materialization")
        != "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
        or track2.get("main_execution_authorized") is not False
    ):
        raise Phase13MainReadinessError("MR_P4_PACKAGE_SELECTION_MISMATCH")


def _json_object(raw: bytes, error_code: str) -> dict[str, JsonValue]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase13MainReadinessError(error_code) from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise Phase13MainReadinessError(error_code)
    return value


def _object_field(
    payload: dict[str, JsonValue], field: str, error_code: str
) -> dict[str, JsonValue]:
    value = payload.get(field)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise Phase13MainReadinessError(error_code)
    return value


def _canonical_hash(value: dict[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = ["validate_package_selection", "validate_track1"]
