from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from memcontam.manifests.phase13 import (
    ConformanceAuthority,
    SourceTrajectoryManifest,
    load_conformance_authority,
)


SHA = "a" * 64


def _authority_payload() -> dict[str, object]:
    return {
        "schema_version": "phase13_prefix_conformance_authority_v1",
        "authority_id": "phase13-prefix-authority-v1",
        "conformance_id": "phase13-ten-condition-prefix-v1",
        "checker_version": "phase13-prefix-checker-v1",
        "checker_script_sha256": SHA,
        "checker_config_sha256": "b" * 64,
        "repository_commit": "e2f980e4b4e4a79479a08d2844f32eaeb4aa05e3",
        "source_run_id": "run-h10",
        "source_manifest_id": "manifest-h10",
        "source_manifest_sha256": "c" * 64,
        "source_checkpoint_id": "checkpoint-1",
        "source_checkpoint_sha256": "d" * 64,
        "source_suffix_id": "suffix-1",
        "source_ordered_stream_sha256": "e" * 64,
        "task": "game24",
        "model_snapshot_id": "gpt-4o-2024-11-20",
        "decoding_contract_id": "decoding-v1",
        "prompt_contract_id": "prompt-v1",
        "tool_contract_id": "tool-v1",
        "parser_contract_id": "parser-v1",
        "verifier_contract_id": "verifier-v1",
        "native_semantics_id": "native-v1",
        "session_contract_id": "session-v1",
        "randomness_contract_id": "randomness-v1",
        "intervention_id": "intervention-v1",
        "future_feedback_cutoff": 0,
        "source_execution_contract_id": "phase13-main-a-h10-execution-v1",
        "source_execution_owner_id": "phase13-h10-execution-owner-v1",
        "analysis_windows": [
            {
                "analysis_window_id": "accuracy-h2-sensitivity",
                "window_length": 2,
                "event_time_start": 0,
                "event_time_end": 1,
                "evidence_status": "prespecified_sensitivity",
                "multiplicity_status": "descriptive_no_inferential_family",
            },
            {
                "analysis_window_id": "recurrence-h2-descriptive",
                "window_length": 2,
                "event_time_start": 0,
                "event_time_end": 1,
                "evidence_status": "descriptive",
                "multiplicity_status": "estimation_only",
            },
            {
                "analysis_window_id": "accuracy-h5-primary",
                "window_length": 5,
                "event_time_start": 0,
                "event_time_end": 4,
                "evidence_status": "confirmatory_primary",
                "multiplicity_status": "primary_holm_family",
            },
        ],
    }


def test_authority_load_requires_external_exact_hash_and_no_follow(tmp_path: Path) -> None:
    payload = _authority_payload()
    path = tmp_path / "authority.json"
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)

    authority = load_conformance_authority(path, hashlib.sha256(raw).hexdigest())

    assert authority.analysis_windows[0].event_time_end == 1
    assert authority.analysis_windows[2].event_time_end == 4
    with pytest.raises(ValueError, match="AUTHORITY_HASH_MISMATCH"):
        load_conformance_authority(path, "0" * 64)
    link = tmp_path / "authority-link.json"
    os.symlink(path, link)
    with pytest.raises(ValueError, match="AUTHORITY_READ_INVALID"):
        load_conformance_authority(link, hashlib.sha256(raw).hexdigest())


def test_phase13_manifests_are_strict_and_require_exact_h10_source() -> None:
    payload = _authority_payload()
    payload["unknown"] = True

    with pytest.raises(ValidationError):
        ConformanceAuthority.model_validate(payload)

    with pytest.raises(ValidationError):
        SourceTrajectoryManifest.model_validate(
            {
                "schema_version": "phase13_source_trajectory_v1",
                "source_run_id": "run-h10",
                "source_manifest_id": "manifest-h10",
                "source_checkpoint_id": "checkpoint-1",
                "source_checkpoint_sha256": "d" * 64,
                "source_suffix_id": "suffix-1",
                "source_ordered_stream_sha256": "e" * 64,
                "task": "game24",
                "model_snapshot_id": "gpt-4o-2024-11-20",
                "decoding_contract_id": "decoding-v1",
                "prompt_contract_id": "prompt-v1",
                "tool_contract_id": "tool-v1",
                "parser_contract_id": "parser-v1",
                "verifier_contract_id": "verifier-v1",
                "native_semantics_id": "native-v1",
                "session_contract_id": "session-v1",
                "randomness_contract_id": "randomness-v1",
                "intervention_id": "intervention-v1",
                "future_feedback_cutoff": 0,
                "source_execution_contract_id": "phase13-main-a-h10-execution-v1",
                "source_execution_owner_id": "phase13-h10-execution-owner-v1",
                "source_raw_path": "raw.jsonl",
                "source_raw_sha256": SHA,
                "event_count": 10,
                "unknown": True,
            }
        )
