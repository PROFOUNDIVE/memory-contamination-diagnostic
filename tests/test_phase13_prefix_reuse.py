from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from memcontam.manifests.phase13 import NotExchangeable, PrefixDerivationArtifact
from memcontam.readiness.phase13_prefix_reuse import derive_prefix_windows


CHECK_IDS = (
    "checkpoint_source_identity",
    "suffix_order",
    "execution_contract_identity",
    "native_semantics",
    "session_randomness",
    "intervention_identity",
    "future_feedback_cutoff",
    "source_manifest_identity",
    "exact_event_range",
    "source_raw_bytes",
)


def _fixture(tmp_path: Path) -> tuple[Path, str, Path, str, dict[str, object], dict[str, object]]:
    events = [
        {
            "event_index": index,
            "status": "succeeded" if index % 2 == 0 else "failed",
            "source_checkpoint_id": "checkpoint-1",
            "source_suffix_id": "suffix-1",
            "task": "game24",
            "model_snapshot_id": "gpt-4o-2024-11-20",
            "session_contract_id": "session-v1",
            "intervention_id": "intervention-v1",
            "state_before_sha256": f"{index:064x}",
            "state_after_sha256": f"{index + 1:064x}",
        }
        for index in range(10)
    ]
    raw_path = tmp_path / "source-events.jsonl"
    raw_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in events
    )
    raw_path.write_bytes(raw_bytes)
    common = {
        "source_run_id": "run-h10",
        "source_manifest_id": "manifest-h10",
        "source_checkpoint_id": "checkpoint-1",
        "source_checkpoint_sha256": "1" * 64,
        "source_suffix_id": "suffix-1",
        "source_ordered_stream_sha256": "2" * 64,
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
    }
    source = {
        "schema_version": "phase13_source_trajectory_v1",
        **common,
        "source_raw_path": str(raw_path),
        "source_raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "event_count": 10,
    }
    source_path = tmp_path / "source-manifest.json"
    source_bytes = (json.dumps(source, sort_keys=True) + "\n").encode()
    source_path.write_bytes(source_bytes)
    authority = {
        "schema_version": "phase13_prefix_conformance_authority_v1",
        "authority_id": "phase13-prefix-authority-v1",
        "conformance_id": "phase13-ten-condition-prefix-v1",
        "checker_version": "phase13-prefix-checker-v1",
        "checker_script_sha256": "3" * 64,
        "checker_config_sha256": "4" * 64,
        "repository_commit": "e2f980e4b4e4a79479a08d2844f32eaeb4aa05e3",
        **common,
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "analysis_windows": [
            {"analysis_window_id": "accuracy-h2", "window_length": 2, "event_time_start": 0, "event_time_end": 1, "evidence_status": "prespecified_sensitivity", "multiplicity_status": "descriptive_no_inferential_family"},
            {"analysis_window_id": "recurrence-h2", "window_length": 2, "event_time_start": 0, "event_time_end": 1, "evidence_status": "descriptive", "multiplicity_status": "estimation_only"},
            {"analysis_window_id": "accuracy-h5", "window_length": 5, "event_time_start": 0, "event_time_end": 4, "evidence_status": "confirmatory_primary", "multiplicity_status": "primary_holm_family"},
            {"analysis_window_id": "recurrence-h5", "window_length": 5, "event_time_start": 0, "event_time_end": 4, "evidence_status": "confirmatory_secondary", "multiplicity_status": "estimation_only"},
        ],
    }
    authority_path = tmp_path / "authority.json"
    authority_bytes = (json.dumps(authority, sort_keys=True) + "\n").encode()
    authority_path.write_bytes(authority_bytes)
    return (
        authority_path,
        hashlib.sha256(authority_bytes).hexdigest(),
        source_path,
        hashlib.sha256(source_bytes).hexdigest(),
        authority,
        source,
    )


def _derive(fixture: tuple[Path, str, Path, str, dict[str, object], dict[str, object]]):
    authority_path, authority_hash, source_path, source_hash, _, _ = fixture
    return derive_prefix_windows(authority_path, authority_hash, source_path, source_hash)


def test_h2_and_h5_derive_from_one_h10_source_with_zero_execution(tmp_path: Path) -> None:
    result = _derive(_fixture(tmp_path))

    assert isinstance(result, PrefixDerivationArtifact)
    assert tuple(check.check_id for check in result.checks) == CHECK_IDS
    assert all(check.verdict == "pass" and len(check.evidence_sha256) == 64 for check in result.checks)
    assert {(row.window_length, row.event_time_range) for row in result.rows} == {
        (2, (0, 1)), (5, (0, 4))
    }
    assert all(tuple(event.event_index for event in row.events) == tuple(range(row.window_length)) for row in result.rows)
    assert all(row.no_new_provider_execution for row in result.rows)
    assert all(row.source_checkpoint_id == "checkpoint-1" for row in result.rows)
    assert all(row.analysis_window.analysis_window_id == row.analysis_window_id for row in result.rows)
    assert all(row.conformance_id == "phase13-ten-condition-prefix-v1" for row in result.rows)
    assert all((row.provider_calls, row.task_presentations, row.memory_evolutions) == (0, 0, 0) for row in result.rows)
    h2 = [row for row in result.rows if row.window_length == 2]
    assert [(row.evidence_status, row.multiplicity_status) for row in h2] == [
        ("prespecified_sensitivity", "descriptive_no_inferential_family"),
        ("descriptive", "estimation_only"),
    ]


Mutation = Callable[[dict[str, object], dict[str, object]], None]


def _mutate_source(field: str, value: object) -> Mutation:
    def mutate(_authority: dict[str, object], source: dict[str, object]) -> None:
        source[field] = value

    return mutate


def _mutate_raw_source_hash(_authority: dict[str, object], source: dict[str, object]) -> None:
    raw_path = Path(str(source["source_raw_path"]))
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[0]["status"] = "mutated"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )


def _mutate_state_chain(_authority: dict[str, object], source: dict[str, object]) -> None:
    raw_path = Path(str(source["source_raw_path"]))
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[1]["state_before_sha256"] = "f" * 64
    raw_bytes = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    raw_path.write_bytes(raw_bytes)
    source["source_raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()


@pytest.mark.parametrize(
    ("check_id", "mutate"),
    [
        ("checkpoint_source_identity", _mutate_source("source_checkpoint_id", "drift")),
        ("suffix_order", _mutate_source("source_ordered_stream_sha256", "9" * 64)),
        ("execution_contract_identity", _mutate_source("model_snapshot_id", "drift-model")),
        ("execution_contract_identity", _mutate_source("source_execution_owner_id", "drift-owner")),
        ("native_semantics", _mutate_state_chain),
        ("session_randomness", _mutate_source("session_contract_id", "drift-session")),
        ("intervention_identity", _mutate_source("intervention_id", "drift-intervention")),
        ("future_feedback_cutoff", _mutate_source("future_feedback_cutoff", 1)),
        ("source_manifest_identity", _mutate_source("source_manifest_id", "drift-manifest")),
        ("exact_event_range", _mutate_source("event_count", 9)),
        ("source_raw_bytes", _mutate_raw_source_hash),
    ],
)
def test_each_failed_invariant_is_visible_and_emits_no_artifact(
    tmp_path: Path, check_id: str, mutate: Mutation
) -> None:
    fixture = _fixture(tmp_path)
    authority_path, authority_hash, source_path, _, authority, source = fixture
    mutated = copy.deepcopy(source)
    mutate(authority, mutated)
    source_bytes = (json.dumps(mutated, sort_keys=True) + "\n").encode()
    source_path.write_bytes(source_bytes)
    authority["source_manifest_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    authority_bytes = (json.dumps(authority, sort_keys=True) + "\n").encode()
    authority_path.write_bytes(authority_bytes)

    result = derive_prefix_windows(
        authority_path,
        hashlib.sha256(authority_bytes).hexdigest(),
        source_path,
        hashlib.sha256(source_bytes).hexdigest(),
    )

    assert isinstance(result, NotExchangeable)
    assert result.derived_artifact is None
    assert check_id in {check.check_id for check in result.checks if check.verdict == "fail"}
    assert all(row.realization_disposition == "not_exchangeable" for row in result.registered_windows)
    assert result.provider_calls == result.task_presentations == result.memory_evolutions == 0


def test_equal_length_status_drift_is_not_exchangeable(tmp_path: Path) -> None:
    authority_path, _, source_path, source_hash, authority, _ = _fixture(tmp_path)
    windows = authority["analysis_windows"]
    assert isinstance(windows, list)
    windows[1]["evidence_status"] = windows[0]["evidence_status"]
    authority_bytes = (json.dumps(authority, sort_keys=True) + "\n").encode()
    authority_path.write_bytes(authority_bytes)

    result = derive_prefix_windows(
        authority_path,
        hashlib.sha256(authority_bytes).hexdigest(),
        source_path,
        source_hash,
    )

    assert isinstance(result, NotExchangeable)
    assert "source_manifest_identity" in {
        check.check_id for check in result.checks if check.verdict == "fail"
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checker_script_sha256", "f" * 64),
        ("checker_config_sha256", "f" * 64),
        ("repository_commit", "f" * 40),
    ],
)
def test_external_authority_hash_rejects_conformance_authority_drift(
    tmp_path: Path, field: str, value: str
) -> None:
    authority_path, authority_hash, source_path, source_hash, authority, _ = _fixture(tmp_path)
    authority[field] = value
    authority_path.write_text(json.dumps(authority, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="AUTHORITY_HASH_MISMATCH"):
        derive_prefix_windows(authority_path, authority_hash, source_path, source_hash)
