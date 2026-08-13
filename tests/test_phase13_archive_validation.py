from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import pytest

from memcontam.readiness import phase13_cli
from .test_phase13_archive_fixture import complete_archive, mutate, resign_source, write_archive


Mutation = Callable[[dict[str, Any]], None]


def _set(path: tuple[str | int, ...], value: Any) -> Mutation:
    def apply(payload: dict[str, Any]) -> None:
        target: Any = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return apply


MUTATIONS: tuple[tuple[str, Mutation, str], ...] = (
    ("authority", _set(("authorities", "analysis", "sha256"), "0" * 64), "AUTHORITY_HASH_MISMATCH"),
    ("checkpoint", _set(("source_attempts", 1, "events", 0, "source_checkpoint_id"), "wrong"), "SOURCE_CHECKPOINT_UNREGISTERED"),
    ("source_run", _set(("derived_windows", 0, "source_run_id"), "missing"), "DERIVED_SOURCE_RUN_MISSING"),
    ("event_range", _set(("derived_windows", 0, "source_event_range"), [0, 0]), "DERIVED_EVENT_RANGE_MISMATCH"),
    ("window_status", _set(("derived_windows", 0, "evidence_status"), "descriptive"), "WINDOW_STATUS_MISMATCH"),
    ("family", _set(("aggregates", 0, "family_id"), "other"), "AGGREGATE_FAMILY_MISMATCH"),
    ("owner", _set(("provider_ledger", 0, "execution_owner_id"), "phase13-offline-compute-owner-v1"), "PROVIDER_OWNER_MISMATCH"),
    ("native", _set(("source_attempts", 1, "events", 0, "native_state_id"), "other"), "NATIVE_STATE_MISMATCH"),
    ("lineage", _set(("source_attempts", 1, "events", 1, "lineage_parent_ids"), []), "EVENT_LINEAGE_MISMATCH"),
    ("offline", _set(("offline_ledger", 0, "provider_calls"), 1), "OFFLINE_PROVIDER_WORK_FORBIDDEN"),
    ("historical", _set(("historical_reference", "availability"), "available"), "HISTORICAL_REFERENCE_INVALID"),
    ("rerun", _set(("source_attempts", 1, "rerun_parent_id"), "missing"), "RERUN_PARENT_MISMATCH"),
    ("weights", _set(("aggregates", 0, "weights"), {"game24-seed-10000": 2.0}), "WEIGHT_RENORMALIZATION_FORBIDDEN"),
    ("claim", _set(("claims", 0, "estimate"), 0.5), "CLAIM_RECONSTRUCTION_MISMATCH"),
)


def _resigned_event(path: tuple[str, ...], value: Any) -> Callable[[Path, dict[str, Any]], None]:
    def apply(root: Path, payload: dict[str, Any]) -> None:
        event = payload["source_attempts"][1]["events"][0]
        target = event
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        resign_source(root, payload)

    return apply


def _short_source(root: Path, payload: dict[str, Any]) -> None:
    payload["source_attempts"][1]["events"] = payload["source_attempts"][1]["events"][:9]
    payload["provider_ledger"] = payload["provider_ledger"][:9]
    resign_source(root, payload)


def _arbitrary_run(root: Path, payload: dict[str, Any]) -> None:
    payload["source_attempts"][1]["source_run_id"] = "arbitrary-source-run"
    payload["derived_windows"][0]["source_run_id"] = "arbitrary-source-run"
    payload["aggregates"][0]["original_weights"] = {"arbitrary-source-run": 1.0}
    payload["aggregates"][0]["weights"] = {"arbitrary-source-run": 1.0}
    resign_source(root, payload)


def _duplicate_call(root: Path, payload: dict[str, Any]) -> None:
    payload["source_attempts"][1]["events"][1]["semantic_call_id"] = "call-0"
    payload["provider_ledger"].pop(1)
    resign_source(root, payload)


def _role_swap(_root: Path, payload: dict[str, Any]) -> None:
    execution = payload["authorities"]["execution"]
    payload["authorities"]["execution"] = payload["authorities"]["analysis"]
    payload["authorities"]["analysis"] = execution


def _forged_invalidation(_root: Path, payload: dict[str, Any]) -> None:
    payload["source_attempts"][0]["raw_evidence_sha256"] = "2" * 64


def _zero_weights(_root: Path, payload: dict[str, Any]) -> None:
    payload["aggregates"][0]["original_weights"] = {"game24-seed-10000": 0.0}
    payload["aggregates"][0]["weights"] = {"game24-seed-10000": 0.0}


REVIEW_MUTATIONS: tuple[tuple[str, Callable[[Path, dict[str, Any]], None], str], ...] = (
    ("checkpoint", _resigned_event(("source_checkpoint_id",), "checkpoint-forged"), "SOURCE_CHECKPOINT_UNREGISTERED"),
    ("task", _resigned_event(("task",), "word_sorting"), "SOURCE_TASK_MISMATCH"),
    ("model", _resigned_event(("model",), "forged-model"), "SOURCE_MODEL_MISMATCH"),
    ("session", _resigned_event(("session_id",), "other-session"), "SOURCE_SESSION_MISMATCH"),
    ("intervention", _resigned_event(("intervention_id",), "forged-root"), "SOURCE_INTERVENTION_MISMATCH"),
    ("write", _resigned_event(("write_event_ids",), ["forged-write"]), "SOURCE_WRITE_IDENTITY_MISMATCH"),
    ("retention", _resigned_event(("retention_event_ids",), ["forged-retain"]), "SOURCE_RETENTION_IDENTITY_MISMATCH"),
    ("eviction", _resigned_event(("eviction_event_ids",), ["forged-evict"]), "SOURCE_EVICTION_IDENTITY_MISMATCH"),
    ("state_before", _resigned_event(("state_before_sha256",), "0" * 64), "EVENT_INITIAL_STATE_MISMATCH"),
    ("short_h10", _short_source, "SOURCE_H10_RANGE_INVALID"),
    ("arbitrary_run", _arbitrary_run, "SOURCE_RUN_UNREGISTERED"),
    ("duplicate_call", _duplicate_call, "DUPLICATE_SEMANTIC_CALL_ID"),
    ("role_swap", _role_swap, "AUTHORITY_ROLE_MISMATCH"),
    ("forged_invalidation", _forged_invalidation, "INVALIDATED_RAW_EVIDENCE_MISMATCH"),
    ("zero_weights", _zero_weights, "AGGREGATE_WEIGHT_INVALID"),
    ("offline_cost", lambda _root, row: row["offline_ledger"][0].update(cost_microusd=1), "OFFLINE_PROVIDER_WORK_FORBIDDEN"),
    ("duplicate_window", lambda _root, row: row["derived_windows"].append(row["derived_windows"][0].copy()), "DUPLICATE_DERIVED_WINDOW_ID"),
    ("duplicate_aggregate", lambda _root, row: row["aggregates"].append(row["aggregates"][0].copy()), "DUPLICATE_AGGREGATE_ID"),
    ("duplicate_claim", lambda _root, row: row["claims"].append(row["claims"][0].copy()), "DUPLICATE_CLAIM_ID"),
)


def test_reconstructs_complete_phase13_archive(tmp_path: Path) -> None:
    from memcontam.manifests.phase13_archive_validation import validate_phase13_archive

    complete_archive(tmp_path)

    report = validate_phase13_archive(tmp_path)

    assert report.archive_valid is True
    assert report.reason_code is None
    assert report.resolved_edges == 36
    assert report.claim_ids == ("game24-h5-score-claim",)


@pytest.mark.parametrize(("name", "mutation", "code"), MUTATIONS, ids=[row[0] for row in MUTATIONS])
def test_single_field_mutation_fails_closed(
    tmp_path: Path, name: str, mutation: Mutation, code: str
) -> None:
    del name
    from memcontam.manifests.phase13_archive_validation import validate_phase13_archive

    payload = mutate(complete_archive(tmp_path), mutation)
    write_archive(tmp_path, payload)

    report = validate_phase13_archive(tmp_path)

    assert report.archive_valid is False
    assert report.reason_code == code
    assert report.claim_ids == ()


def test_phase13_cli_routes_archive_validator(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    complete_archive(tmp_path)

    phase13_cli.run(argparse.Namespace(phase13_command="validate-calibration-v2-archive", archive=tmp_path))

    assert '"archive_valid": true' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "mutation", "code"), REVIEW_MUTATIONS, ids=[row[0] for row in REVIEW_MUTATIONS]
)
def test_reviewer_resigned_mutations_fail_closed(
    tmp_path: Path,
    name: str,
    mutation: Callable[[Path, dict[str, Any]], None],
    code: str,
) -> None:
    del name
    from memcontam.manifests.phase13_archive_validation import validate_phase13_archive

    payload = complete_archive(tmp_path)
    mutation(tmp_path, payload)
    write_archive(tmp_path, payload)

    report = validate_phase13_archive(tmp_path)

    assert report.archive_valid is False
    assert report.reason_code == code
    assert report.claim_ids == ()
