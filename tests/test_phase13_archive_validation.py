from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import pytest

from memcontam.readiness import phase13_cli
from test_phase13_archive_fixture import complete_archive, mutate, write_archive


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
    ("checkpoint", _set(("source_attempts", 1, "events", 0, "source_checkpoint_id"), "wrong"), "SOURCE_CHECKPOINT_MISMATCH"),
    ("source_run", _set(("derived_windows", 0, "source_run_id"), "missing"), "DERIVED_SOURCE_RUN_MISSING"),
    ("event_range", _set(("derived_windows", 0, "source_event_range"), [0, 3]), "DERIVED_EVENT_RANGE_MISMATCH"),
    ("window_status", _set(("derived_windows", 0, "status"), "NOT_ESTIMABLE"), "WINDOW_STATUS_PROMOTION_FORBIDDEN"),
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


def test_reconstructs_complete_phase13_archive(tmp_path: Path) -> None:
    from memcontam.manifests.phase13_archive_validation import validate_phase13_archive

    complete_archive(tmp_path)

    report = validate_phase13_archive(tmp_path)

    assert report.archive_valid is True
    assert report.reason_code is None
    assert report.resolved_edges == 30
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
