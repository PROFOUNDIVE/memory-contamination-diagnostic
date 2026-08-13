from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from memcontam.readiness.phase13_execution_contract import (
    Phase13ExecutionError,
    load_execution_registry,
    parse_execution_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/phase13/authority/execution_registry_v1.json"


def _payload() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _resign(payload: dict[str, Any]) -> bytes:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("registry_hash", None)
    payload["registry_hash"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return json.dumps(payload, sort_keys=True).encode()


def test_windows_have_explicit_independent_status_and_one_source_execution() -> None:
    registry = load_execution_registry(REGISTRY, ROOT)
    windows = {row.analysis_window_id: row for row in registry.analysis_windows}
    primary = [row for row in registry.analysis_windows if row.evidence_status == "confirmatory_primary"]

    assert [(row.analysis_window_id, row.window_length) for row in primary] == [
        ("accuracy-h5-primary", 5)
    ]
    assert windows["accuracy-h5-primary"].multiplicity_status == "primary_holm_family"
    assert windows["recurrence-h5-secondary"].multiplicity_status == "estimation_only"
    assert windows["persistence-h5-secondary"].evidence_status == "confirmatory_secondary"
    assert all(row.event_time_start == 0 and row.event_time_end == row.window_length - 1 for row in registry.analysis_windows)
    assert all(row.provider_execution_multiplicity == 0 for row in registry.analysis_windows if row.window_length in {2, 5})
    assert all(row.realization_disposition == "prefix_view" for row in registry.analysis_windows if row.window_length in {2, 5})
    assert all(row.realization_disposition == "source_execution" for row in registry.analysis_windows if row.window_length == 10)
    assert sum(row.provider_execution_multiplicity for row in registry.analysis_windows) == 1


def test_main_and_calibration_call_illustrations_recompute_independently() -> None:
    registry = load_execution_registry(REGISTRY, ROOT)
    prefix, execution = registry.call_components

    assert len(registry.execution_templates) == 51
    assert sum(row.nominal_semantic_calls_per_trial for row in registry.execution_templates if row.task == "game24") == execution.nominal_calls_per_activation
    assert sum(row.raw_maximum_semantic_calls_per_trial for row in registry.execution_templates if row.task == "game24") == execution.raw_maximum_calls_per_activation

    for illustration, multiplicity, expected in (
        (registry.planning_illustrations.main, "main_seed_multiplicity", (7680, 11370, 11939)),
        (registry.planning_illustrations.calibration, "calibration_seed_multiplicity", (9216, 13644, 14327)),
    ):
        task_seeds = sum(getattr(row, multiplicity) for row in registry.execution_templates if row.baseline == "nomem")
        nominal = task_seeds * prefix.nominal_calls_per_activation + 10 * sum(
            getattr(row, multiplicity) * row.nominal_semantic_calls_per_trial
            for row in registry.execution_templates
        )
        raw = task_seeds * prefix.raw_maximum_calls_per_activation + 10 * sum(
            getattr(row, multiplicity) * row.raw_maximum_semantic_calls_per_trial
            for row in registry.execution_templates
        )
        reserved = (raw * 105 + 99) // 100
        assert (nominal, raw, reserved) == expected
        assert (
            illustration.nominal_semantic_calls,
            illustration.raw_maximum_semantic_calls,
            illustration.reserved_semantic_calls,
        ) == expected
        assert illustration.raw_maximum_transport_attempts == raw * 4
        assert illustration.reserved_transport_attempts == reserved * 4


Mutation = Callable[[dict[str, Any]], None]


def _window(payload: dict[str, Any], identity: str) -> dict[str, Any]:
    return next(row for row in payload["analysis_windows"] if row["analysis_window_id"] == identity)


def _primary_count(payload: dict[str, Any]) -> None:
    _window(payload, "recurrence-h5-secondary")["evidence_status"] = "confirmatory_primary"


def _realization(payload: dict[str, Any]) -> None:
    _window(payload, "accuracy-h5-primary")["realization_disposition"] = "source_execution"


def _multiplicity(payload: dict[str, Any]) -> None:
    _window(payload, "recurrence-h5-secondary")["multiplicity_status"] = "primary_holm_family"


def _provider_multiplicity(payload: dict[str, Any]) -> None:
    _window(payload, "accuracy-h2-sensitivity")["provider_execution_multiplicity"] = 1


def _remove_window(payload: dict[str, Any]) -> None:
    payload["analysis_windows"].pop()


def _add_window(payload: dict[str, Any]) -> None:
    row = copy.deepcopy(_window(payload, "recurrence-h10-descriptive"))
    row["analysis_window_id"] = "arbitrary-h10"
    payload["analysis_windows"].append(row)


def _status_drift(payload: dict[str, Any]) -> None:
    _window(payload, "recurrence-h5-secondary")["evidence_status"] = "descriptive"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_primary_count, "PRIMARY_WINDOW_INVALID"),
        (_realization, "WINDOW_REALIZATION_INVALID"),
        (_multiplicity, "WINDOW_MULTIPLICITY_INVALID"),
        (_provider_multiplicity, "WINDOW_EXECUTION_MULTIPLICITY_INVALID"),
        (_remove_window, "WINDOW_INVENTORY_INVALID"),
        (_add_window, "WINDOW_INVENTORY_INVALID"),
        (_status_drift, "WINDOW_INVENTORY_INVALID"),
    ],
)
def test_resigned_window_mutations_fail_without_status_inheritance(
    mutate: Mutation, code: str
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(Phase13ExecutionError) as caught:
        parse_execution_registry(_resign(payload), ROOT)

    assert caught.value.code == code
