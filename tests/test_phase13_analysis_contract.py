from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from memcontam.readiness.phase13_analysis_contract import (
    Phase13AnalysisError,
    load_analysis_registry,
    parse_analysis_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/phase13/authority/analysis_registry_v1.json"
TASKS = ("game24", "math_equation_balancer", "word_sorting")
BASELINES = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
PAIRS = (
    ("P01", "fh_bounded", "rag_frozen", "required_confirmatory", True),
    ("P02", "fh_bounded", "bot_style", "required_confirmatory", True),
    ("P03", "fh_bounded", "reflexion_style", "required_confirmatory", True),
    ("P04", "rag_frozen", "bot_style", "planned_secondary", False),
    ("P05", "rag_frozen", "reflexion_style", "planned_secondary", False),
    ("P06", "bot_style", "reflexion_style", "planned_secondary", False),
)
NON_PRIMARY_WINDOWS = (
    "accuracy-h2-sensitivity",
    "recurrence-h2-descriptive",
    "recurrence-h5-secondary",
    "persistence-h5-secondary",
    "propagation-h5-conditional",
    "collapse-h5-exploratory",
    "accuracy-h10-sensitivity",
    "recurrence-h10-descriptive",
    "persistence-h10-descriptive",
    "propagation-h10-conditional",
    "collapse-h10-exploratory",
)


def _payload() -> dict[str, Any]:
    return json.loads(REGISTRY.read_bytes())


def _resign(payload: dict[str, Any]) -> bytes:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("registry_hash", None)
    payload["registry_hash"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return json.dumps(payload, sort_keys=True).encode()


def test_committed_registry_has_exact_support_and_planning_inventory() -> None:
    registry = load_analysis_registry(REGISTRY, ROOT)

    assert tuple((row.baseline, row.support_population_id) for row in registry.support.level_1) == tuple(
        (baseline, f"l1-{baseline}-structural-support") for baseline in BASELINES
    )
    assert tuple(
        (row.pair_id, row.left_baseline, row.right_baseline, row.status, row.route_gating)
        for row in registry.support.level_2
    ) == PAIRS
    assert registry.support.level_3.baselines == BASELINES
    assert registry.support.level_3.status == "sensitivity_only"
    assert registry.support.level_3.route_gating is False
    assert tuple((row.route, row.level_1, row.level_2, row.level_3) for row in registry.planning.targets) == (
        ("3w", 10, 10, None),
        ("5w", 16, 16, 8),
    )
    assert registry.planning.deterministic.planning_value == "1.000"
    assert registry.planning.deterministic.required_state == "conformance_passed"
    assert registry.planning.stochastic.denominator == 12
    assert registry.planning.stochastic.interval_id == "support-planning-cp95-one-sided-v1"
    assert registry.inference.interval_id == "main-paired-seed-bootstrap95-v1"
    assert registry.planning.stochastic.interval_id != registry.inference.interval_id
    assert registry.planning.calibration_seeds == tuple(range(10000, 10012))


def test_primary_families_are_independently_enumerated_per_task() -> None:
    registry = load_analysis_registry(REGISTRY, ROOT)
    expected_estimands = tuple(
        [f"l1-{baseline}-clean-contam" for baseline in BASELINES]
        + [f"l2-{pair_id.lower()}-clean-contam-did" for pair_id in ("P01", "P02", "P03")]
    )

    for task in TASKS:
        family = next(item for item in registry.inference.families if item.task == task)
        assert family.family_id == f"{task}-h5-primary-holm-v1"
        assert tuple(slot.estimand_id for slot in family.slots) == expected_estimands
        assert tuple(slot.order for slot in family.slots) == tuple(range(1, 8))
        assert all(slot.analysis_window_id == "accuracy-h5-primary" for slot in family.slots)
    assert len(registry.inference.families) == 3
    assert registry.inference.cross_task_family is False
    assert registry.inference.bootstrap.replicates == 20_000
    assert registry.inference.bootstrap.rng_seed == 13
    assert registry.inference.holm.alpha == "0.05"
    assert registry.inference.not_estimable.retain_family_slot is True
    assert registry.inference.not_estimable.reject_null is False
    assert registry.inference.not_estimable.shrink_family is False
    assert registry.inference.not_estimable.renormalize_weights is False


def test_non_primary_windows_and_offline_compute_are_explicit() -> None:
    registry = load_analysis_registry(REGISTRY, ROOT)

    assert tuple(row.analysis_window_id for row in registry.non_primary_windows) == NON_PRIMARY_WINDOWS
    assert all(row.inference_status == "estimation_only" for row in registry.non_primary_windows)
    assert registry.excluded_conditions == ("nomem", "filter_challenge")
    assert tuple(row.operation for row in registry.offline_compute.rows) == (
        "prefix_derivation", "paired_seed_bootstrap", "report_rendering"
    )
    assert all(row.owner_id == registry.offline_compute.owner_id for row in registry.offline_compute.rows)
    assert all(
        (row.provider_calls, row.task_presentations, row.memory_evolutions) == (0, 0, 0)
        for row in registry.offline_compute.rows
    )


Mutation = Callable[[dict[str, Any]], None]


def _slot(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["inference"]["families"][0]["slots"][0]


def _remove_slot(payload: dict[str, Any]) -> None:
    payload["inference"]["families"][0]["slots"].pop()


def _add_slot(payload: dict[str, Any]) -> None:
    payload["inference"]["families"][0]["slots"].append(copy.deepcopy(_slot(payload)))


def _reorder_slot(payload: dict[str, Any]) -> None:
    slots = payload["inference"]["families"][0]["slots"]
    slots[0], slots[1] = slots[1], slots[0]


def _promote_h2(payload: dict[str, Any]) -> None:
    payload["non_primary_windows"][0]["inference_status"] = "primary_holm_family"


def _gate_p04(payload: dict[str, Any]) -> None:
    payload["support"]["level_2"][3]["route_gating"] = True


def _insert_nomem(payload: dict[str, Any]) -> None:
    payload["support"]["level_1"].append(copy.deepcopy(payload["support"]["level_1"][0]))
    payload["support"]["level_1"][-1]["baseline"] = "nomem"


def _insert_filter(payload: dict[str, Any]) -> None:
    payload["support"]["level_1"].append(copy.deepcopy(payload["support"]["level_1"][0]))
    payload["support"]["level_1"][-1]["baseline"] = "filter_challenge"


def _replicate_drift(payload: dict[str, Any]) -> None:
    payload["inference"]["bootstrap"]["replicates"] = 19_999


def _rng_drift(payload: dict[str, Any]) -> None:
    payload["inference"]["bootstrap"]["rng_seed"] = 14


def _alpha_drift(payload: dict[str, Any]) -> None:
    payload["inference"]["holm"]["alpha"] = "0.10"


def _conflate_interval(payload: dict[str, Any]) -> None:
    payload["inference"]["interval_id"] = payload["planning"]["stochastic"]["interval_id"]


def _renormalize(payload: dict[str, Any]) -> None:
    payload["inference"]["not_estimable"]["renormalize_weights"] = True


def _cross_task(payload: dict[str, Any]) -> None:
    payload["inference"]["cross_task_family"] = True


def _offline_charge(payload: dict[str, Any]) -> None:
    payload["offline_compute"]["rows"][0]["provider_calls"] = 1


def _offline_owner(payload: dict[str, Any]) -> None:
    payload["offline_compute"]["rows"][0]["owner_id"] = "phase13-h10-execution-owner-v1"


@pytest.mark.parametrize(
    "mutate",
    [
        _remove_slot, _add_slot, _reorder_slot, _promote_h2, _gate_p04,
        _insert_nomem, _insert_filter, _replicate_drift, _rng_drift, _alpha_drift,
        _conflate_interval, _renormalize, _cross_task, _offline_charge, _offline_owner,
    ],
)
def test_resigned_semantic_drift_is_rejected(mutate: Mutation) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(Phase13AnalysisError) as caught:
        parse_analysis_registry(_resign(payload), ROOT)

    assert caught.value.code == "ANALYSIS_SEMANTICS_INVALID"
