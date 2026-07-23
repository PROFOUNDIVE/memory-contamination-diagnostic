from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from memcontam.config import phase12
from memcontam.config.phase12 import Phase12ConfigError, load_phase12_config, resolve_phase12_config
from memcontam.experiment.phase12 import contracts
from memcontam.experiment.phase12.contracts import (
    BaselineConditionSpec,
    MftManifest,
    RouteFeasibilityReport,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-CONFIG-001.json"
ROUTE_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-ROUTE-001.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _route_fixture() -> dict[str, Any]:
    return json.loads(ROUTE_FIXTURE_PATH.read_text(encoding="utf-8"))


def _set(payload: dict[str, Any], path: str, value: Any) -> None:
    target: Any = payload
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    target[parts[-1]] = value


def _remove(payload: dict[str, Any], path: str) -> None:
    target: Any = payload
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    target.pop(int(parts[-1]) if isinstance(target, list) else parts[-1])


def _set_many(payload: dict[str, Any], values: dict[str, Any]) -> None:
    for path, value in values.items():
        _set(payload, path, value)


def test_loads_the_frozen_phase12_config_contract() -> None:
    resolved = resolve_phase12_config(load_phase12_config(FIXTURE_PATH))

    assert resolved.repository_commit == "830b89c8c169ffa9cdea472887fdae134dbae7cf"
    assert (
        resolved.authoritative_experiment_design.sha256
        == "984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0"
    )
    assert resolved.logging_schema_version == "logging_v3"
    assert resolved.contract_level == "phase12"
    assert all(
        isinstance(condition, BaselineConditionSpec) for condition in resolved.source.conditions
    )


@pytest.mark.parametrize(
    ("container", "field_name"),
    [
        (None, "sensitivity_cells"),
        ("template_package", "runtime_refs"),
        ("template_package", "call_policy"),
    ],
)
def test_rejects_missing_or_inapplicable_matrix_fields(
    tmp_path: Path, container: str | None, field_name: str
) -> None:
    payload = copy.deepcopy(_fixture())
    target = payload if container is None else payload[container]
    assert isinstance(target, dict)
    target.pop(field_name)
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match="REQUIRED_CONFIG_CELL_MISSING"):
        load_phase12_config(path)


def test_rejects_unrelated_sensitivity_fields(tmp_path: Path) -> None:
    payload = copy.deepcopy(_fixture())
    payload["sensitivity_cells"][1]["horizon"] = 2
    path = tmp_path / "invalid-sensitivity.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match="UNRELATED_SENSITIVITY_FIELD"):
        load_phase12_config(path)


@pytest.mark.parametrize(
    ("variant_id", "reason"),
    [(variant["id"], variant["reason"]) for variant in _fixture()["invalid_variants"]],
)
def test_rejects_frozen_config_invalid_variants(
    tmp_path: Path, variant_id: str, reason: str
) -> None:
    payload = copy.deepcopy(_fixture())
    mutations = {
        "timing-with-horizon": lambda: _set(payload, "sensitivity_cells.1.horizon", 2),
        "primary-python": lambda: _set(payload, "tool_mode", "python_sandbox"),
        "primary-online-rag": lambda: _set(payload, "conditions.2.rag_mode", "online_ext"),
        "nomem-arm": lambda: _set(
            payload, "conditions.0.execution_key", {"kind": "memory_arm", "arm": "clean"}
        ),
        "premature-selected-route": lambda: _set(
            payload, "route_selection_manifest_id", "route-001"
        ),
        "prefix-as-clean-arm": lambda: _set(
            payload, "prefix_execution_key", {"kind": "memory_arm", "arm": "clean"}
        ),
        "selected-route-without-seed-allocation": lambda: _set_many(
            payload, {"selection_status": "selected", "route_selection_manifest_id": "route-001"}
        ),
        "scientific-exploratory-without-activation": lambda: _set_many(
            payload,
            {
                "selection_status": "selected",
                "route_selection_manifest_id": "route-001",
                "seed_allocation_manifest_id": "seed-001",
                "exploratory_activation_status": "active",
            },
        ),
        "missing-required-sensitivity-cell": lambda: _remove(
            payload, "template_package.sensitivity.0"
        ),
        "missing-replication-matrix": lambda: _remove(payload, "template_package.replication"),
        "missing-call-policy": lambda: _remove(payload, "template_package.call_policy"),
        "illegal-robustness-evidence-layer": lambda: _set(
            payload, "template_package.sensitivity.0.evidence_layer", "robustness"
        ),
        "stale-repository-commit": lambda: _set(payload, "repository_commit", "stale"),
        "stale-experiment-design": lambda: _set(
            payload, "authoritative_experiment_design.sha256", "stale"
        ),
        "phase12-logging-v2": lambda: _set(
            payload, "logging_contract.schema_version", "logging_v2"
        ),
        "reversed-main-replication-models": lambda: _set_many(
            payload,
            {
                "template_package.core.model_snapshot": "frontier-model-v1",
                "template_package.replication.model_snapshot": "gpt-4o-v1",
            },
        ),
    }
    mutations[variant_id]()
    path = tmp_path / f"{variant_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match=reason):
        load_phase12_config(path)


def test_rejects_unknown_top_level_config_field(tmp_path: Path) -> None:
    payload = _fixture()
    payload["hidden_default"] = True
    path = tmp_path / "unknown-field.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match="extra_forbidden"):
        load_phase12_config(path)


def test_route_fixture_governance_shapes_parse_without_semantic_validation() -> None:
    route = _route_fixture()
    reports = tuple(
        RouteFeasibilityReport.model_validate(item) for item in route["feasibility_reports"]
    )
    mft = MftManifest.model_validate(route["valid_mft_manifest"])

    assert {report.candidate_route for report in reports} == {"3w", "5w"}
    assert all(report.run_template_registry_id and report.pilot_b_manifest_id for report in reports)
    assert mft.manifest_id == "mft-001"
    assert not hasattr(phase12, "validate_route_selection")
    assert not hasattr(phase12, "validate_exploratory_activation")
    assert not hasattr(contracts, "validate_route_selection")
    assert not hasattr(contracts, "validate_exploratory_activation")
