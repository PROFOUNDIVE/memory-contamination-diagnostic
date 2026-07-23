from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from memcontam.config.phase12 import (
    CanonicalExploratoryConfig,
    Phase12ConfigError,
    load_all_canonical_configs,
    load_phase12_config,
)
from memcontam.experiment.phase12.contracts import canonical_json_hash
from memcontam.experiment.phase12.code_matrix import build_code_matrix
from memcontam.phase12_types import CANONICAL_RUN_FAMILY_MEMBERS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "phase12"
CONFIG_NAMES = (
    "readiness.yaml",
    "pilot_a.yaml",
    "pilot_b.yaml",
    "main_3w.yaml",
    "main_5w.yaml",
    "exploratory_code.yaml",
)


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_all_pre_route_and_candidate_configs_resolve_without_selection() -> None:
    resolved = load_all_canonical_configs(CONFIG_ROOT)

    assert tuple(resolved) == CONFIG_NAMES
    assert resolved == load_all_canonical_configs(CONFIG_ROOT)
    assert {item.contract_level for item in resolved.values()} == {"phase12"}
    assert {item.logging_schema_version for item in resolved.values()} == {"logging_v3"}
    assert all(item.source.route_selection_manifest_id is None for item in resolved.values())
    assert all(item.source.seed_allocation_manifest_id is None for item in resolved.values())
    assert CANONICAL_RUN_FAMILY_MEMBERS == {
        "readiness": ("readiness",),
        "pilot_a": ("pilot_a",),
        "pilot_b": ("pilot_b",),
        "main": ("main_a", "main_b", "main_c"),
        "exploratory_code": ("exploratory_code",),
    }

    for name in ("main_3w.yaml", "main_5w.yaml"):
        source = resolved[name].source
        assert source.selection_status == "candidate"
        assert source.candidate_route == name.removeprefix("main_").removesuffix(".yaml")

    exploratory = resolved["exploratory_code.yaml"].source
    assert isinstance(exploratory, CanonicalExploratoryConfig)
    assert exploratory.activation_status == "inactive"
    plan = build_code_matrix(
        {
            **exploratory.model_dump(mode="python"),
            "oci_contract_path": ROOT / exploratory.oci_contract_path,
        }
    )
    assert (
        plan.exploratory_run_template_registry_id
        == exploratory.registry_ids["exploratory_run_templates"]
    )
    assert exploratory.exploratory_run_template_registry_hash == canonical_json_hash(
        {
            "abstract_slots": exploratory.abstract_slots,
            "estimated_exploratory_calls": exploratory.estimated_exploratory_calls,
            "registry_id": exploratory.exploratory_run_template_registry_id,
        }
    )


def test_configs_reject_missing_or_cross_layer_ids(tmp_path: Path) -> None:
    missing = _yaml(CONFIG_ROOT / "readiness.yaml")
    registry_ids = missing["registry_ids"]
    assert isinstance(registry_ids, dict)
    registry_ids.pop("candidate")
    missing_path = tmp_path / "readiness.yaml"
    missing_path.write_text(yaml.safe_dump(missing), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match="FROZEN_REGISTRY_ID_MISSING"):
        load_phase12_config(missing_path)

    cross_layer = _yaml(CONFIG_ROOT / "main_3w.yaml")
    registry_ids = cross_layer["registry_ids"]
    assert isinstance(registry_ids, dict)
    registry_ids["candidate"] = "exploratory-registry-v1"
    cross_layer_path = tmp_path / "main_3w.yaml"
    cross_layer_path.write_text(yaml.safe_dump(cross_layer), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match="CROSS_LAYER_REGISTRY_ID"):
        load_phase12_config(cross_layer_path)

    stale_route = _yaml(CONFIG_ROOT / "readiness.yaml")
    candidate_routes = stale_route["candidate_routes"]
    assert isinstance(candidate_routes, list)
    assert isinstance(candidate_routes[0], dict)
    candidate_routes[0]["run_template_registry_id"] = "local-registry"
    stale_route_path = tmp_path / "stale-route.yaml"
    stale_route_path.write_text(yaml.safe_dump(stale_route), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match="FROZEN_REGISTRY_ID_UNKNOWN"):
        load_phase12_config(stale_route_path)

    stale_code = _yaml(CONFIG_ROOT / "exploratory_code.yaml")
    stale_code["exploratory_run_template_registry_hash"] = "stale"
    stale_code_path = tmp_path / "stale-code.yaml"
    stale_code_path.write_text(yaml.safe_dump(stale_code), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match="FROZEN_REGISTRY_HASH_UNKNOWN"):
        load_phase12_config(stale_code_path)

    mutated_registry = _yaml(CONFIG_ROOT / "exploratory_code.yaml")
    mutated_registry["abstract_slots"] = ["game24|exploratory|slot-002"]
    mutated_registry_path = tmp_path / "mutated-registry.yaml"
    mutated_registry_path.write_text(yaml.safe_dump(mutated_registry), encoding="utf-8")

    with pytest.raises(Phase12ConfigError, match="FROZEN_REGISTRY_HASH_UNKNOWN"):
        load_phase12_config(mutated_registry_path)
