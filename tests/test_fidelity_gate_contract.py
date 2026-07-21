from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from memcontam.baselines.contracts import (
    BASELINE_EXECUTION_CONTRACT_V2,
    BASELINE_FIDELITY_V2,
    FAILURE_TAXONOMY_V2,
)
from memcontam.clients.config import ProviderConfig
from memcontam.clients.provider_profile import normalize_provider_profile
from memcontam.cli import validate_config
from memcontam.config.resolution import resolve_run_config


ROOT = Path(__file__).resolve().parents[1]


def _profile():
    return normalize_provider_profile(
        ProviderConfig(provider="replay"),
        served_models=["replay"],
        model_snapshots={"replay": "v1"},
    )


def _v2_config(gate_layer: str = "structural") -> dict:
    return {
        "logging": {
            "memory_policy_version": BASELINE_FIDELITY_V2,
            "prompt_version": BASELINE_FIDELITY_V2,
        },
        "run": {
            "retry_policy_version": BASELINE_FIDELITY_V2,
            "baseline_execution_contract_version": BASELINE_EXECUTION_CONTRACT_V2,
            "failure_taxonomy_version": FAILURE_TAXONOMY_V2,
            "fidelity_gate_layer": gate_layer,
        },
    }


def test_fidelity_v2_config_requires_the_complete_version_tuple() -> None:
    config = _v2_config()
    del config["run"]["failure_taxonomy_version"]

    with pytest.raises(ValueError, match="complete Baseline-Fidelity-V2 version tuple"):
        resolve_run_config(config, provider_profile=_profile())


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("logging", "memory_policy_version"),
        ("logging", "prompt_version"),
        ("run", "retry_policy_version"),
    ],
)
def test_fidelity_v2_config_rejects_v1_version_tuple_members(section: str, field: str) -> None:
    config = _v2_config()
    config[section][field] = "baseline_fidelity_v1"

    with pytest.raises(ValueError, match="complete Baseline-Fidelity-V2 version tuple"):
        resolve_run_config(config, provider_profile=_profile())


def test_fidelity_v2_validate_config_rejects_v1_prompt_version(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "baseline_fidelity_v2_structural_replay.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["logging"]["prompt_version"] = "baseline_fidelity_v1"
    config_path = tmp_path / "invalid-v2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(SystemExit, match="complete Baseline-Fidelity-V2 version tuple"):
        validate_config(config_path)


def test_structural_f1a_config_validates_with_its_explicit_replay_contract() -> None:
    validate_config(ROOT / "configs" / "baseline_fidelity_v2_structural_replay.yaml")


def test_fidelity_v2_config_rejects_unknown_fidelity_gate_layer() -> None:
    with pytest.raises(ValueError, match="fidelity_gate_layer"):
        resolve_run_config(_v2_config("F1B"), provider_profile=_profile())


@pytest.mark.parametrize("gate_layer", ["structural", "source_contract", "real_retriever"])
def test_fidelity_v2_config_resolves_only_declared_gate_layers(gate_layer: str) -> None:
    resolved = resolve_run_config(_v2_config(gate_layer), provider_profile=_profile())

    assert resolved["run"]["fidelity_gate_layer"] == gate_layer


def test_structural_f1a_config_cannot_claim_f1b_or_f1c() -> None:
    resolved = resolve_run_config(
        copy.deepcopy(_v2_config("structural")), provider_profile=_profile()
    )

    assert resolved["run"]["fidelity_gate_layer"] == "structural"
    assert resolved["run"]["fidelity_gate_layer"] not in {"source_contract", "real_retriever"}
