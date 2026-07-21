from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from memcontam.baselines import bot_runtime
from memcontam.baselines.contracts import (
    BASELINE_EXECUTION_CONTRACT_V2,
    BASELINE_FIDELITY_V2,
    FAILURE_TAXONOMY_V2,
)


ROOT = Path(__file__).resolve().parents[1]
V2_AUTHORITY = ROOT / "docs" / "baseline-fidelity-v2.md"
V2_CONFIGS = {
    "structural": ROOT / "configs" / "baseline_fidelity_v2_structural_replay.yaml",
    "source_contract": ROOT / "configs" / "baseline_fidelity_v2_source_contract_replay.yaml",
    "real_retriever": ROOT / "configs" / "baseline_fidelity_v2_bge_smoke.yaml",
}


def test_bot_policy_uses_native_novelty_without_legacy_model_novelty_stage() -> None:
    assert callable(getattr(bot_runtime, "evaluate_native_novelty", None))
    assert callable(getattr(bot_runtime, "freeze_native_transition", None))
    assert "bot_novelty_decide" not in inspect.getsource(bot_runtime)


def test_baseline_fidelity_v2_contract_constants_share_the_v2_boundary() -> None:
    assert BASELINE_FIDELITY_V2 == "baseline_fidelity_v2"
    assert BASELINE_EXECUTION_CONTRACT_V2 == BASELINE_FIDELITY_V2
    assert FAILURE_TAXONOMY_V2 == BASELINE_FIDELITY_V2


def test_v2_gate_configs_share_versions_but_not_fidelity_layers() -> None:
    for expected_layer, path in V2_CONFIGS.items():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        versions = {
            config["logging"]["memory_policy_version"],
            config["logging"]["prompt_version"],
            config["run"]["retry_policy_version"],
            config["run"]["baseline_execution_contract_version"],
            config["run"]["failure_taxonomy_version"],
        }
        assert versions == {BASELINE_FIDELITY_V2}
        assert config["run"]["fidelity_gate_layer"] == expected_layer


def test_v2_authority_forbids_v1_v2_pooling_or_migration() -> None:
    text = V2_AUTHORITY.read_text(encoding="utf-8")
    assert "V1 and V2 artifacts must fail closed when pooled." in text
    assert "There is no artifact migration from V1 to V2." in text
    assert "F1A evidence cannot be represented as F1B or F1C evidence." in text
