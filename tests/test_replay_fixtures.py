from __future__ import annotations

from pathlib import Path

import pytest
import yaml


CONFIG_PATH = Path("configs/g0_rag_bot_faithful_replay.yaml")
FIXTURE_PATH = Path("data/replay/g0_rag_bot_faithful_v1.yaml")
VERSIONED_CONFIGS = [
    (Path("configs/pilot_game24.yaml"), "pilot_game24_fixture_v1"),
    (Path("configs/pilot_multitask_replay.yaml"), "pilot_multitask_replay_fixture_v1"),
    (Path("configs/g0_rag_bot_faithful_replay.yaml"), "g0_rag_bot_faithful_v1"),
    (Path("configs/g0_fh_reflexion_dc_faithful_replay.yaml"), "g0_fh_reflexion_dc_faithful_v1"),
    (
        Path("configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml"),
        "g0_dc_rs_reflexion_fidelity_followup_v1",
    ),
]
CONTRACT_CONFIG_PATH = Path("configs/logging_contract_replay.yaml")
CONTRACT_SAMPLE_IDS = ["game24_pilot_001", "meb_pilot_001", "word_sorting_pilot_001"]
CONTRACT_EXPECTED_STAGES = [
    "no_memory_generate",
    "rag_generate",
    "full_history_generate",
    "reflexion_generate",
    "reflexion_reflect",
    "bot_problem_distill",
    "bot_instantiate_solve",
    "bot_thought_distill",
    "bot_novelty_decide",
]
EXPECTED_STAGES = [
    "rag_generate",
    "bot_problem_distill",
    "bot_instantiate_solve",
    "bot_thought_distill",
    "bot_novelty_decide",
]

V0_5_CONFIG_PATH = Path("configs/g0_fh_reflexion_dc_faithful_replay.yaml")
V0_5_FIXTURE_PATH = Path("data/replay/g0_fh_reflexion_dc_faithful_v1.yaml")
V0_5_EXPECTED_STAGES = [
    "full_history_generate",
    "reflexion_generate",
    "reflexion_reflect",
    "dynamic_cheatsheet_generate",
    "dynamic_cheatsheet_curate",
]

FOLLOWUP_CONFIG_PATH = Path("configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml")
FOLLOWUP_FIXTURE_PATH = Path("data/replay/g0_dc_rs_reflexion_fidelity_followup_v1.yaml")
FOLLOWUP_EXPECTED_STAGES = [
    "dc_rs_synthesize",
    "dc_rs_generate",
    "reflexion_generate",
    "reflexion_reflect",
]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _expected_valid_combinations(config: dict) -> int:
    valid_pairs = 0
    for baseline in config["baselines"]:
        valid_pairs += 1 if baseline == "no_memory" else len(config["arms"])
    return len(config["tasks"]) * valid_pairs


@pytest.mark.parametrize("config_path,fixture_version", VERSIONED_CONFIGS)
def test_replay_configs_are_explicitly_versioned(config_path: Path, fixture_version: str) -> None:
    config = _load_yaml(config_path)

    assert config["run"]["stage"] == "replay"
    assert config["run"]["provider"] == "replay"
    assert config["run"]["model_snapshots"] == {model: fixture_version for model in config["models"]}
    assert config["logging"]["schema_version"] == "logging_v1"
    assert config["replay"]["fixture_version"] == fixture_version


def test_faithful_replay_fixture_covers_all_stages() -> None:
    config = _load_yaml(CONFIG_PATH)
    fixture = _load_yaml(FIXTURE_PATH)

    assert config["replay"]["responses_by_sample"] == fixture["responses_by_sample"]
    assert set(fixture["responses_by_sample"].keys()) == {
        "game24_pilot_001",
        "game24_pilot_002",
        "game24_pilot_003",
        "meb_pilot_001",
        "meb_pilot_002",
        "meb_pilot_003",
        "word_sorting_pilot_001",
        "word_sorting_pilot_002",
        "word_sorting_pilot_003",
    }

    for sample_id, stages in fixture["responses_by_sample"].items():
        assert list(stages) == EXPECTED_STAGES, sample_id
        assert all(stages[stage] for stage in EXPECTED_STAGES)


def test_logging_contract_replay_fixture_is_offline_and_expands_to_39_combinations() -> None:
    config = _load_yaml(CONTRACT_CONFIG_PATH)
    fixture = config["replay"]["responses_by_sample"]

    assert config["run"]["mode"] == "faithful"
    assert config["run"]["stage"] == "replay"
    assert config["run"]["provider"] == "replay"
    assert config["run"]["model_snapshots"] == {"replay_logging_contract": "logging_contract_fixture_v1"}
    assert config["models"] == ["replay_logging_contract"]
    assert config["logging"]["schema_version"] == "logging_v1"
    assert config["replay"]["fixture_version"] == "logging_contract_fixture_v1"
    assert config["embedding"]["offline_fallback"] is True
    assert config["live_smoke"]["enabled"] is False
    assert set(fixture) == set(CONTRACT_SAMPLE_IDS)
    assert _expected_valid_combinations(config) == 39

    for sample_id in CONTRACT_SAMPLE_IDS:
        stages = fixture[sample_id]
        assert list(stages) == CONTRACT_EXPECTED_STAGES, sample_id
        assert all(stages[stage] for stage in CONTRACT_EXPECTED_STAGES)

    retry = fixture["game24_pilot_001"]["reflexion_generate"]
    assert isinstance(retry, list)
    assert len(retry) == 2
    assert "1 + 1 + 1 + 1" in retry[0]
    assert "6 / (1 - (3 / 4))" in retry[1]


@pytest.mark.parametrize("stage", ["bot_problem_distill", "bot_thought_distill", "bot_novelty_decide"])
def test_faithful_replay_rejects_internal_answer_leakage(stage: str) -> None:
    fixture = _load_yaml(FIXTURE_PATH)
    leaked = [
        sample_id
        for sample_id, stages in fixture["responses_by_sample"].items()
        if "final:" in stages[stage].lower()
    ]

    assert leaked == []


def test_v0_5_config_matches_fixture() -> None:
    config = _load_yaml(V0_5_CONFIG_PATH)
    fixture = _load_yaml(V0_5_FIXTURE_PATH)

    assert config["replay"]["responses_by_sample"] == fixture["responses_by_sample"]


def test_v0_5_replay_fixture_covers_all_stages() -> None:
    fixture = _load_yaml(V0_5_FIXTURE_PATH)

    assert set(fixture["responses_by_sample"].keys()) == {
        "game24_pilot_001",
        "game24_pilot_002",
        "game24_pilot_003",
        "meb_pilot_001",
        "meb_pilot_002",
        "meb_pilot_003",
        "word_sorting_pilot_001",
        "word_sorting_pilot_002",
        "word_sorting_pilot_003",
    }

    for sample_id, stages in fixture["responses_by_sample"].items():
        assert list(stages) == V0_5_EXPECTED_STAGES, sample_id
        assert all(stages[stage] for stage in V0_5_EXPECTED_STAGES)


def test_v0_5_only_game24_pilot_001_has_invalid_reflexion_generate() -> None:
    fixture = _load_yaml(V0_5_FIXTURE_PATH)
    invalid = [
        sample_id
        for sample_id, stages in fixture["responses_by_sample"].items()
        if "final: 1 + 1 + 1 + 1" in stages["reflexion_generate"]
    ]
    assert invalid == ["game24_pilot_001"]


def test_v0_5_only_game24_pilot_001_lacks_cheatsheet_tag() -> None:
    fixture = _load_yaml(V0_5_FIXTURE_PATH)
    missing_tag = [
        sample_id
        for sample_id, stages in fixture["responses_by_sample"].items()
        if "<cheatsheet>" not in stages["dynamic_cheatsheet_curate"]
    ]
    assert missing_tag == ["game24_pilot_001"]


@pytest.mark.parametrize("stage", ["reflexion_reflect", "dynamic_cheatsheet_curate"])
def test_v0_5_replay_rejects_internal_answer_leakage(stage: str) -> None:
    fixture = _load_yaml(V0_5_FIXTURE_PATH)
    leaked = [
        sample_id
        for sample_id, stages in fixture["responses_by_sample"].items()
        if "final:" in stages[stage].lower()
    ]

    assert leaked == []


def test_followup_config_matches_fixture() -> None:
    config = _load_yaml(FOLLOWUP_CONFIG_PATH)
    fixture = _load_yaml(FOLLOWUP_FIXTURE_PATH)

    assert config["replay"]["responses_by_sample"] == fixture["responses_by_sample"]


def test_followup_replay_fixture_covers_all_stages() -> None:
    fixture = _load_yaml(FOLLOWUP_FIXTURE_PATH)

    assert set(fixture["responses_by_sample"].keys()) == {
        "game24_pilot_001",
        "game24_pilot_002",
        "game24_pilot_003",
        "meb_pilot_001",
        "meb_pilot_002",
        "meb_pilot_003",
        "word_sorting_pilot_001",
        "word_sorting_pilot_002",
        "word_sorting_pilot_003",
    }

    for sample_id, stages in fixture["responses_by_sample"].items():
        expected = list(FOLLOWUP_EXPECTED_STAGES)
        if sample_id != "game24_pilot_001":
            expected.remove("reflexion_reflect")
        assert list(stages) == expected, sample_id
        assert all(stages[stage] for stage in expected)


def test_followup_only_game24_pilot_001_has_reflexion_retry() -> None:
    fixture = _load_yaml(FOLLOWUP_FIXTURE_PATH)
    retry_samples = [
        sample_id
        for sample_id, stages in fixture["responses_by_sample"].items()
        if isinstance(stages["reflexion_generate"], list)
    ]
    assert retry_samples == ["game24_pilot_001"]


def test_followup_dc_rs_synthesize_contains_cheatsheet_tag() -> None:
    fixture = _load_yaml(FOLLOWUP_FIXTURE_PATH)
    missing = [
        sample_id
        for sample_id, stages in fixture["responses_by_sample"].items()
        if "<cheatsheet>" not in stages["dc_rs_synthesize"]
    ]
    assert missing == []


@pytest.mark.parametrize("stage", ["reflexion_reflect", "dc_rs_synthesize"])
def test_followup_replay_rejects_internal_answer_leakage(stage: str) -> None:
    fixture = _load_yaml(FOLLOWUP_FIXTURE_PATH)
    leaked = [
        sample_id
        for sample_id, stages in fixture["responses_by_sample"].items()
        if stage in stages and "final:" in stages[stage].lower()
    ]

    assert leaked == []


def test_followup_reflexion_retry_has_wrong_then_correct_generate() -> None:
    fixture = _load_yaml(FOLLOWUP_FIXTURE_PATH)
    stages = fixture["responses_by_sample"]["game24_pilot_001"]
    assert isinstance(stages["reflexion_generate"], list)
    assert len(stages["reflexion_generate"]) == 2
    assert "1 + 1 + 1 + 1" in stages["reflexion_generate"][0]
    assert "6 / (1 - (3 / 4))" in stages["reflexion_generate"][1]
