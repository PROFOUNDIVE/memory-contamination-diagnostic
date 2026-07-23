from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import memcontam.cli as cli


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
PHASE11_CONTRACT_CONFIG_PATH = Path("configs/logging_contract_phase11_replay.yaml")
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
    assert config["run"]["model_snapshots"] == {
        model: fixture_version for model in config["models"]
    }
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
    assert config["run"]["model_snapshots"] == {
        "replay_logging_contract": "logging_contract_fixture_v1"
    }
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


def test_phase11_logging_contract_replay_fixture_is_offline_and_expands_to_39_combinations() -> (
    None
):
    config = _load_yaml(PHASE11_CONTRACT_CONFIG_PATH)
    fixture = config["replay"]["responses_by_sample"]

    assert config["run"]["mode"] == "faithful"
    assert config["run"]["stage"] == "replay"
    assert config["run"]["provider"] == "replay"
    assert config["run"]["contract_level"] == "phase11"
    assert config["run"]["model_snapshots"] == {
        "replay_logging_contract_phase11": "logging_contract_phase11_fixture_v1"
    }
    assert config["models"] == ["replay_logging_contract_phase11"]
    assert config["logging"]["schema_version"] == "logging_v2"
    assert config["replay"]["fixture_version"] == "logging_contract_phase11_fixture_v1"
    assert config["memory"]["corpus_path"] == "data/memory/catalog_v3.jsonl"
    assert config["memory"]["corpus_version"] == "memory_catalog_v3"
    assert config["embedding"]["offline_fallback"] is True
    assert config["live_smoke"]["enabled"] is False
    assert config["evaluation"] == {
        "evaluation_law_id": "phase11_logging_contract_online_replay_v1",
        "regime": "online",
        "task_law_id": "locked_three_tasks_limit1_v1",
        "inference_law_id": "logging_contract_phase11_replay_fixture_v1",
        "checkpoint_policy_id": None,
    }
    assert config["target_contamination_set"] == {
        "target_set_id": "controlled_injected_derived_v1",
        "definition_version": "phase11_v1",
        "included_classes": ["injected", "derived"],
        "require_exact_lineage": True,
    }
    assert set(fixture) == set(CONTRACT_SAMPLE_IDS)
    assert _expected_valid_combinations(config) == 39

    for sample_id in CONTRACT_SAMPLE_IDS:
        stages = fixture[sample_id]
        assert list(stages) == CONTRACT_EXPECTED_STAGES, sample_id
        assert all(stages[stage] for stage in CONTRACT_EXPECTED_STAGES)


@pytest.mark.parametrize(
    "section,expected",
    [
        ("evaluation", "logging_v2 requires evaluation"),
        ("target_contamination_set", "logging_v2 requires target_contamination_set"),
    ],
)
def test_phase11_config_validation_fails_closed_when_typed_sections_are_missing(
    tmp_path: Path, section: str, expected: str
) -> None:
    config = _load_yaml(PHASE11_CONTRACT_CONFIG_PATH)
    config.pop(section)
    path = tmp_path / f"missing_{section}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(SystemExit, match=expected):
        cli.validate_config(path)


@pytest.mark.parametrize(
    "section,key,value,expected",
    [
        ("evaluation", "regime", "later", "evaluation.regime"),
        (
            "target_contamination_set",
            "included_classes",
            ["injected", "bogus"],
            "target_contamination_set.included_classes",
        ),
    ],
)
def test_phase11_config_validation_fails_closed_for_unknown_typed_values(
    tmp_path: Path, section: str, key: str, value: object, expected: str
) -> None:
    config = _load_yaml(PHASE11_CONTRACT_CONFIG_PATH)
    config[section][key] = value
    path = tmp_path / f"bad_{section}_{key}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(SystemExit, match=expected):
        cli.validate_config(path)


@pytest.mark.parametrize(
    "stage", ["bot_problem_distill", "bot_thought_distill", "bot_novelty_decide"]
)
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


def test_followup_reflexion_reflection_uses_the_strict_corrective_schema() -> None:
    fixture = _load_yaml(FOLLOWUP_FIXTURE_PATH)

    for stages in fixture["responses_by_sample"].values():
        payload_text = stages.get("reflexion_reflect")
        if payload_text is None:
            continue
        payload = json.loads(payload_text)
        assert payload == {
            "mode": "corrective",
            "failure_class": "incorrect_answer",
            "reflection_text": payload["reflection_text"],
            "explicitly_used_memory_ids": [],
        }
