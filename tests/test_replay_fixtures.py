from __future__ import annotations

from pathlib import Path

import pytest
import yaml


CONFIG_PATH = Path("configs/g0_rag_bot_faithful_replay.yaml")
FIXTURE_PATH = Path("data/replay/g0_rag_bot_faithful_v1.yaml")
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


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
