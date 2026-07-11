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
