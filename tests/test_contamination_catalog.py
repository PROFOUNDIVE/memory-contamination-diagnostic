from __future__ import annotations

from pathlib import Path

from memcontam.contamination.catalog import load_catalog


def _catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data/contamination/catalog_v0.jsonl"


def test_catalog_contains_only_locked_tasks() -> None:
    catalog = load_catalog(_catalog_path())
    tasks = {entry["task"] for entry in catalog}

    assert tasks <= {"game24", "word_sorting", "math_equation_balancer"}
    assert tasks == {"game24", "word_sorting", "math_equation_balancer"}


def test_catalog_baseline_filtering_keeps_only_targeted_entries() -> None:
    catalog = load_catalog(_catalog_path())

    for baseline in ["retrieval_rag", "bot_style", "no_memory"]:
        filtered = [entry for entry in catalog if baseline in entry.get("target_baselines", [])]

        if baseline == "no_memory":
            assert filtered == []


def test_catalog_entries_have_required_fields() -> None:
    catalog = load_catalog(_catalog_path())

    for entry in catalog:
        assert {"entry_id", "task", "type", "target_baselines", "content"} <= entry.keys()
