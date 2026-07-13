from __future__ import annotations

from pathlib import Path

import pytest

from memcontam.contamination.catalog import load_catalog
from memcontam.memory.corpus import KNOWN_BASELINES, load_corpus


def _catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data/contamination/catalog_v0.jsonl"


def _memory_corpus_v2_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data/memory/catalog_v2.jsonl"


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


@pytest.fixture(scope="module")
def v2_corpus():
    return load_corpus(_memory_corpus_v2_path())


def test_v2_corpus_has_eighteen_records_and_nine_pairs(v2_corpus) -> None:
    assert len(v2_corpus) == 18

    clean = [r for r in v2_corpus if r.clean_or_contaminated == "clean"]
    corrupted = [r for r in v2_corpus if r.clean_or_contaminated == "contaminated"]
    assert len(clean) == 9
    assert len(corrupted) == 9

    clean_ids = {r.entry_id for r in clean}
    for r in corrupted:
        assert r.paired_clean_entry_id in clean_ids


def test_v2_corpus_ids_are_unique(v2_corpus) -> None:
    ids = [r.entry_id for r in v2_corpus]
    assert len(ids) == len(set(ids))


def test_v2_corpus_has_one_target_baseline_per_record(v2_corpus) -> None:
    for r in v2_corpus:
        assert len(r.target_baselines) == 1
        assert r.target_baselines[0] in KNOWN_BASELINES


@pytest.mark.parametrize(
    "baseline,memory_type",
    [
        ("full_history", "full_history_transcript"),
        ("reflexion_style", "verbal_reflection"),
        ("dynamic_cheatsheet_optional", "cheatsheet_item"),
    ],
)
def test_v2_corpus_memory_type_matches_baseline(v2_corpus, baseline: str, memory_type: str) -> None:
    for r in v2_corpus:
        if r.target_baselines == [baseline]:
            assert r.memory_type == memory_type


def test_v2_corpus_memory_types_are_native_baseline_categories(v2_corpus) -> None:
    for r in v2_corpus:
        assert r.memory_type in {
            "full_history_transcript",
            "verbal_reflection",
            "cheatsheet_item",
        }


def test_v2_corpus_contaminated_pairs_match_clean_record(v2_corpus) -> None:
    clean_by_id = {r.entry_id: r for r in v2_corpus if r.clean_or_contaminated == "clean"}
    for r in v2_corpus:
        if r.clean_or_contaminated != "contaminated":
            continue
        clean = clean_by_id[r.paired_clean_entry_id]
        assert clean.task == r.task
        assert clean.memory_type == r.memory_type
        assert clean.target_baselines == r.target_baselines


def test_v2_corpus_clean_payload_matches_task_strategy(v2_corpus) -> None:
    v1 = load_corpus(Path(__file__).resolve().parents[1] / "data/memory/catalog_v1.jsonl")
    v1_clean_by_task = {
        r.task: r.content for r in v1 if r.clean_or_contaminated == "clean"
    }
    for r in v2_corpus:
        if r.clean_or_contaminated == "clean":
            assert r.content == v1_clean_by_task[r.task]


def test_v2_corpus_contaminated_payload_does_not_leak_answers(v2_corpus) -> None:
    for r in v2_corpus:
        if r.clean_or_contaminated == "contaminated":
            assert "final:" not in r.content.lower()
