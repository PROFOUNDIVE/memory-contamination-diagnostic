from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from memcontam.contamination.catalog import load_catalog
from memcontam.memory.corpus import KNOWN_BASELINES, load_corpus
from memcontam.verifiers.game24 import verify_expression
from memcontam.verifiers.math_equation_balancer import verify_answer
from memcontam.verifiers.word_sorting import verify_words


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


def test_v2_corpus_has_twenty_four_records_and_twelve_pairs(v2_corpus) -> None:
    assert len(v2_corpus) == 24

    clean = [r for r in v2_corpus if r.clean_or_contaminated == "clean"]
    corrupted = [r for r in v2_corpus if r.clean_or_contaminated == "contaminated"]
    assert len(clean) == 12
    assert len(corrupted) == 12

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
        ("dynamic_cheatsheet_rs_optional", "dc_rs_io_pair"),
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
            "dc_rs_io_pair",
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
    v1_clean_by_task = {r.task: r.content for r in v1 if r.clean_or_contaminated == "clean"}
    dc_rs_clean_by_task = {
        "game24": '{"numbers":[1,2,3,3],"target":9}',
        "math_equation_balancer": '{"input":"7 + 8 = ?"}',
        "word_sorting": '{"words":["kiwi","apple","mango"]}',
    }
    for r in v2_corpus:
        if r.clean_or_contaminated == "clean":
            if r.target_baselines == ["dynamic_cheatsheet_rs_optional"]:
                assert r.content == dc_rs_clean_by_task[r.task]
            else:
                assert r.content == v1_clean_by_task[r.task]


def test_v2_corpus_contaminated_payload_does_not_leak_answers(v2_corpus) -> None:
    for r in v2_corpus:
        if r.clean_or_contaminated == "contaminated":
            assert "final:" not in r.content.lower()


def test_dc_rs_io_pair_schema_accepted() -> None:
    row = {
        "entry_id": "dc_rs_clean_game24_schema_001",
        "task": "game24",
        "target_baselines": ["dynamic_cheatsheet_rs_optional"],
        "memory_type": "dc_rs_io_pair",
        "content": '{"numbers":[1,2,3,3],"target":9}',
        "output_text": "1 + 2 + 3 + 3",
        "source": "pilot_warmup_dc_rs",
        "clean_or_contaminated": "clean",
        "paired_clean_entry_id": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "catalog.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        records = load_corpus(path)
        assert len(records) == 1
        record = records[0]
        assert record.memory_type == "dc_rs_io_pair"
        assert record.output_text == "1 + 2 + 3 + 3"
        assert "dynamic_cheatsheet_rs_optional" in KNOWN_BASELINES


def test_dc_rs_warmup_rows_are_disjoint_and_verifier_checked(v2_corpus) -> None:
    locked_inputs = {entry["content"] for entry in load_catalog(_catalog_path())}
    cases = [
        {
            "task": "game24",
            "clean_id": "dc_rs_clean_game24_001",
            "corrupted_id": "dc_rs_corrupted_game24_001",
            "content": '{"numbers":[1,2,3,3],"target":9}',
            "clean_output": "1 + 2 + 3 + 3",
            "corrupted_output": "1 * 2 + 3 + 3",
            "clean_check": lambda output, spec: verify_expression(
                output, spec["numbers"], target=spec["target"]
            ),
            "input_spec": {"numbers": [1, 2, 3, 3], "target": 9},
        },
        {
            "task": "math_equation_balancer",
            "clean_id": "dc_rs_clean_meb_001",
            "corrupted_id": "dc_rs_corrupted_meb_001",
            "content": '{"input":"7 + 8 = ?"}',
            "clean_output": "7 + 8 = 15",
            "corrupted_output": "7 + 8 = 14",
            "clean_check": lambda output, spec: verify_answer(output, spec),
            "input_spec": {"target": "7 + 8 = 15", "target_value": "15"},
        },
        {
            "task": "word_sorting",
            "clean_id": "dc_rs_clean_word_sorting_001",
            "corrupted_id": "dc_rs_corrupted_word_sorting_001",
            "content": '{"words":["kiwi","apple","mango"]}',
            "clean_output": '["apple","kiwi","mango"]',
            "corrupted_output": '["mango","apple","kiwi"]',
            "clean_check": lambda output, spec: verify_words(json.loads(output), spec),
            "input_spec": ["apple", "kiwi", "mango"],
        },
    ]

    records_by_id = {record.entry_id: record for record in v2_corpus}

    for case in cases:
        clean = records_by_id[case["clean_id"]]
        corrupted = records_by_id[case["corrupted_id"]]

        assert clean.memory_type == "dc_rs_io_pair"
        assert corrupted.memory_type == clean.memory_type
        assert clean.target_baselines == ["dynamic_cheatsheet_rs_optional"]
        assert corrupted.target_baselines == clean.target_baselines
        assert corrupted.paired_clean_entry_id == clean.entry_id
        assert clean.content not in locked_inputs
        assert corrupted.content not in locked_inputs

        clean_result = case["clean_check"](clean.output_text, case["input_spec"])
        corrupted_result = case["clean_check"](corrupted.output_text, case["input_spec"])

        assert clean_result.is_correct is True
        assert corrupted_result.is_correct is False

        if case["task"] == "game24":
            assert clean_result.reason == "ok"
            assert corrupted_result.reason == "value_does_not_match_target"
        elif case["task"] == "math_equation_balancer":
            assert clean_result.reason == "ok"
            assert corrupted_result.reason == "wrong_answer"
        else:
            assert clean_result.reason == "ok"
            assert corrupted_result.reason == "wrong_order"
