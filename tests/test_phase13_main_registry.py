from __future__ import annotations

import hashlib
from pathlib import Path

from memcontam.main_registry import freeze_task_pool
from scripts.build_phase13_main_registries import build


def test_freeze_task_pool_excludes_registered_and_duplicate_game24_signatures() -> None:
    rows = (
        {"input": "2 5 8 11", "target": "24"},
        {"input": "11 8 5 2", "target": "24"},
        {"input": "3 3 8 8", "target": "24"},
    )

    registry = freeze_task_pool(
        task="game24",
        rows=rows,
        excluded_signatures=frozenset({"3,3,8,8"}),
    )

    assert [row.sample_id for row in registry.rows] == ["phase13_main_game24_0001"]
    assert registry.rows[0].canonical_signature == "2,5,8,11"
    assert [(item.source_row, item.reason) for item in registry.exclusions] == [
        (2, "duplicate_canonical_signature"),
        (3, "registered_non_main_signature"),
    ]


def test_freeze_task_pool_imports_meb_and_word_sorting_task_shapes() -> None:
    meb = freeze_task_pool(
        task="math_equation_balancer",
        rows=({"input": "14 ? 16 ? 20 ? 15 ? 23 = 25", "target": "14 - 16 / 20 * 15 + 23 = 25", "target_value": 25},),
        excluded_signatures=frozenset(),
    )
    words = freeze_task_pool(
        task="word_sorting",
        rows=({"input": "Sort the following words alphabetically: List: syndrome therefrom", "target": "syndrome therefrom"},),
        excluded_signatures=frozenset(),
    )

    assert meb.rows[0].model_dump(mode="json") == {
        "sample_id": "phase13_main_meb_0001",
        "input": "14 ? 16 ? 20 ? 15 ? 23 = 25",
        "verifier_spec": {
            "target": "14 - 16 / 20 * 15 + 23 = 25",
            "target_value": 25,
        },
        "canonical_signature": "14,16,20,15,23,25",
        "source_row": 1,
    }
    assert words.rows[0].model_dump(mode="json") == {
        "sample_id": "phase13_main_word_sorting_0001",
        "words": ["syndrome", "therefrom"],
        "sorted_words": ["syndrome", "therefrom"],
        "canonical_signature": "syndrome|therefrom",
        "source_row": 1,
    }


def test_approved_sources_rebuild_identical_main_registries(tmp_path: Path) -> None:
    output = tmp_path / "main"

    manifest = build(
        Path("/home/hyunwoo/git/buffer-of-thought-llm"),
        Path("/home/hyunwoo/git/dynamic-cheatsheet"),
        output,
    )

    registries = manifest["registries"]
    expected = {
        "game24": (95, "ae682f138d8035fc1de9382eb8903730d392851def720351a78846df160b615f"),
        "math_equation_balancer": (
            250,
            "dfa07c8c3ada1b0030a735cca97022f98dfb8da30d8ce86f82013eb51b4a7037",
        ),
        "word_sorting": (
            250,
            "e7ff0507512af4e71ae027a5226984b175d9b75dca898df79ca88535326c9c54",
        ),
    }
    for task, (count, digest) in expected.items():
        path = output / f"{task}_main_v1.jsonl"
        assert len(path.read_text(encoding="utf-8").splitlines()) == count
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert registries[task]["sha256"] == digest
