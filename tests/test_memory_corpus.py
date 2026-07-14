from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from memcontam.memory.corpus import (
    CorpusValidationError,
    build_arm_corpus,
    load_corpus,
)
from memcontam.memory.stores import MemoryEntry


LOCKED_TASKS = {"game24", "math_equation_balancer", "word_sorting"}


def _catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data/memory/catalog_v1.jsonl"


def _content_hash(entries: list[MemoryEntry]) -> str:
    payload = json.dumps(
        [entry.model_dump() for entry in entries],
        sort_keys=True,
        separators=(",", ":"),
    )
    from hashlib import sha256

    return sha256(payload.encode("utf-8")).hexdigest()


def test_clean_and_contaminated_arms_share_identical_clean_records() -> None:
    records = load_corpus(_catalog_path())
    tasks = {record.task for record in records}

    assert LOCKED_TASKS <= tasks, f"missing locked tasks: {LOCKED_TASKS - tasks}"

    for task in sorted(LOCKED_TASKS):
        clean_entries, _ = build_arm_corpus(records, task, "clean")
        contaminated_entries, _ = build_arm_corpus(records, task, "contaminated")
        filtered_entries, _ = build_arm_corpus(records, task, "contaminated_filter")

        assert all(
            entry.clean_or_contaminated == "clean" for entry in clean_entries
        ), f"clean arm for {task} contains non-clean records"

        clean_ids = {entry.entry_id for entry in clean_entries}
        contaminated_ids = {entry.entry_id for entry in contaminated_entries}
        assert clean_ids <= contaminated_ids, (
            f"clean record IDs for {task} are not a subset of contaminated IDs"
        )

        clean_by_id = {entry.entry_id: entry for entry in clean_entries}
        contaminated_by_id = {entry.entry_id: entry for entry in contaminated_entries}
        for entry_id in clean_ids:
            clean_entry = clean_by_id[entry_id]
            contaminated_entry = contaminated_by_id[entry_id]
            assert clean_entry.content == contaminated_entry.content, (
                f"content drift for {entry_id} between clean and contaminated arms"
            )
            assert clean_entry.model_dump() == contaminated_entry.model_dump(), (
                f"byte-identical fields diverge for {entry_id}"
            )

        added_ids = contaminated_ids - clean_ids
        assert all(
            contaminated_by_id[entry_id].clean_or_contaminated == "contaminated"
            for entry_id in added_ids
        ), f"contaminated arm for {task} added non-corrupted records"

        assert len(filtered_entries) <= len(contaminated_entries), (
            f"filter arm for {task} grew the corpus"
        )
        assert all(
            entry.clean_or_contaminated != "contaminated" for entry in filtered_entries
        ), f"filter arm for {task} still contains contaminated records"

        assert _content_hash(clean_entries) == _content_hash(clean_entries), (
            f"clean arm hash for {task} is not stable"
        )


def _write_fixture(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "catalog.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _valid_clean_record(entry_id: str, task: str = "game24") -> dict:
    return {
        "entry_id": entry_id,
        "task": task,
        "target_baselines": ["retrieval_rag"],
        "memory_type": "strategy",
        "content": "Break the problem into smaller sub-expressions and check each step.",
        "source": "pilot_warmup",
        "clean_or_contaminated": "clean",
        "paired_clean_entry_id": None,
    }


def _valid_dc_rs_record(
    entry_id: str,
    task: str = "game24",
    content: str = '{"numbers":[1,2,3,3],"target":9}',
    output_text: str = "1 + 2 + 3 + 3",
    paired_clean_entry_id: str | None = None,
) -> dict:
    return {
        "entry_id": entry_id,
        "task": task,
        "target_baselines": ["dynamic_cheatsheet_rs_optional"],
        "memory_type": "dc_rs_io_pair",
        "content": content,
        "output_text": output_text,
        "source": "pilot_warmup_dc_rs",
        "clean_or_contaminated": "clean" if paired_clean_entry_id is None else "contaminated",
        "paired_clean_entry_id": paired_clean_entry_id,
    }


def test_corpus_rejects_invalid_or_answer_leaking_records() -> None:
    base = [
        _valid_clean_record("clean_game24_001"),
        {
            "entry_id": "clean_game24_001",
            "task": "game24",
            "target_baselines": ["retrieval_rag"],
            "memory_type": "strategy",
            "content": "Duplicate ID record.",
            "source": "pilot_warmup",
            "clean_or_contaminated": "clean",
            "paired_clean_entry_id": None,
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        duplicate_path = _write_fixture(tmp_path, base)
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(duplicate_path)
        assert "clean_game24_001" in str(exc.value)

        unknown_task = [
            _valid_clean_record("clean_game24_002"),
            {
                **_valid_clean_record("clean_unknown_001"),
                "task": "unknown_task",
                "entry_id": "clean_unknown_001",
            },
        ]
        unknown_task_path = _write_fixture(tmp_path, unknown_task)
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(unknown_task_path)
        assert "unknown_task" in str(exc.value)
        assert "clean_unknown_001" in str(exc.value)

        unknown_baseline = [
            {
                **_valid_clean_record("clean_game24_003"),
                "target_baselines": ["retrieval_rag", "unsupported_baseline"],
            }
        ]
        unknown_baseline_path = _write_fixture(tmp_path, unknown_baseline)
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(unknown_baseline_path)
        assert "unsupported_baseline" in str(exc.value)
        assert "clean_game24_003" in str(exc.value)

        missing_source = [
            {
                **_valid_clean_record("clean_game24_004"),
                "source": "",
            }
        ]
        missing_source_path = _write_fixture(tmp_path, missing_source)
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(missing_source_path)
        assert "clean_game24_004" in str(exc.value)
        assert "source" in str(exc.value).lower()

        answer_marker = [
            {
                **_valid_clean_record("clean_game24_005"),
                "content": "The answer is final: 24 because 6 / (1 - 3/4).",
            }
        ]
        answer_marker_path = _write_fixture(tmp_path, answer_marker)
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(answer_marker_path)
        assert "clean_game24_005" in str(exc.value)

        raw_answer = [
            {
                **_valid_clean_record("clean_meb_005", task="math_equation_balancer"),
                "content": "For similar inputs, the balanced equation is 2 + 5 = 7.",
            }
        ]
        raw_answer_path = _write_fixture(tmp_path, raw_answer)
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(raw_answer_path)
        assert "clean_meb_005" in str(exc.value)

        corrupted_without_pair = [
            _valid_clean_record("clean_game24_006"),
            {
                "entry_id": "corrupted_game24_001",
                "task": "game24",
                "target_baselines": ["retrieval_rag"],
                "memory_type": "wrong_rule",
                "content": "A misleading rule that does not help.",
                "source": "injected",
                "clean_or_contaminated": "contaminated",
                "paired_clean_entry_id": "missing_clean_id",
            },
        ]
        corrupted_without_pair_path = _write_fixture(tmp_path, corrupted_without_pair)
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(corrupted_without_pair_path)
        assert "corrupted_game24_001" in str(exc.value)
        assert "missing_clean_id" in str(exc.value)


def test_dc_rs_io_pair_loads_and_propagates_output_text_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = _write_fixture(tmp_path, [_valid_dc_rs_record("dc_rs_clean_game24_001")])
        records = load_corpus(path)
        assert len(records) == 1
        record = records[0]
        assert record.memory_type == "dc_rs_io_pair"
        assert record.content == '{"numbers":[1,2,3,3],"target":9}'
        assert record.output_text == "1 + 2 + 3 + 3"

        entries, _ = build_arm_corpus(records, "game24", "clean")
        assert len(entries) == 1
        entry = entries[0]
        assert entry.content == record.content
        assert entry.metadata.get("output_text") == "1 + 2 + 3 + 3"
        assert "task" in entry.metadata
        assert "source" in entry.metadata
        assert "target_baselines" in entry.metadata


def test_dc_rs_io_pair_rejects_missing_output_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        row = _valid_dc_rs_record("dc_rs_clean_game24_002")
        del row["output_text"]
        path = _write_fixture(tmp_path, [row])
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(path)
        assert "dc_rs_clean_game24_002" in str(exc.value)
        assert "output_text" in str(exc.value).lower()


def test_dc_rs_io_pair_rejects_blank_output_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for blank in ["", "   ", "\t"]:
            row = _valid_dc_rs_record("dc_rs_clean_game24_003", output_text=blank)
            path = _write_fixture(tmp_path, [row])
            with pytest.raises(CorpusValidationError) as exc:
                load_corpus(path)
            assert "dc_rs_clean_game24_003" in str(exc.value)
            assert "output_text" in str(exc.value).lower()


def test_dc_rs_io_pair_rejects_blank_content() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        row = _valid_dc_rs_record("dc_rs_clean_game24_004")
        row["content"] = ""
        path = _write_fixture(tmp_path, [row])
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(path)
        assert "dc_rs_clean_game24_004" in str(exc.value)
        assert "content" in str(exc.value).lower()


def test_dc_rs_io_pair_rejects_leaking_output_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        row = _valid_dc_rs_record(
            "dc_rs_clean_game24_005",
            output_text="The answer is final: 24 because 6 / (1 - 3/4).",
        )
        path = _write_fixture(tmp_path, [row])
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(path)
        assert "dc_rs_clean_game24_005" in str(exc.value)
        assert "final:" in str(exc.value)


def test_non_dc_rs_records_parse_without_output_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = _write_fixture(tmp_path, [_valid_clean_record("clean_game24_007")])
        records = load_corpus(path)
        assert len(records) == 1
        assert records[0].output_text is None

        entries, _ = build_arm_corpus(records, "game24", "clean")
        assert "output_text" not in entries[0].metadata


@pytest.mark.parametrize(
    "task,content,clean_id,corrupted_id,clean_output,corrupted_output,expected_filter_dropped",
    [
        (
            "game24",
            '{"numbers":[1,2,3,3],"target":9}',
            "dc_rs_clean_game24_001",
            "dc_rs_corrupted_game24_001",
            "1 + 2 + 3 + 3",
            "1 * 2 + 3 + 3",
            1,
        ),
        (
            "math_equation_balancer",
            '{"input":"7 + 8 = ?"}',
            "dc_rs_clean_meb_001",
            "dc_rs_corrupted_meb_001",
            "7 + 8 = 15",
            "7 + 8 = 14",
            1,
        ),
        (
            "word_sorting",
            '{"words":["kiwi","apple","mango"]}',
            "dc_rs_clean_word_sorting_001",
            "dc_rs_corrupted_word_sorting_001",
            '["apple","kiwi","mango"]',
            '["mango","apple","kiwi"]',
            1,
        ),
    ],
)
def test_dc_rs_arm_construction_keeps_clean_pair_and_filters_corruption(
    task: str,
    content: str,
    clean_id: str,
    corrupted_id: str,
    clean_output: str,
    corrupted_output: str,
    expected_filter_dropped: int,
) -> None:
    rows = [
        _valid_dc_rs_record(clean_id, task=task, content=content, output_text=clean_output),
        _valid_dc_rs_record(
            corrupted_id,
            task=task,
            content=content,
            output_text=corrupted_output,
            paired_clean_entry_id=clean_id,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = _write_fixture(tmp_path, rows)
        records = load_corpus(path)

        clean_entries, _ = build_arm_corpus(records, task, "clean")
        contaminated_entries, _ = build_arm_corpus(records, task, "contaminated")
        filtered_entries, filtered_meta = build_arm_corpus(records, task, "contaminated_filter")

        assert [entry.entry_id for entry in clean_entries] == [clean_id]
        assert [entry.entry_id for entry in contaminated_entries] == sorted([clean_id, corrupted_id])
        assert [entry.entry_id for entry in filtered_entries] == [clean_id]
        assert filtered_meta == {"filter": "drop_known_contaminated", "dropped": expected_filter_dropped}
        assert contaminated_entries[0].content == content
        assert contaminated_entries[1].content == content
        assert contaminated_entries[0].metadata.get("output_text") == clean_output
        assert contaminated_entries[1].metadata.get("output_text") == corrupted_output
