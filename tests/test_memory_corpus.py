from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from hashlib import sha256

from memcontam.memory.corpus import (
    CorpusValidationError,
    build_arm_corpus,
    load_corpus,
)
from memcontam.memory.oracle_qa import drop_known_contaminated
from memcontam.memory.stores import MemoryEntry


LOCKED_TASKS = {"game24", "math_equation_balancer", "word_sorting"}


def _catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data/memory/catalog_v1.jsonl"


def _phase11_catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data/memory/catalog_v3.jsonl"


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

        assert all(entry.clean_or_contaminated == "clean" for entry in clean_entries), (
            f"clean arm for {task} contains non-clean records"
        )

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
        assert all(entry.clean_or_contaminated != "contaminated" for entry in filtered_entries), (
            f"filter arm for {task} still contains contaminated records"
        )

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


def _with_seed_provenance(row: dict, contamination_class: str = "clean") -> dict:
    return {
        **row,
        "contamination_class": contamination_class,
        "lineage_status": "exact",
        "lineage_basis": "seed",
        "direct_parent_ids": [],
        "injected_root_ids": [row["entry_id"]] if contamination_class == "injected" else [],
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


def _valid_v2_dc_rs_record(
    entry_id: str,
    task: str = "game24",
    content: str = '{"input":{"numbers":[1,2,3,3],"target":9},"sample_id":"warmup","task_name":"game24"}',
    generated_output: str = "Strategy: preserve intermediate values. Code: enumerate candidates.",
    parsed_answer: str | None = "candidate expression",
) -> dict:
    return {
        "entry_id": entry_id,
        "task": task,
        "target_baselines": ["dynamic_cheatsheet_rs_optional"],
        "memory_type": "dc_rs_io_pair",
        "content": content,
        "generated_output": generated_output,
        "parsed_answer": parsed_answer,
        "source": "baseline-fidelity-v2-dc-rs-contract",
        "clean_or_contaminated": "clean",
        "paired_clean_entry_id": None,
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


def test_dc_rs_v2_pair_preserves_raw_generated_output_and_separate_parsed_answer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture(Path(tmp), [_valid_v2_dc_rs_record("dc_rs_v2_game24_001")])
        record = load_corpus(path)[0]
        entries, _ = build_arm_corpus([record], "game24", "clean")

    assert (
        record.generated_output
        == "Strategy: preserve intermediate values. Code: enumerate candidates."
    )
    assert record.parsed_answer == "candidate expression"
    assert record.output_text is None
    assert entries[0].metadata["generated_output"] == record.generated_output
    assert entries[0].metadata["parsed_answer"] == record.parsed_answer


def test_dc_rs_legacy_output_text_is_readable_but_rejected_as_v2_raw_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        record = load_corpus(
            _write_fixture(Path(tmp), [_valid_dc_rs_record("dc_rs_legacy_game24_001")])
        )[0]

    assert record.output_text == "1 + 2 + 3 + 3"
    assert record.generated_output is None
    with pytest.raises(CorpusValidationError, match="generated_output"):
        record.require_dc_rs_v2_evidence()


def test_non_dc_rs_records_parse_without_output_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = _write_fixture(tmp_path, [_valid_clean_record("clean_game24_007")])
        records = load_corpus(path)
        assert len(records) == 1
        assert records[0].output_text is None

        entries, _ = build_arm_corpus(records, "game24", "clean")
        assert "output_text" not in entries[0].metadata


def test_legacy_records_do_not_emit_phase11_metadata_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture(Path(tmp), [_valid_clean_record("clean_game24_008")])
        records = load_corpus(path)

        entries, _ = build_arm_corpus(records, "game24", "clean")
        metadata = entries[0].metadata

        for key in [
            "contamination_class",
            "lineage_status",
            "lineage_basis",
            "direct_parent_ids",
            "injected_root_ids",
        ]:
            assert key not in metadata


def test_phase11_catalog_v3_records_have_seed_provenance_and_metadata() -> None:
    records = load_corpus(_phase11_catalog_path())

    assert len(records) == 24
    for record in records:
        assert record.lineage_status == "exact"
        assert record.lineage_basis == "seed"
        assert record.direct_parent_ids == []
        if record.clean_or_contaminated == "clean":
            assert record.contamination_class == "clean"
            assert record.injected_root_ids == []
        else:
            assert record.contamination_class == "injected"
            assert record.injected_root_ids == [record.entry_id]

    entries, _ = build_arm_corpus(records, "game24", "contaminated")
    injected = next(entry for entry in entries if entry.clean_or_contaminated == "contaminated")
    assert injected.metadata["contamination_class"] == "injected"
    assert injected.metadata["lineage_status"] == "exact"
    assert injected.metadata["lineage_basis"] == "seed"
    assert injected.metadata["injected_root_ids"] == [injected.entry_id]
    assert injected.metadata["direct_parent_ids"] == []


def test_phase11_seed_provenance_rejects_injected_record_without_self_root() -> None:
    clean = _with_seed_provenance(_valid_clean_record("clean_game24_seed_001"))
    injected = _with_seed_provenance(
        {
            **_valid_clean_record("corrupted_game24_seed_001"),
            "content": "A misleading rule that does not help.",
            "source": "injected_corruption",
            "clean_or_contaminated": "contaminated",
            "paired_clean_entry_id": "clean_game24_seed_001",
        },
        "injected",
    )
    injected["injected_root_ids"] = []

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture(Path(tmp), [clean, injected])
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(path)
        assert "corrupted_game24_seed_001" in str(exc.value)
        assert "injected_root_ids" in str(exc.value)


def test_phase11_seed_provenance_rejects_clean_record_with_injected_roots() -> None:
    clean = _with_seed_provenance(_valid_clean_record("clean_game24_seed_002"))
    clean["injected_root_ids"] = ["some_injected_root"]

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture(Path(tmp), [clean])
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(path)
        assert "clean_game24_seed_002" in str(exc.value)
        assert "clean seed" in str(exc.value)


@pytest.mark.parametrize("contamination_class", ["derived", "natural"])
def test_phase11_seed_provenance_rejects_invalid_seed_classes(
    contamination_class: str,
) -> None:
    row = _with_seed_provenance(
        {
            **_valid_clean_record(f"bad_{contamination_class}_seed_001"),
            "clean_or_contaminated": "contaminated",
            "paired_clean_entry_id": "clean_game24_seed_003",
        },
        contamination_class,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = _write_fixture(
            Path(tmp), [_with_seed_provenance(_valid_clean_record("clean_game24_seed_003")), row]
        )
        with pytest.raises(CorpusValidationError) as exc:
            load_corpus(path)
        assert f"bad_{contamination_class}_seed_001" in str(exc.value)
        assert "seed" in str(exc.value)


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
        assert [entry.entry_id for entry in contaminated_entries] == sorted(
            [clean_id, corrupted_id]
        )
        assert [entry.entry_id for entry in filtered_entries] == [clean_id]
        assert filtered_meta is not None
        assert filtered_meta["filter_name"] == "drop_known_contaminated"
        assert filtered_meta["input_count"] == 2
        assert filtered_meta["kept_count"] == 1
        assert filtered_meta["removed_count"] == expected_filter_dropped
        assert filtered_meta["dropped"] == expected_filter_dropped
        assert len(filtered_meta["decisions"]) == 2
        assert filtered_meta["input_source_ids"] == sorted([clean_id, corrupted_id])
        assert filtered_meta["kept_source_ids"] == [clean_id]
        assert filtered_meta["removed_source_ids"] == [corrupted_id]
        removed_decision = next(
            decision
            for decision in filtered_meta["decisions"]
            if decision["entry_id"] == corrupted_id
        )
        assert removed_decision["ground_truth"] == "contaminated"
        assert removed_decision["action"] == "removed"
        assert removed_decision["reason"] == "known_contaminated"
        assert removed_decision["score"] is None
        assert contaminated_entries[0].content == content
        assert contaminated_entries[1].content == content
        assert contaminated_entries[0].metadata.get("output_text") == clean_output
        assert contaminated_entries[1].metadata.get("output_text") == corrupted_output


def test_drop_known_contaminated_records_item_level_decisions_and_counts() -> None:
    entries = [
        MemoryEntry(
            entry_id="clean_game24_001",
            content="Break the problem into smaller sub-expressions.",
            memory_type="strategy",
            clean_or_contaminated="clean",
            source_trial_id=None,
            metadata={},
        ),
        MemoryEntry(
            entry_id="corrupted_game24_001",
            content="A misleading rule that does not help.",
            memory_type="wrong_rule",
            clean_or_contaminated="contaminated",
            source_trial_id=None,
            metadata={},
        ),
    ]

    kept, telemetry = drop_known_contaminated(entries)

    assert [entry.entry_id for entry in kept] == ["clean_game24_001"]
    assert telemetry["filter_name"] == "drop_known_contaminated"
    assert telemetry["input_count"] == 2
    assert telemetry["kept_count"] == 1
    assert telemetry["removed_count"] == 1
    assert telemetry["dropped"] == 1
    assert telemetry["input_source_ids"] == ["clean_game24_001", "corrupted_game24_001"]
    assert telemetry["kept_source_ids"] == ["clean_game24_001"]
    assert telemetry["removed_source_ids"] == ["corrupted_game24_001"]

    decisions = telemetry["decisions"]
    assert len(decisions) == 2

    clean_decision = next(d for d in decisions if d["entry_id"] == "clean_game24_001")
    assert (
        clean_decision["content_hash"]
        == sha256("Break the problem into smaller sub-expressions.".encode("utf-8")).hexdigest()
    )
    assert clean_decision["ground_truth"] == "clean"
    assert clean_decision["action"] == "kept"
    assert clean_decision["reason"] == "clean"
    assert clean_decision["score"] is None

    corrupted_decision = next(d for d in decisions if d["entry_id"] == "corrupted_game24_001")
    assert (
        corrupted_decision["content_hash"]
        == sha256("A misleading rule that does not help.".encode("utf-8")).hexdigest()
    )
    assert corrupted_decision["ground_truth"] == "contaminated"
    assert corrupted_decision["action"] == "removed"
    assert corrupted_decision["reason"] == "known_contaminated"
    assert corrupted_decision["score"] is None


def test_drop_known_contaminated_empty_input() -> None:
    kept, telemetry = drop_known_contaminated([])
    assert kept == []
    assert telemetry["filter_name"] == "drop_known_contaminated"
    assert telemetry["input_count"] == 0
    assert telemetry["kept_count"] == 0
    assert telemetry["removed_count"] == 0
    assert telemetry["decisions"] == []
    assert telemetry["input_source_ids"] == []
    assert telemetry["kept_source_ids"] == []
    assert telemetry["removed_source_ids"] == []
