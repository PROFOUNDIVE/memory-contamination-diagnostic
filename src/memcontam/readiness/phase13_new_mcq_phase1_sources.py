from __future__ import annotations

import csv
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from memcontam.readiness.phase13_core_bundle import CoreTask
from memcontam.readiness.phase13_new_mcq_candidate import DisplayedMcq, mcq_normalize
from memcontam.readiness.phase13_new_mcq_phase1_models import Phase1SourcePaths

MMLU_REVISION: Final = "475d58ba0cc18a15fd5d4221f41919199e692331"
MMLU_SHA256: Final = "a6db33e44c7a8d6a0a9665aabe6596a5e7436bebb62412d1219821283835e457"
GPQA_REVISION: Final = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
GPQA_MAIN_SHA256: Final = "acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3"
GPQA_DIAMOND_SHA256: Final = "d6413fa81bdbc1bf08a83cc81c1a369bcbaf9a51d27c027e0b3f219e584be372"
SPLIT_LAW_ID: Final = "phase13_source_identity_hash_50_50_v1"
GPQA_DISPLAY_LAW_ID: Final = "gpqa_display_perm_v1"


@dataclass(frozen=True, slots=True)
class Phase1FreezeError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


class _MmluRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    question_id: int
    question: str
    options: tuple[str, ...]
    answer_index: int
    category: str


class _DiamondMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    upstream_record_id: str


class _DiamondRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    metadata: _DiamondMetadata


@dataclass(frozen=True, slots=True)
class SourceFacts:
    dataset_identity: str
    revision: str
    selected_config: Literal["mmlu_pro_validation", "gpqa_main", "gpqa_extended"]
    source_sha256: str
    exclusion_source_sha256: str | None
    excluded_rows: int


@dataclass(frozen=True, slots=True)
class CertificationItem:
    opaque_item_id: str
    canonical_local_sample_id: str
    permitted_content_hash: str
    displayed: DisplayedMcq


@dataclass(frozen=True, slots=True)
class TaskSourcePool:
    task_id: CoreTask
    facts: SourceFacts
    items: tuple[CertificationItem, ...]


def load_source_pools(
    paths: Phase1SourcePaths,
) -> tuple[TaskSourcePool, TaskSourcePool, TaskSourcePool]:
    mmlu_bytes = _verified_bytes(
        paths.mmlu_source, MMLU_SHA256, "PHASE1_MMLU_SOURCE_HASH_MISMATCH"
    )
    rows = _load_mmlu(paths.mmlu_source)
    mmlu_facts = SourceFacts(
        "TIGER-Lab/MMLU-Pro", MMLU_REVISION, "mmlu_pro_validation",
        hashlib.sha256(mmlu_bytes).hexdigest(), None, 0,
    )
    engineering = _mmlu_pool(rows, "mmlu_pro_engineering", mmlu_facts)
    physics = _mmlu_pool(rows, "mmlu_pro_physics", mmlu_facts)
    gpqa = _load_gpqa(paths)
    return engineering, physics, gpqa


def split_items(pool: TaskSourcePool) -> tuple[tuple[CertificationItem, ...], tuple[CertificationItem, ...], str]:
    ranked = sorted(
        pool.items,
        key=lambda item: _sha256(
            f"{SPLIT_LAW_ID}\0{pool.task_id}\0{pool.facts.source_sha256}\0{item.opaque_item_id}"
        ),
    )
    build_size = len(ranked) // 2
    split_identity = _json_hash(
        {
            "law": SPLIT_LAW_ID,
            "task": pool.task_id,
            "source_sha256": pool.facts.source_sha256,
            "build": [item.opaque_item_id for item in ranked[:build_size]],
            "calibration": [item.opaque_item_id for item in ranked[build_size:]],
        }
    )
    return tuple(ranked[:build_size]), tuple(ranked[build_size:]), split_identity


def _load_mmlu(path: Path) -> tuple[_MmluRow, ...]:
    try:
        dataset = importlib.import_module("datasets").load_dataset(
            "parquet", data_files=str(path), split="train"
        )
        return TypeAdapter(tuple[_MmluRow, ...]).validate_python(dataset)
    except (AttributeError, ImportError, TypeError, ValidationError, ValueError) as error:
        raise Phase1FreezeError("PHASE1_MMLU_SOURCE_INVALID") from error


def _mmlu_pool(
    rows: tuple[_MmluRow, ...], task_id: CoreTask, facts: SourceFacts
) -> TaskSourcePool:
    category = "engineering" if task_id == "mmlu_pro_engineering" else "physics"
    selected = tuple(row for row in rows if row.category == category)
    if len(selected) != 5:
        raise Phase1FreezeError("PHASE1_MMLU_SOURCE_CARDINALITY_INVALID")
    items = tuple(
        _mmlu_item(task_id, category, row)
        for row in selected
    )
    return TaskSourcePool(task_id, facts, items)


def _load_gpqa(
    paths: Phase1SourcePaths,
) -> TaskSourcePool:
    main_bytes = _verified_bytes(
        paths.gpqa_main_source, GPQA_MAIN_SHA256, "PHASE1_GPQA_MAIN_HASH_MISMATCH"
    )
    diamond_bytes = _verified_bytes(
        paths.gpqa_diamond_evaluation,
        GPQA_DIAMOND_SHA256,
        "PHASE1_GPQA_DIAMOND_HASH_MISMATCH",
    )
    excluded = {
        row.metadata.upstream_record_id
        for row in TypeAdapter(tuple[_DiamondRow, ...]).validate_json(
            b"[" + b",".join(diamond_bytes.splitlines()) + b"]"
        )
    }
    rows = _gpqa_csv(paths.gpqa_main_source, excluded, require_all_exclusions=True)
    selected_config: Literal["gpqa_main", "gpqa_extended"] = "gpqa_main"
    source_bytes = main_bytes
    if not rows:
        if paths.gpqa_extended_source is None or paths.gpqa_extended_sha256 is None:
            raise Phase1FreezeError("PHASE1_GPQA_EXTENDED_FALLBACK_UNAVAILABLE")
        source_bytes = _verified_bytes(
            paths.gpqa_extended_source,
            paths.gpqa_extended_sha256,
            "PHASE1_GPQA_EXTENDED_HASH_MISMATCH",
        )
        rows = _gpqa_csv(paths.gpqa_extended_source, excluded, require_all_exclusions=False)
        selected_config = "gpqa_extended"
    if not rows:
        raise Phase1FreezeError("PHASE1_GPQA_NO_ELIGIBLE_ROWS")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    facts = SourceFacts(
        "Idavidrein/gpqa", GPQA_REVISION, selected_config, source_hash,
        hashlib.sha256(diamond_bytes).hexdigest(), len(excluded),
    )
    items = tuple(_gpqa_item(row, source_hash, selected_config) for row in rows)
    return TaskSourcePool("gpqa_diamond", facts, items)


def _gpqa_csv(
    path: Path, excluded: set[str], *, require_all_exclusions: bool
) -> tuple[dict[str, str], ...]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
        required = {"Record ID", "Question", "Correct Answer", *(f"Incorrect Answer {i}" for i in range(1, 4))}
        if not rows or not required <= set(rows[0]):
            raise Phase1FreezeError("PHASE1_GPQA_SOURCE_INVALID")
        record_ids = tuple(row["Record ID"] for row in rows)
        if len(set(record_ids)) != len(record_ids):
            raise Phase1FreezeError("PHASE1_GPQA_SOURCE_INVALID")
        if require_all_exclusions and not excluded <= set(record_ids):
            raise Phase1FreezeError("PHASE1_GPQA_DIAMOND_EXCLUSION_MISMATCH")
        return tuple(row for row in rows if row["Record ID"] not in excluded)
    except (KeyError, OSError, UnicodeError) as error:
        raise Phase1FreezeError("PHASE1_GPQA_SOURCE_INVALID") from error


def _gpqa_item(
    row: dict[str, str], source_hash: str, source_config: Literal["gpqa_main", "gpqa_extended"]
) -> CertificationItem:
    options = (row["Correct Answer"], *(row[f"Incorrect Answer {i}"] for i in range(1, 4)))
    seed = f"phase1-certification:{source_hash}"
    keyed = tuple(
        (
            _sha256(
                f"{GPQA_DISPLAY_LAW_ID}\0gpqa_diamond\0{seed}\0{row['Record ID']}\0"
                f"{index}\0{_sha256(option)}"
            ),
            index,
            option,
        )
        for index, option in enumerate(options)
    )
    ordered = tuple(sorted(keyed))
    ordered_indices = tuple(index for _, index, _ in ordered)
    display_id = _json_hash(
        {
            "version": GPQA_DISPLAY_LAW_ID,
            "task": "gpqa_diamond",
            "trajectory_seed": seed,
            "upstream_record_id": row["Record ID"],
            "ordered_source_indices": list(ordered_indices),
            "ordered_option_hashes": [_sha256(option) for _, _, option in ordered],
        }
    )
    source_key = f"{GPQA_REVISION}:{source_config}:{row['Record ID']}"
    opaque_id = _sha256(f"phase13_new_mcq_opaque_item_v1\0gpqa_diamond\0{source_key}")
    displayed = DisplayedMcq(
        opaque_id,
        row["Question"],
        tuple(option for _, _, option in ordered),
        ordered_indices.index(0),
        display_id,
    )
    return _item("gpqa_diamond", source_key, displayed)


def _mmlu_item(task_id: CoreTask, category: str, row: _MmluRow) -> CertificationItem:
    source_key = f"{MMLU_REVISION}:{category}:{row.question_id}"
    opaque_id = _sha256(f"phase13_new_mcq_opaque_item_v1\0{task_id}\0{source_key}")
    display_id = _json_hash(
        {"task": task_id, "question_id": row.question_id, "source_order": True}
    )
    displayed = DisplayedMcq(
        opaque_id, row.question, row.options, row.answer_index, display_id
    )
    return _item(task_id, source_key, displayed)


def _item(
    task_id: CoreTask, source_key: str, displayed: DisplayedMcq
) -> CertificationItem:
    content_hash = _json_hash(
        {
            "stem": mcq_normalize(displayed.stem),
            "options": sorted(mcq_normalize(value) for value in displayed.options),
        }
    )
    return CertificationItem(
        displayed.query_id,
        _sha256(source_key),
        content_hash,
        displayed,
    )


def _verified_bytes(path: Path, expected: str, code: str) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise Phase1FreezeError(code) from error
    if hashlib.sha256(content).hexdigest() != expected:
        raise Phase1FreezeError(code)
    return content


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_hash(value: dict[str, str | bool | int | list[str] | list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "CertificationItem",
    "Phase1FreezeError",
    "SourceFacts",
    "TaskSourcePool",
    "load_source_pools",
    "split_items",
]
