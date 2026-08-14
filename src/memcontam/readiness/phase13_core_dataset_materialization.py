from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from memcontam.readiness.phase13_core_datasets import (
    GPQA_REPO,
    GPQA_REVISION,
    DYNAMIC_CHEATSHEET_REVISION,
    MMLU_PRO_REPO,
    MMLU_PRO_REVISION,
    CoreDatasetError,
    CoreTask,
)
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.multiple_choice import build_instance


class _MmluRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: int
    question: str
    options: tuple[str, ...]
    answer: str
    answer_index: int
    cot_content: str
    category: str
    src: str


class _GpqaRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    record_id: str = Field(alias="Record ID")
    question: str = Field(alias="Question")
    correct: str = Field(alias="Correct Answer")
    incorrect_1: str = Field(alias="Incorrect Answer 1")
    incorrect_2: str = Field(alias="Incorrect Answer 2")
    incorrect_3: str = Field(alias="Incorrect Answer 3")
    domain: str = Field(alias="High-level domain")


class _SelectionTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    released_arrow_sha256: str
    released_ordered_input_sha256: str
    question_ids: tuple[int, ...]


class _Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["mmlu_pro_dynamic_cheatsheet_selection_v1"]
    upstream_repo: Literal["TIGER-Lab/MMLU-Pro"]
    upstream_revision: Literal["475d58ba0cc18a15fd5d4221f41919199e692331"]
    upstream_test_lfs_sha256: Literal[
        "23e607af63e384127a344aa13efa207144215d951d914a572aed907b38879ba4"
    ]
    dynamic_cheatsheet_revision: Literal["5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9"]
    selection_method: Literal[
        "released category dataset shuffle(seed=10), first 250; identities mapped to pinned official question_id"
    ]
    tasks: dict[Literal["mmlu_pro_engineering", "mmlu_pro_physics"], _SelectionTask]

    @model_validator(mode="after")
    def exact_tasks(self) -> _Selection:
        if set(self.tasks) != {"mmlu_pro_engineering", "mmlu_pro_physics"}:
            raise ValueError("MMLU_SELECTION_TASK_SET_MISMATCH")
        return self


def build_rows(
    mmlu_path: Path,
    gpqa_path: Path,
    selection_path: Path,
    *,
    cache_dir: Path,
) -> dict[CoreTask, tuple[TaskInstance, ...]]:
    try:
        selection = _Selection.model_validate_json(selection_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CoreDatasetError("MMLU_SELECTION_INVALID") from error
    if (
        selection.upstream_revision != MMLU_PRO_REVISION
        or selection.dynamic_cheatsheet_revision != DYNAMIC_CHEATSHEET_REVISION
        or selection.upstream_test_lfs_sha256 != _sha256(mmlu_path)
    ):
        raise CoreDatasetError("MMLU_SELECTION_REVISION_MISMATCH")
    load_dataset = importlib.import_module("datasets").load_dataset
    mmlu_raw = load_dataset(
        "parquet", data_files=str(mmlu_path), split="train", cache_dir=str(cache_dir)
    )
    gpqa_raw = load_dataset(
        "csv", data_files=str(gpqa_path), split="train", cache_dir=str(cache_dir)
    )
    mmlu = tuple(_MmluRow.model_validate(row) for row in mmlu_raw)
    gpqa = tuple(_GpqaRow.model_validate(row) for row in gpqa_raw)
    if len(gpqa) != 198 or len({row.record_id for row in gpqa}) != len(gpqa):
        raise CoreDatasetError("GPQA_DIAMOND_IDENTITY_MISMATCH")
    return {
        "mmlu_pro_engineering": _mmlu_rows(
            mmlu, selection.tasks["mmlu_pro_engineering"], "mmlu_pro_engineering"
        ),
        "mmlu_pro_physics": _mmlu_rows(
            mmlu, selection.tasks["mmlu_pro_physics"], "mmlu_pro_physics"
        ),
        "gpqa_diamond": tuple(_gpqa_instance(row, index) for index, row in enumerate(gpqa)),
    }


def _mmlu_rows(
    rows: tuple[_MmluRow, ...],
    selection: _SelectionTask,
    task: Literal["mmlu_pro_engineering", "mmlu_pro_physics"],
) -> tuple[TaskInstance, ...]:
    selected = set(selection.question_ids)
    by_id = {row.question_id: (index, row) for index, row in enumerate(rows)}
    if len(selected) != 250 or not selected.issubset(by_id):
        raise CoreDatasetError("MMLU_RELEASE_SELECTION_MISMATCH")
    result = []
    for question_id in sorted(selected):
        index, row = by_id[question_id]
        if (
            row.category != selection.category
            or not 0 <= row.answer_index < len(row.options)
            or row.answer != chr(65 + row.answer_index)
        ):
            raise CoreDatasetError("MMLU_RELEASE_SELECTION_MISMATCH")
        result.append(
            build_instance(
                {
                    "sample_id": f"{task}:{question_id}",
                    "task_name": task,
                    "input": {"question": row.question, "options": list(row.options)},
                    "verifier_spec": {
                        "answer_index": row.answer_index,
                        "answer_label": row.answer,
                    },
                    "metadata": {
                        "dataset_repo": MMLU_PRO_REPO,
                        "dataset_revision": MMLU_PRO_REVISION,
                        "dataset_config": "default",
                        "dataset_split": "test",
                        "upstream_question_id": question_id,
                        "upstream_row_index": index,
                        "category": row.category,
                        "source": row.src,
                        "selection_release_revision": DYNAMIC_CHEATSHEET_REVISION,
                    },
                }
            )
        )
    return tuple(result)


def _gpqa_instance(row: _GpqaRow, index: int) -> TaskInstance:
    candidates = (
        (row.correct, True, 0),
        (row.incorrect_1, False, 1),
        (row.incorrect_2, False, 2),
        (row.incorrect_3, False, 3),
    )
    ordered = sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"sha256_record_v1\0{row.record_id}\0{item[2]}\0{item[0]}".encode()
        ).digest(),
    )
    answer_index = next(position for position, item in enumerate(ordered) if item[1])
    return build_instance(
        {
            "sample_id": f"gpqa_diamond:{row.record_id}",
            "task_name": "gpqa_diamond",
            "input": {"question": row.question, "options": [item[0] for item in ordered]},
            "verifier_spec": {
                "answer_index": answer_index,
                "answer_label": chr(65 + answer_index),
            },
            "metadata": {
                "dataset_repo": GPQA_REPO,
                "dataset_revision": GPQA_REVISION,
                "dataset_config": "gpqa_diamond",
                "dataset_split": "train",
                "upstream_record_id": row.record_id,
                "upstream_row_index": index,
                "category": row.domain,
                "source": "gpqa_diamond.csv",
                "choice_order_version": "sha256_record_v1",
            },
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
