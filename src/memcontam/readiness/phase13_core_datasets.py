from __future__ import annotations

import hashlib
import importlib.resources
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from memcontam.readiness.phase13_core_bundle import (
    CORE_TASKS,
    Artifact,
    CoreDatasetError,
    CoreDatasetManifest,
    CoreDatasetReport,
    CoreSources,
    CoreTask,
    SelectionProvenance,
    SourceArtifact,
    TaskDatasetReport,
    load_bundle_manifest,
    load_bundle_task,
    validate_output_root,
    write_bundle,
)
from memcontam.tasks.base import TaskInstance


MMLU_PRO_REVISION: Final = "475d58ba0cc18a15fd5d4221f41919199e692331"
GPQA_REVISION: Final = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
MMLU_PRO_REPO: Final = "TIGER-Lab/MMLU-Pro"
GPQA_REPO: Final = "Idavidrein/gpqa"
MMLU_PRO_TEST_SHA256: Final = "23e607af63e384127a344aa13efa207144215d951d914a572aed907b38879ba4"
GPQA_SOURCE_SHA256: Final = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
DYNAMIC_CHEATSHEET_REVISION: Final = "5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9"
SELECTION_PATH: Final = Path(
    str(
        importlib.resources.files("memcontam.readiness").joinpath(
            "data/mmlu_pro_dc_selection_v1.json"
        )
    )
)


class Download(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        repo_type: str,
        filename: str,
        revision: str,
        token: bool | None = None,
        cache_dir: str | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class _DownloadedSources:
    mmlu_test: Path
    gpqa_diamond: Path


def materialize_core_datasets(
    root: Path,
    *,
    download: Download | None = None,
) -> CoreDatasetManifest:
    _validate_materialization_root(root)
    validate_output_root(root)
    downloader = _hf_download if download is None else download
    from memcontam.readiness.phase13_core_dataset_materialization import build_rows

    with tempfile.TemporaryDirectory(prefix="memcontam-core-sources-") as directory:
        try:
            snapshots = Path(directory)
            downloaded = _download_sources(downloader, snapshots / "hub-cache")
            mmlu_bytes = downloaded.mmlu_test.read_bytes()
            gpqa_bytes = downloaded.gpqa_diamond.read_bytes()
            selection_bytes = SELECTION_PATH.read_bytes()
            mmlu_sha = hashlib.sha256(mmlu_bytes).hexdigest()
            gpqa_sha = hashlib.sha256(gpqa_bytes).hexdigest()
            if mmlu_sha != MMLU_PRO_TEST_SHA256 or gpqa_sha != GPQA_SOURCE_SHA256:
                raise CoreDatasetError("CORE_DATASET_SOURCE_MISMATCH")
            mmlu_path = snapshots / "mmlu.parquet"
            gpqa_path = snapshots / "gpqa.csv"
            selection_path = snapshots / "selection.json"
            mmlu_path.write_bytes(mmlu_bytes)
            gpqa_path.write_bytes(gpqa_bytes)
            selection_path.write_bytes(selection_bytes)
            rows = build_rows(
                mmlu_path,
                gpqa_path,
                selection_path,
                cache_dir=snapshots / "datasets-cache",
            )
        except CoreDatasetError:
            raise
        except Exception as error:
            raise CoreDatasetError("CORE_DATASET_MATERIALIZATION_FAILED") from error
    sources = CoreSources(
        mmlu_pro=SourceArtifact(
            repo=MMLU_PRO_REPO,
            revision=MMLU_PRO_REVISION,
            path="data/test-00000-of-00001.parquet",
            sha256=mmlu_sha,
        ),
        gpqa=SourceArtifact(
            repo=GPQA_REPO,
            revision=GPQA_REVISION,
            path="gpqa_diamond.csv",
            sha256=gpqa_sha,
        ),
    )
    selection = SelectionProvenance(
        resource="memcontam.readiness/data/mmlu_pro_dc_selection_v1.json",
        sha256=hashlib.sha256(selection_bytes).hexdigest(),
        upstream_revision=MMLU_PRO_REVISION,
        dynamic_cheatsheet_revision=DYNAMIC_CHEATSHEET_REVISION,
    )
    return write_bundle(root, rows, sources=sources, selection=selection)


def load_core_task(root: Path, task: CoreTask) -> tuple[TaskInstance, ...]:
    return load_bundle_task(root, task)


def paired_trajectory_order(
    rows: Sequence[TaskInstance],
    *,
    trajectory_seed: int,
) -> tuple[TaskInstance, ...]:
    if not rows:
        raise CoreDatasetError("EMPTY_TASK_TRAJECTORY")
    task_names = {row.task_name for row in rows}
    sample_ids = {row.sample_id for row in rows}
    if len(task_names) != 1 or len(sample_ids) != len(rows):
        raise CoreDatasetError("TASK_TRAJECTORY_NOT_ISOLATED")
    task = next(iter(task_names))
    return tuple(
        sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"sha256_task_seed_v1\0{task}\0{trajectory_seed}\0{row.sample_id}".encode()
            ).digest(),
        )
    )


def trajectory_order_sha256(rows: Sequence[TaskInstance], trajectory_seed: int) -> str:
    ordered = paired_trajectory_order(rows, trajectory_seed=trajectory_seed)
    return hashlib.sha256("\n".join(row.sample_id for row in ordered).encode()).hexdigest()


def validate_core_datasets(
    root: Path,
    *,
    trajectory_seed: int,
    expected_counts: Mapping[CoreTask, int] | None = None,
) -> CoreDatasetReport:
    expected = (
        {"mmlu_pro_engineering": 250, "mmlu_pro_physics": 250, "gpqa_diamond": 198}
        if expected_counts is None
        else expected_counts
    )
    manifest, manifest_sha = load_bundle_manifest(root)
    selection_sha = hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest()
    if (
        manifest.selection.sha256 != selection_sha
        or manifest.selection.upstream_revision != MMLU_PRO_REVISION
        or manifest.selection.dynamic_cheatsheet_revision != DYNAMIC_CHEATSHEET_REVISION
    ):
        raise CoreDatasetError("CORE_DATASET_SELECTION_MISMATCH")
    if (
        manifest.sources.mmlu_pro.repo != MMLU_PRO_REPO
        or manifest.sources.mmlu_pro.revision != MMLU_PRO_REVISION
        or manifest.sources.mmlu_pro.path != "data/test-00000-of-00001.parquet"
        or manifest.sources.mmlu_pro.sha256 != MMLU_PRO_TEST_SHA256
        or manifest.sources.gpqa.repo != GPQA_REPO
        or manifest.sources.gpqa.revision != GPQA_REVISION
        or manifest.sources.gpqa.path != "gpqa_diamond.csv"
        or manifest.sources.gpqa.sha256 != GPQA_SOURCE_SHA256
    ):
        raise CoreDatasetError("CORE_DATASET_SOURCE_MISMATCH")
    reports: dict[CoreTask, TaskDatasetReport] = {}
    for task in CORE_TASKS:
        rows = load_core_task(root, task)
        if task in expected and len(rows) != expected[task]:
            raise CoreDatasetError("CORE_DATASET_COUNT_MISMATCH")
        reports[task] = TaskDatasetReport(
            rows=len(rows),
            artifact_sha256=manifest.artifacts[task].sha256,
            order_sha256=trajectory_order_sha256(rows, trajectory_seed),
        )
    return CoreDatasetReport(
        trajectory_seed=trajectory_seed,
        manifest_sha256=manifest_sha,
        selection_sha256=selection_sha,
        sources=manifest.sources,
        tasks=reports,
    )


def _download_sources(download: Download, cache_dir: Path) -> _DownloadedSources:
    try:
        gpqa = Path(
            download(
                repo_id=GPQA_REPO,
                repo_type="dataset",
                filename="gpqa_diamond.csv",
                revision=GPQA_REVISION,
                token=True,
                cache_dir=str(cache_dir),
            )
        )
    except PermissionError as error:
        raise CoreDatasetError("GPQA_ACCESS_REQUIRED") from error
    try:
        mmlu = Path(
            download(
                repo_id=MMLU_PRO_REPO,
                repo_type="dataset",
                filename="data/test-00000-of-00001.parquet",
                revision=MMLU_PRO_REVISION,
                cache_dir=str(cache_dir),
            )
        )
    except PermissionError as error:
        raise CoreDatasetError("MMLU_PRO_DOWNLOAD_FAILED") from error
    return _DownloadedSources(mmlu, gpqa)


def _hf_download(**kwargs: Any) -> str:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import constants
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError

    constants.HF_HUB_DISABLE_XET = True
    try:
        return hf_hub_download(**kwargs)
    except (GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError) as error:
        raise PermissionError("dataset download authorization failed") from error


def _validate_materialization_root(root: Path) -> None:
    absolute = Path(os.path.abspath(root))
    repository = next(
        (
            candidate
            for candidate in (absolute.parent, *absolute.parents)
            if (candidate / ".git").exists()
        ),
        None,
    )
    if repository is None:
        return
    if absolute != repository / "data/phase13/core/materialized":
        raise CoreDatasetError("CORE_DATASET_OUTPUT_NOT_PROTECTED")


__all__ = [
    "Artifact",
    "CoreDatasetError",
    "CoreDatasetManifest",
    "CoreDatasetReport",
    "GPQA_REVISION",
    "MMLU_PRO_REVISION",
    "SELECTION_PATH",
    "load_core_task",
    "materialize_core_datasets",
    "paired_trajectory_order",
    "validate_core_datasets",
]
