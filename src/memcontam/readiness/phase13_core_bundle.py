from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.multiple_choice import build_instance


CoreTask: TypeAlias = Literal[
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
    "gpqa_diamond",
]
CORE_TASKS: tuple[CoreTask, ...] = (
    "mmlu_pro_engineering",
    "mmlu_pro_physics",
    "gpqa_diamond",
)


class CoreDatasetError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceArtifact(_FrozenModel):
    repo: str
    revision: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CoreSources(_FrozenModel):
    mmlu_pro: SourceArtifact
    gpqa: SourceArtifact


class SelectionProvenance(_FrozenModel):
    resource: Literal["memcontam.readiness/data/mmlu_pro_dc_selection_v1.json"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_revision: str
    dynamic_cheatsheet_revision: str


class Artifact(_FrozenModel):
    path: str
    rows: int
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CoreDatasetManifest(_FrozenModel):
    schema_version: Literal["phase13_core_dataset_bundle_v1"]
    sources: CoreSources
    selection: SelectionProvenance
    artifacts: dict[CoreTask, Artifact]

    @model_validator(mode="after")
    def exact_artifacts(self) -> CoreDatasetManifest:
        if set(self.artifacts) != set(CORE_TASKS):
            raise CoreDatasetError("CORE_DATASET_ARTIFACT_SET_MISMATCH")
        return self


class CoreDatasetSeal(_FrozenModel):
    schema_version: Literal["phase13_core_dataset_seal_v1"]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TaskDatasetReport(_FrozenModel):
    rows: int
    artifact_sha256: str
    order_sha256: str


class CoreDatasetReport(_FrozenModel):
    trajectory_seed: int
    manifest_sha256: str
    selection_sha256: str
    sources: CoreSources
    tasks: dict[CoreTask, TaskDatasetReport]


def write_bundle(
    root: Path,
    rows: Mapping[CoreTask, Sequence[TaskInstance]],
    *,
    sources: CoreSources,
    selection: SelectionProvenance,
) -> CoreDatasetManifest:
    validate_output_root(root)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        manifest = _write_bundle_contents(stage, rows, sources=sources, selection=selection)
        _fsync_directory(stage)
        if root.exists() or root.is_symlink():
            raise CoreDatasetError("CORE_DATASET_OUTPUT_EXISTS")
        os.rename(stage, root)
        _fsync_directory(root.parent)
        return manifest
    except OSError as error:
        raise CoreDatasetError("CORE_DATASET_WRITE_FAILED") from error
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _write_bundle_contents(
    root: Path,
    rows: Mapping[CoreTask, Sequence[TaskInstance]],
    *,
    sources: CoreSources,
    selection: SelectionProvenance,
) -> CoreDatasetManifest:
    artifacts: dict[CoreTask, Artifact] = {}
    for task in CORE_TASKS:
        payload = "".join(
            json.dumps(row.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows[task]
        )
        path = root / f"{task}.jsonl"
        _write_atomic(path, payload.encode())
        artifacts[task] = Artifact(
            path=path.name,
            rows=len(rows[task]),
            sha256=hashlib.sha256(payload.encode()).hexdigest(),
        )
    manifest = CoreDatasetManifest(
        schema_version="phase13_core_dataset_bundle_v1",
        sources=sources,
        selection=selection,
        artifacts=artifacts,
    )
    manifest_bytes = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    _write_atomic(root / "manifest.json", manifest_bytes)
    seal = CoreDatasetSeal(
        schema_version="phase13_core_dataset_seal_v1",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    _write_atomic(
        root / "seal.json",
        json.dumps(seal.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode(),
    )
    return manifest


def load_bundle_manifest(root: Path) -> tuple[CoreDatasetManifest, str]:
    manifest_bytes = _read(root / "manifest.json")
    seal_bytes = _read(root / "seal.json")
    try:
        manifest = CoreDatasetManifest.model_validate_json(manifest_bytes)
        seal = CoreDatasetSeal.model_validate_json(seal_bytes)
    except ValueError as error:
        raise CoreDatasetError("CORE_DATASET_MANIFEST_INVALID") from error
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != seal.manifest_sha256:
        raise CoreDatasetError("CORE_DATASET_SEAL_MISMATCH")
    return manifest, digest


def load_bundle_task(
    root: Path,
    task: CoreTask,
    *,
    expected_sha256: str | None = None,
) -> tuple[TaskInstance, ...]:
    manifest, _ = load_bundle_manifest(root)
    artifact = manifest.artifacts[task]
    if artifact.path != f"{task}.jsonl":
        raise CoreDatasetError("CORE_DATASET_PATH_MISMATCH")
    raw = _read(root / artifact.path)
    if hashlib.sha256(raw).hexdigest() != artifact.sha256:
        raise CoreDatasetError("CORE_DATASET_HASH_MISMATCH")
    if expected_sha256 is not None and artifact.sha256 != expected_sha256:
        raise CoreDatasetError("CORE_DATASET_CANONICAL_MISMATCH")
    try:
        result = tuple(build_instance(json.loads(line)) for line in raw.splitlines())
    except (KeyError, TypeError, ValueError) as error:
        raise CoreDatasetError("CORE_DATASET_SCHEMA_MISMATCH") from error
    if len(result) != artifact.rows or any(row.task_name != task for row in result):
        raise CoreDatasetError("CORE_DATASET_SCHEMA_MISMATCH")
    return result


def _read(path: Path) -> bytes:
    try:
        return read_regular_nofollow(path)
    except AuthorityFileError as error:
        raise CoreDatasetError("CORE_DATASET_FILE_NOT_REGULAR") from error


def _write_atomic(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_output_root(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise CoreDatasetError("CORE_DATASET_OUTPUT_EXISTS")
    parent = root.parent
    if not parent.is_dir():
        raise CoreDatasetError("CORE_DATASET_OUTPUT_PARENT_MISSING")
    for candidate in (parent, *parent.parents):
        if candidate.is_symlink():
            raise CoreDatasetError("CORE_DATASET_OUTPUT_SYMLINK")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
