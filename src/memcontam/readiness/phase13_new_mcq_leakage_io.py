from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.readiness.phase13_new_mcq_leakage import validate_leakage_artifact
from memcontam.readiness.phase13_new_mcq_leakage_models import (
    AuditDocument,
    DocumentEvidence,
    EvaluationItem,
    LeakageArtifact,
    LeakageArtifactError,
)
from memcontam.readiness.phase13_new_mcq_rag_frozen import AcceptedDocument

TASKS = ("mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond")
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _EvaluationInput(_FrozenModel):
    question: str
    options: tuple[str, ...]


class _EvaluationMetadata(_FrozenModel):
    category: str
    dataset_config: str
    dataset_repo: str
    dataset_revision: str
    source: str
    upstream_question_id: int | None = None
    upstream_record_id: str | None = None


class _EvaluationRow(_FrozenModel):
    sample_id: str
    task_name: str
    input: _EvaluationInput
    metadata: _EvaluationMetadata


class _SourceTask(_FrozenModel):
    question_bearing_rows: Literal[0]
    source_registry_ids: tuple[str, ...]
    main_exclusion_registry_hash: str


class _SourceRegistry(_FrozenModel):
    status: Literal["COMPLETE"]
    tasks: dict[str, _SourceTask]


class _ManifestArtifact(_FrozenModel):
    path: str
    rows: int
    sha256: str


class _EvaluationManifest(_FrozenModel):
    artifacts: dict[str, _ManifestArtifact]


@dataclass(frozen=True, slots=True)
class LeakageInputs:
    documents: tuple[AuditDocument, ...]
    evaluation_items: tuple[EvaluationItem, ...]
    input_hashes: tuple[tuple[str, str], ...]


def load_leakage_inputs(root: Path, evaluation_root: Path) -> LeakageInputs:
    try:
        registry = _SourceRegistry.model_validate_json(
            (root / "source_eligibility_registry_v1.json").read_bytes()
        )
        manifest_path = evaluation_root / "manifest.json"
        manifest = _EvaluationManifest.model_validate_json(manifest_path.read_bytes())
        manifest_hash = _sha256(manifest_path)
        if (
            set(registry.tasks) != set(TASKS)
            or any(
                source.main_exclusion_registry_hash != manifest_hash
                for source in registry.tasks.values()
            )
        ):
            raise LeakageArtifactError("NEW_MCQ_LEAKAGE_SOURCE_REGISTRY_INVALID")
        _validate_evaluation_manifest(evaluation_root, manifest)
        documents = tuple(
            document
            for task in TASKS
            for document in _load_documents(root, task, registry.tasks[task])
        )
        items = tuple(
            item
            for task in TASKS
            for item in _load_evaluation(evaluation_root / f"{task}.jsonl", task)
        )
    except (OSError, UnicodeError, ValidationError, json.JSONDecodeError) as error:
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_INPUT_INVALID") from error
    hashes = {
        "source_eligibility_registry": _sha256(root / "source_eligibility_registry_v1.json"),
        "evaluation_manifest": _sha256(evaluation_root / "manifest.json"),
    }
    for task in TASKS:
        hashes[f"accepted:{task}"] = _sha256(root / "accepted" / f"{task}.jsonl")
        hashes[f"evaluation:{task}"] = _sha256(evaluation_root / f"{task}.jsonl")
    return LeakageInputs(documents, items, tuple(sorted(hashes.items())))


def write_leakage_artifact(path: Path, artifact: LeakageArtifact) -> None:
    validate_leakage_artifact(artifact)
    path.write_text(
        json.dumps(asdict(artifact), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_leakage_artifact(path: Path) -> LeakageArtifact:
    try:
        payload = _JSON_OBJECT.validate_json(path.read_bytes())
        _validate_artifact_shape(payload)
        artifact = TypeAdapter(LeakageArtifact).validate_python(payload)
    except (OSError, ValidationError) as error:
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_ARTIFACT_INVALID") from error
    validate_leakage_artifact(artifact)
    return artifact


def _validate_artifact_shape(payload: dict[str, JsonValue]) -> None:
    expected_artifact = {field.name for field in fields(LeakageArtifact)}
    evidence = payload.get("document_evidence")
    if set(payload) != expected_artifact or not isinstance(evidence, list):
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_ARTIFACT_INVALID")
    expected_evidence = {field.name for field in fields(DocumentEvidence)}
    if any(not isinstance(row, dict) or set(row) != expected_evidence for row in evidence):
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_ARTIFACT_INVALID")


def _load_documents(
    root: Path,
    task: str,
    source: _SourceTask,
) -> tuple[AuditDocument, ...]:
    rows = tuple(
        AcceptedDocument.model_validate_json(line)
        for line in (root / "accepted" / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
    )
    if (
        len(rows) != 24
        or len({row.document_id for row in rows}) != 24
        or any(
            row.task_id != task
            or row.content_hash != canonical_json_hash(row.text)
            or not set(row.source_registry_ids) <= set(source.source_registry_ids)
            for row in rows
        )
    ):
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_DOCUMENT_REGISTRY_INVALID")
    return tuple(
        AuditDocument(
            row.document_id,
            row.task_id,
            row.text,
            row.source_registry_ids,
            (),
        )
        for row in rows
    )


def _load_evaluation(path: Path, task: str) -> tuple[EvaluationItem, ...]:
    rows = tuple(
        _EvaluationRow.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    if not rows or any(row.task_name != task for row in rows):
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_EVALUATION_INVALID")
    return tuple(_evaluation_item(row) for row in rows)


def _validate_evaluation_manifest(
    root: Path,
    manifest: _EvaluationManifest,
) -> None:
    if set(manifest.artifacts) != set(TASKS):
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_EVALUATION_INVALID")
    for task in TASKS:
        artifact = manifest.artifacts[task]
        path = root / f"{task}.jsonl"
        rows = sum(bool(line) for line in path.read_text(encoding="utf-8").splitlines())
        if (
            artifact.path != path.name
            or artifact.rows != rows
            or artifact.sha256 != _sha256(path)
        ):
            raise LeakageArtifactError("NEW_MCQ_LEAKAGE_EVALUATION_INVALID")


def _evaluation_item(row: _EvaluationRow) -> EvaluationItem:
    metadata = row.metadata
    upstream_id = (
        metadata.upstream_record_id
        if metadata.upstream_record_id is not None
        else str(metadata.upstream_question_id)
    )
    source_key = (
        f"{metadata.dataset_repo}@{metadata.dataset_revision}:"
        f"{metadata.dataset_config}:{metadata.source}:{upstream_id}"
    )
    return EvaluationItem(
        row.task_name,
        row.sample_id,
        row.input.question,
        row.input.options,
        (f"evaluation:{row.sample_id}", f"source:{source_key}"),
        (row.sample_id, source_key),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "LeakageInputs",
    "load_leakage_artifact",
    "load_leakage_inputs",
    "write_leakage_artifact",
]
