from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import BranchCorpus, BranchCorpusSet, CleanCorpus
from memcontam.readiness.phase13_new_mcq_rag import (
    Candidate,
    TASKS,
    validate_new_mcq_rag_package,
)

from .phase13_new_mcq_rag_artifacts import leakage, runtime, source_eligibility
from .phase13_new_mcq_rag_frozen import EXPECTED_CLASSES, SerializedCleanIndex
from .phase13_new_mcq_rag_manifest import (
    REMAINING_OBJECTS,
    Artifact,
    package_reconstruction_identity,
)

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class _CachedProvider:
    def __init__(self, provider: BgeM3EmbeddingProvider) -> None:
        self.provider = provider
        self.cache: dict[str, list[float]] = {}
        self.embedding_contract: dict[str, str | int | bool] = {
            "dimension": BgeM3EmbeddingProvider.VECTOR_DIMENSION,
            "normalized": BgeM3EmbeddingProvider.NORMALIZE_EMBEDDINGS,
            "production_identity": (
                f"{BgeM3EmbeddingProvider.MODEL_ID}@{BgeM3EmbeddingProvider.REVISION}"
            ),
            "provider": BgeM3EmbeddingProvider.MODEL_ID,
        }

    def encode_query(self, text: str) -> list[float]:
        return self.provider.encode_query(text)

    def encode_document(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.provider.encode_document(text)
        return self.cache[text]


def materialize_new_mcq_rag_package(root: Path, evaluation_root: Path, cache_root: Path) -> None:
    provider = _CachedProvider(
        BgeM3EmbeddingProvider(cache_folder=cache_root, local_files_only=True, batch_size=32)
    )
    candidates = {task: _candidates(root, task) for task in TASKS}
    _write_accepted(root, candidates)
    _write_json(
        root / "source_eligibility_registry_v1.json",
        source_eligibility(root, evaluation_root),
    )
    _write_json(root / "embedding_runtime_v1.json", runtime(cache_root, provider.provider))
    _write_json(root / "leakage_report_v1.json", leakage(root, evaluation_root))
    indices = {task: _clean_index(task, candidates[task], provider) for task in TASKS}
    (root / "indices").mkdir(exist_ok=True)
    for task, payload in indices.items():
        _write_json(root / "indices" / f"{task}.json", payload)
    _write_manifest(root, candidates, indices)
    _write_status(root)
    validate_new_mcq_rag_package(root, evaluation_root)


def _candidates(root: Path, task: str) -> tuple[Candidate, ...]:
    return tuple(
        Candidate.model_validate_json(line)
        for line in (root / "candidates" / f"{task}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )


def _write_accepted(root: Path, candidates: dict[str, tuple[Candidate, ...]]) -> None:
    (root / "accepted").mkdir(exist_ok=True)
    for task, rows in candidates.items():
        payload = "".join(
            json.dumps(
                {
                    **row.model_dump(mode="json"),
                    "review_status": "accepted",
                    "content_hash": canonical_json_hash(row.text),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
        (root / "accepted" / f"{task}.jsonl").write_text(payload, encoding="utf-8")


def _clean_index(
    task: str,
    rows: tuple[Candidate, ...],
    provider: _CachedProvider,
) -> dict[str, JsonValue]:
    clean = CleanCorpus.from_documents(
        [{"id": row.document_id, "text": row.text} for row in rows],
        corpus_id=f"new_mcq_rag_v1::{task}",
    )
    corpus = BranchCorpus(
        branch="clean",
        documents=clean.documents,
        active_document_ids=tuple(row.document_id for row in clean.documents),
        serialization_id=f"{clean.corpus_id}|clean|branch-corpus-v3",
    )
    corpus_set = BranchCorpusSet(
        clean=clean,
        branches={"clean": corpus},
        serialization_id=f"{clean.corpus_id}|base",
    )
    index = build_branch_indices(corpus_set, provider, None).branches["clean"]
    return _JSON_OBJECT.validate_python(
        {
            "schema_version": "new_mcq_rag_serialized_clean_index_v1",
            "task_id": task,
            "corpus_serialization_id": corpus.serialization_id,
            "corpus_content_hash": corpus.content_hash,
            "index_serialization_id": index.serialization_id,
            "index_artifact_hash": index.artifact_hash,
            "embedding_contract": dict(index.embedding_contract),
            "top_k": 3,
            "documents": [document.payload() for document in index.documents],
            "vectors": {
                document_id: list(vector) for document_id, vector in index.vectors.items()
            },
        }
    )


def _write_manifest(
    root: Path,
    candidates: dict[str, tuple[Candidate, ...]],
    indices: dict[str, dict[str, JsonValue]],
) -> None:
    artifacts = {
        "complete_source_eligibility_registry": [
            _artifact(root, "source_eligibility_registry_v1.json")
        ],
        "accepted_document_registry": [
            _artifact(root, f"accepted/{task}.jsonl") for task in TASKS
        ],
        "partial_embedding_runtime_artifact": [_artifact(root, "embedding_runtime_v1.json")],
        "serialized_clean_index_artifacts": [
            _artifact(root, f"indices/{task}.json") for task in TASKS
        ],
        "partial_clean_document_leakage_evidence": [
            _artifact(root, "leakage_report_v1.json")
        ],
    }
    assert set(artifacts) == set(EXPECTED_CLASSES)
    tasks: dict[str, JsonValue] = {}
    task_artifacts: list[dict[str, str]] = []
    for task, rows in candidates.items():
        serialized = SerializedCleanIndex.model_validate(indices[task])
        candidate_artifact = _artifact(root, f"candidates/{task}.jsonl")
        review_artifact = _artifact(root, f"reviews/{task}.json")
        accepted_artifact = _artifact(root, f"accepted/{task}.jsonl")
        index_artifact = _artifact(root, f"indices/{task}.json")
        task_artifacts.extend(
            (candidate_artifact, review_artifact, accepted_artifact, index_artifact)
        )
        tasks[task] = _JSON_OBJECT.validate_python(
            {
                "documents": 24,
                "candidate": candidate_artifact,
                "review": review_artifact,
                "accepted": accepted_artifact,
                "index": index_artifact,
                "corpus_hash": canonical_json_hash(
                    [{"id": row.document_id, "text": row.text} for row in rows]
                ),
                "index_hashes": {"clean": serialized.index_artifact_hash},
            }
        )
    source_registry = _artifact(root, "source_registry_v1.json")
    authoring_contract = _artifact(root, "authoring_contract_v1.json")
    reconstruction = package_reconstruction_identity(
        tuple(
            Artifact.model_validate(artifact)
            for artifact in (
                source_registry,
                authoring_contract,
                *(item for values in artifacts.values() for item in values),
                *task_artifacts,
            )
        )
    )
    _write_json(
        root / "package_manifest_v1.json",
        _JSON_OBJECT.validate_python(
            {
                "schema_version": "new_mcq_rag_package_manifest_v1",
                "source_registry": source_registry,
                "authoring_contract": authoring_contract,
                "required_artifacts": artifacts,
                "tasks": tasks,
                "package_reconstruction_identity": reconstruction,
                "promotion": {
                    "status": "NOT_READY",
                    "reason": "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN",
                    "remaining_objects": list(REMAINING_OBJECTS),
                },
            }
        ),
    )


def _write_status(root: Path) -> None:
    manifest = json.loads((root / "package_manifest_v1.json").read_text(encoding="utf-8"))
    status = json.loads((root.parent / "new_mcq_rag_status_v1.json").read_text(encoding="utf-8"))
    status["candidate_package"] = {
        "path": str(root / "package_manifest_v1.json"),
        "sha256": _sha256(root / "package_manifest_v1.json"),
        "status": "CLEAN_PACKAGE_NOT_READY",
        "reconstruction_identity": manifest["package_reconstruction_identity"],
    }
    status["cells"] = {
        task: {
            "status": "NOT_READY",
            "reason": "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN",
            "entry_condition_met": False,
            "missing_objects": list(REMAINING_OBJECTS),
            "index_hashes": manifest["tasks"][task]["index_hashes"],
        }
        for task in TASKS
    }
    _write_json(root.parent / "new_mcq_rag_status_v1.json", status, pretty=True)


def _artifact(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(root / relative)}


def _write_json(path: Path, value: JsonValue, *, pretty: bool = False) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=not pretty,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["materialize_new_mcq_rag_package"]
