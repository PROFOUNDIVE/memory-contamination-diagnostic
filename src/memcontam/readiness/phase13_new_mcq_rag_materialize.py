from __future__ import annotations

import json
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import BranchCorpusSet, CleanCorpus, build_branch_corpora
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_models import semantics
from memcontam.readiness.phase13_new_mcq_leakage import AuditDocument, audit_documents
from memcontam.readiness.phase13_new_mcq_leakage_io import (
    load_leakage_inputs,
    write_leakage_artifact,
)
from memcontam.readiness.phase13_new_mcq_rag import Candidate, TASKS, validate_new_mcq_rag_package

from .phase13_new_mcq_rag_artifacts import runtime, source_eligibility
from .phase13_new_mcq_rag_authority import authority_selection
from .phase13_new_mcq_rag_models import (
    BRANCHES,
    InterventionRegistry,
)
from .phase13_new_mcq_rag_materialize_output import (
    sha256,
    write_json,
    write_manifest,
    write_status,
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

    @property
    def metadata(self) -> dict[str, str | int | bool]:
        return {
            "model_id": BgeM3EmbeddingProvider.MODEL_ID,
            "revision": BgeM3EmbeddingProvider.REVISION,
            "vector_dimension": BgeM3EmbeddingProvider.VECTOR_DIMENSION,
            "normalize_embeddings": BgeM3EmbeddingProvider.NORMALIZE_EMBEDDINGS,
        }

    def encode_query(self, text: str) -> list[float]:
        return self.provider.encode_query(text)

    def encode_document(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.provider.encode_document(text)
        return self.cache[text]


def materialize_new_mcq_rag_package(root: Path, evaluation_root: Path, cache_root: Path) -> None:
    with TemporaryDirectory(dir=root.parent, prefix=".new_mcq-stage-") as directory:
        stage_root = Path(directory) / root.name
        shutil.copytree(root, stage_root)
        _materialize_staged_package(stage_root, evaluation_root, cache_root)
        validate_new_mcq_rag_package(stage_root, evaluation_root)
        _publish_staged_package(stage_root, root)


def _materialize_staged_package(root: Path, evaluation_root: Path, cache_root: Path) -> None:
    (root / "relevance_universe_v1.json").unlink(missing_ok=True)
    provider = _CachedProvider(
        BgeM3EmbeddingProvider(cache_folder=cache_root, local_files_only=True, batch_size=32)
    )
    candidates = {task: _candidates(root, task) for task in TASKS}
    _write_accepted(root, candidates)
    write_json(root / "source_eligibility_registry_v1.json", source_eligibility(root, evaluation_root))
    write_json(root / "embedding_runtime_v1.json", runtime(cache_root, provider.provider), pretty=True)
    selection = authority_selection()
    write_json(root / "authority_selection_v1.json", selection.model_dump(mode="json"), pretty=True)
    registry = _intervention_registry(sha256(root / "authority_selection_v1.json"))
    write_json(root / "intervention_registry_v1.json", registry.model_dump(mode="json"), pretty=True)
    indices = {
        task: _branch_indices(task, candidates[task], registry, provider) for task in TASKS
    }
    (root / "indices").mkdir(exist_ok=True)
    for task, payload in indices.items():
        write_json(root / "indices" / f"{task}.json", payload)
    _write_leakage(root, evaluation_root, registry, provider)
    write_manifest(root, candidates, indices)
    write_status(root)


def _publish_staged_package(stage_root: Path, root: Path) -> None:
    stage_status = stage_root.parent / "new_mcq_rag_status_v1.json"
    live_status = root.parent / "new_mcq_rag_status_v1.json"
    backup_root = stage_root.parent / "previous-package"
    backup_status = stage_root.parent / "previous-status.json"
    root.replace(backup_root)
    try:
        stage_root.replace(root)
        live_status.replace(backup_status)
        stage_status.replace(live_status)
    except OSError:
        if root.exists():
            root.replace(stage_root)
        backup_root.replace(root)
        if backup_status.exists():
            backup_status.replace(live_status)
        raise


def _candidates(root: Path, task: str) -> tuple[Candidate, ...]:
    return tuple(
        Candidate.model_validate_json(line)
        for line in (root / "candidates" / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
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


def _intervention_registry(authority_selection_sha256: str) -> InterventionRegistry:
    roles = {"false": "contam", "correct": "correct", "irrelevant": "irrelevant"}
    tasks: dict[str, JsonValue] = {}
    for task in TASKS:
        documents: dict[str, JsonValue] = {}
        for role, semantic_id, text in semantics("MCQ-H2-DETAIL-LENGTH-v1"):
            branch = roles[role]
            documents[branch] = {
                "document_id": f"new_mcq_h2::{task}::{branch}",
                "task_id": task,
                "role": branch,
                "semantic_id": semantic_id,
                "text": text,
                "source_registry_ids": ["phase13_protocol_revised_v8"],
                "content_hash": canonical_json_hash(text),
            }
        tasks[task] = _JSON_OBJECT.validate_python({
            "selected_candidate_id": "MCQ-H2-DETAIL-LENGTH-v1",
            "candidate_family_status": "ACCEPTED_H2",
            "documents": documents,
        })
    return InterventionRegistry.model_validate(
        _JSON_OBJECT.validate_python({
            "schema_version": "phase13_new_mcq_rag_intervention_registry_v1",
            "protocol_authority_sha256": (
                "022879f559b145e30e645b6ccbd139e9927899d370f1956d27a0562580acf85f"
            ),
            "experiment_authority_sha256": (
                "4b1db4e55e68ec8e00fe022b9bea1685bebb340138df0e39fddc7823aafdc374"
            ),
            "authority_selection_sha256": authority_selection_sha256,
            "authority_stack": [
                "phase13_theory_revised_v1",
                "phase13_baseline_revised_v5",
                "phase13_protocol_revised_v8",
                "phase13_experiment_revised_v8",
            ],
            "tasks": tasks,
        })
    )


def _branch_indices(
    task: str,
    rows: tuple[Candidate, ...],
    registry: InterventionRegistry,
    provider: _CachedProvider,
) -> dict[str, JsonValue]:
    clean = CleanCorpus.from_documents(
        [{"id": row.document_id, "text": row.text} for row in rows],
        corpus_id=f"new_mcq_rag_v1::{task}",
    )
    documents = registry.tasks[task].documents
    corpora = build_branch_corpora(
        clean,
        {
            "false": documents["contam"].model_dump(mode="json"),
            "correct": documents["correct"].model_dump(mode="json"),
            "irrelevant": documents["irrelevant"].model_dump(mode="json"),
        },
    )
    retained = BranchCorpusSet(
        clean,
        {branch: corpora.branches[branch] for branch in BRANCHES},
        corpora.serialization_id,
    )
    indices = build_branch_indices(retained, provider, None).branches
    return _JSON_OBJECT.validate_python(
        {
            "schema_version": "new_mcq_rag_serialized_branch_indices_v1",
            "task_id": task,
            "top_k": 3,
            "branches": {
                branch: {
                    "branch": branch,
                    "corpus_serialization_id": retained.branches[branch].serialization_id,
                    "corpus_content_hash": retained.branches[branch].content_hash,
                    "index_serialization_id": indices[branch].serialization_id,
                    "index_artifact_hash": indices[branch].artifact_hash,
                    "embedding_contract": dict(indices[branch].embedding_contract),
                    "documents": [document.payload() for document in indices[branch].documents],
                    "vectors": {key: list(value) for key, value in indices[branch].vectors.items()},
                }
                for branch in BRANCHES
            },
        }
    )


def _write_leakage(
    root: Path,
    evaluation_root: Path,
    registry: InterventionRegistry,
    provider: _CachedProvider,
) -> None:
    inputs = load_leakage_inputs(root, evaluation_root)
    interventions = tuple(
        AuditDocument(
            document.document_id,
            task,
            document.text,
            document.source_registry_ids,
            (),
        )
        for task, task_registry in registry.tasks.items()
        for document in task_registry.documents.values()
    )
    hashes = dict(inputs.input_hashes)
    hashes["intervention_registry"] = sha256(root / "intervention_registry_v1.json")
    hashes["authority_selection"] = sha256(root / "authority_selection_v1.json")
    artifact = audit_documents(
        inputs.documents + interventions,
        inputs.evaluation_items,
        provider,
        hashes,
    )
    write_leakage_artifact(root / "leakage_report_v1.json", artifact)


__all__ = ["materialize_new_mcq_rag_package"]
