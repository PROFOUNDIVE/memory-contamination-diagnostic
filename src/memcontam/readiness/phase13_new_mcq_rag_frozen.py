from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.rag.branch_index import BranchIndex, EmbeddingProvider
from memcontam.rag.phase12_corpus import BranchCorpus, Document
from memcontam.readiness.phase13_new_mcq_bge import validate_runtime_artifact, verify_runtime_binding
from memcontam.readiness.phase13_new_mcq_candidate_evidence_v2_models import semantics
from memcontam.readiness.phase13_new_mcq_rag_authority import authority_selection
from .phase13_new_mcq_rag_index_validation import (
    load_serialized_indices,
    validate_serialized_branch,
    validate_serialized_bundle,
)
from memcontam.readiness.phase13_new_mcq_rag_models import (
    BRANCHES,
    EXPECTED_CLASSES,
    TASKS,
    AcceptedDocument,
    AuthoritySelection,
    BranchName,
    FrozenArtifactError,
    InterventionRegistry,
    SerializedIndexBundle,
)


@dataclass(frozen=True, slots=True)
class FrozenRagState:
    state: RagFrozenStateV3
    index_artifact_hash: str
    reconstruction_identity: str


def validate_frozen_artifacts(
    root: Path,
    evaluation_root: Path,
    expected_corpus_hashes: dict[str, str],
) -> dict[str, dict[BranchName, str]]:
    from memcontam.readiness.phase13_new_mcq_leakage_io import (
        load_leakage_artifact,
        load_leakage_inputs,
    )

    accepted = {task: _accepted(root, task) for task in TASKS}
    if any(
        canonical_json_hash([{"id": row.document_id, "text": row.text} for row in accepted[task]])
        != expected_corpus_hashes[task]
        for task in TASKS
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_ACCEPTED_DOCUMENT_REGISTRY_INVALID")
    _validate_source_eligibility(root)
    validate_runtime_artifact(root)
    _authority_selection(root)
    interventions = _interventions(root)
    leakage = load_leakage_artifact(root / "leakage_report_v1.json")
    leakage_inputs = load_leakage_inputs(root, evaluation_root)
    expected_hashes = dict(leakage_inputs.input_hashes)
    expected_hashes["intervention_registry"] = _sha256(root / "intervention_registry_v1.json")
    expected_hashes["authority_selection"] = _sha256(root / "authority_selection_v1.json")
    expected_document_ids = {
        document.document_id for document in leakage_inputs.documents
    } | {
        document.document_id
        for task in interventions.tasks.values()
        for document in task.documents.values()
    }
    if (
        leakage.status != "PASS"
        or dict(leakage.input_hashes) != expected_hashes
        or {row.document_id for row in leakage.document_evidence} != expected_document_ids
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_LEAKAGE_AUDIT_INVALID")
    hashes: dict[str, dict[BranchName, str]] = {}
    for task in TASKS:
        bundle = load_serialized_indices(root, task)
        validate_serialized_bundle(bundle, accepted[task], interventions.tasks[task])
        hashes[task] = {
            branch: bundle.branches[branch].index_artifact_hash for branch in BRANCHES
        }
    return hashes


def load_frozen_rag_state(
    root: Path,
    task: str,
    branch: BranchName,
    embedder: EmbeddingProvider,
    *,
    allow_test_embedder: bool = False,
) -> FrozenRagState:
    return _load_frozen_rag_state(root, task, branch, embedder, allow_test_embedder, True)


def load_frozen_clean_state(
    root: Path,
    task: str,
    embedder: EmbeddingProvider,
    *,
    allow_test_embedder: bool = False,
) -> FrozenRagState:
    return load_frozen_rag_state(
        root, task, "clean", embedder, allow_test_embedder=allow_test_embedder
    )


def _load_frozen_clean_state_for_test(
    root: Path, task: str, embedder: EmbeddingProvider
) -> FrozenRagState:
    return _load_frozen_rag_state(root, task, "clean", embedder, True, False)


def _load_frozen_rag_state_for_test(
    root: Path, task: str, branch: BranchName, embedder: EmbeddingProvider
) -> FrozenRagState:
    return _load_frozen_rag_state(root, task, branch, embedder, True, False)


def _load_frozen_rag_state(
    root: Path,
    task: str,
    branch: BranchName,
    embedder: EmbeddingProvider,
    allow_test_embedder: bool,
    verify_snapshot: bool,
) -> FrozenRagState:
    metadata = getattr(embedder, "metadata", {})
    if (
        task not in TASKS
        or metadata.get("model_id") != BgeM3EmbeddingProvider.MODEL_ID
        or metadata.get("revision") != BgeM3EmbeddingProvider.REVISION
        or metadata.get("vector_dimension") != BgeM3EmbeddingProvider.VECTOR_DIMENSION
        or metadata.get("normalize_embeddings") is not True
        or (not allow_test_embedder and not isinstance(embedder, BgeM3EmbeddingProvider))
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_RUNTIME_IDENTITY_INVALID")
    if verify_snapshot:
        if not isinstance(embedder, BgeM3EmbeddingProvider):
            raise FrozenArtifactError("NEW_MCQ_RAG_RUNTIME_SNAPSHOT_UNVERIFIED")
        verify_runtime_binding(root, embedder)
    serialized = load_serialized_indices(root, task).branches[branch]
    validate_serialized_branch(
        task,
        branch,
        serialized,
        _accepted(root, task),
        _interventions(root).tasks[task],
    )
    documents = tuple(Document.from_mapping(row) for row in serialized.documents)
    corpus = BranchCorpus(
        branch, documents, tuple(row.document_id for row in documents), serialized.corpus_serialization_id
    )
    index = BranchIndex(
        branch,
        documents,
        serialized.embedding_contract,
        serialized.vectors,
        serialized.index_serialization_id,
        embedder,
    )
    identity = canonical_json_hash(
        {"task": task, "branch": branch, "index": index.artifact_hash, "schema": "v1"}
    )
    return FrozenRagState(RagFrozenStateV3(branch, corpus, index), index.artifact_hash, identity)


def _accepted(root: Path, task: str) -> tuple[AcceptedDocument, ...]:
    rows = tuple(
        AcceptedDocument.model_validate_json(line)
        for line in (root / "accepted" / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
    )
    if len(rows) != 24 or len({row.document_id for row in rows}) != 24 or any(
        row.task_id != task or row.content_hash != canonical_json_hash(row.text) for row in rows
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_ACCEPTED_DOCUMENT_REGISTRY_INVALID")
    return rows


def _interventions(root: Path) -> InterventionRegistry:
    registry = InterventionRegistry.model_validate_json(
        (root / "intervention_registry_v1.json").read_bytes()
    )
    roles = {"false": "contam", "correct": "correct", "irrelevant": "irrelevant"}
    expected = {
        roles[role]: (semantic_id, text)
        for role, semantic_id, text in semantics("MCQ-H2-DETAIL-LENGTH-v1")
    }
    if (
        registry.authority_selection_sha256 != _sha256(root / "authority_selection_v1.json")
        or set(registry.tasks) != set(TASKS)
        or registry.authority_stack != (
        "phase13_theory_revised_v1",
        "phase13_baseline_revised_v5",
        "phase13_protocol_revised_v8",
        "phase13_experiment_revised_v8",
        )
    ):
        raise FrozenArtifactError("NEW_MCQ_RAG_INTERVENTION_REGISTRY_INVALID")
    for task, task_registry in registry.tasks.items():
        if (
            set(task_registry.documents) != set(expected)
            or any(
            document.task_id != task
            or document.role != role
            or (document.semantic_id, document.text) != expected[role]
            or document.source_registry_ids != ("phase13_protocol_revised_v8",)
            or document.content_hash != canonical_json_hash(document.text)
            for role, document in task_registry.documents.items()
            )
        ):
            raise FrozenArtifactError("NEW_MCQ_RAG_INTERVENTION_REGISTRY_INVALID")
    return registry


def _authority_selection(root: Path) -> AuthoritySelection:
    selection = AuthoritySelection.model_validate_json(
        (root / "authority_selection_v1.json").read_bytes()
    )
    if selection != authority_selection():
        raise FrozenArtifactError("NEW_MCQ_RAG_AUTHORITY_SELECTION_INVALID")
    return selection


def _validate_source_eligibility(root: Path) -> None:
    source = json.loads((root / "source_eligibility_registry_v1.json").read_bytes())
    if source.get("status") != "COMPLETE" or set(source.get("tasks", {})) != set(TASKS):
        raise FrozenArtifactError("NEW_MCQ_RAG_SOURCE_ELIGIBILITY_INVALID")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "BRANCHES",
    "EXPECTED_CLASSES",
    "AcceptedDocument",
    "BranchName",
    "FrozenArtifactError",
    "FrozenRagState",
    "InterventionRegistry",
    "SerializedIndexBundle",
    "load_frozen_clean_state",
    "load_frozen_rag_state",
    "validate_frozen_artifacts",
]
