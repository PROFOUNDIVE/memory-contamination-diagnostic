from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.rag.branch_index import EmbeddingProvider
from .phase13_new_mcq_rag_frozen import (
    FrozenRagState,
    load_frozen_clean_state,
    validate_frozen_artifacts,
)
from .phase13_new_mcq_rag_manifest import (
    ManifestEvidence,
    validate_package_manifest,
)
TASKS = ("mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond")
STRATA = (
    "requirement_quantifier_constraint_interpretation",
    "option_wise_evidence_comparison_elimination",
    "contradiction_counterexample_consistency_checking",
    "uncertainty_management_final_answer_verification",
)


class NewMcqRagError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Source(_FrozenModel):
    source_registry_id: str
    source_class: Literal["public_task_specification"]
    repo: str
    revision: str
    path: str
    sha256: str
    allowed_sections: tuple[str, ...]
    evaluation_rows_eligible: Literal[False]
    gold_fields_eligible: Literal[False]


class SourceRegistry(_FrozenModel):
    schema_version: Literal["new_mcq_rag_source_registry_v1"]
    sources: tuple[Source, ...]
    task_sources: dict[str, tuple[str, ...]]


class Candidate(_FrozenModel):
    schema_version: Literal["new_mcq_rag_clean_doc_v1"]
    document_id: str
    task_id: str
    semantic_stratum: str
    document_ordinal: int
    text: str
    source_registry_ids: tuple[str, ...]
    authoring_template_id: Literal["new_mcq_procedural_atomic_v1"]
    review_status: Literal["candidate"]


class Review(_FrozenModel):
    schema_version: Literal["new_mcq_procedural_review_v1"]
    task_id: str
    author_session_id: str
    reviewer_session_id: str
    review_contract_id: Literal["new_mcq_procedural_review_v1"]
    reviewed_document_ids: tuple[str, ...]
    document_verdicts: dict[str, Literal["PASS", "REVISE"]]
    verdict: Literal["PASS", "REVISE"]
    reasons: dict[str, str]


class _EvaluationInput(BaseModel):
    question: str
    options: tuple[str, ...]


class _EvaluationRow(BaseModel):
    input: _EvaluationInput


@dataclass(frozen=True, slots=True)
class NewMcqRagReport:
    status: str
    reason: str
    reviewed_candidates: dict[str, int]
    source_classes: dict[str, str]
    candidate_corpus_hashes: dict[str, str]
    candidate_artifact_hashes: dict[str, str]
    review_artifact_hashes: dict[str, str]
    clean_index_hashes: dict[str, str]
    remaining_objects: tuple[str, ...]
    promotion_ready: bool


@dataclass(frozen=True, slots=True)
class _TaskValidationContext:
    task: str
    registry: SourceRegistry
    sources: dict[str, Source]
    all_texts: set[str]
    evaluation_root: Path


def validate_new_mcq_rag_package(root: Path, evaluation_root: Path) -> NewMcqRagReport:
    registry = SourceRegistry.model_validate_json((root / "source_registry_v1.json").read_bytes())
    sources = {source.source_registry_id: source for source in registry.sources}
    if set(registry.task_sources) != set(TASKS) or len(sources) != len(registry.sources):
        raise NewMcqRagError("NEW_MCQ_RAG_SOURCE_REGISTRY_INVALID")

    reviewed: dict[str, int] = {}
    source_classes: dict[str, str] = {}
    corpus_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    review_hashes: dict[str, str] = {}
    all_texts: set[str] = set()
    for task in TASKS:
        candidate_path = root / "candidates" / f"{task}.jsonl"
        review_path = root / "reviews" / f"{task}.json"
        candidates = tuple(
            Candidate.model_validate_json(line)
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
        )
        review = Review.model_validate_json(review_path.read_bytes())
        _validate_task(
            candidates,
            review,
            _TaskValidationContext(task, registry, sources, all_texts, evaluation_root),
        )
        reviewed[task] = len(candidates)
        source_classes[task] = sources[candidates[0].source_registry_ids[0]].source_class
        corpus_hashes[task] = canonical_json_hash(
            [{"id": row.document_id, "text": row.text} for row in candidates]
        )
        candidate_hashes[task] = _sha256(candidate_path)
        review_hashes[task] = _sha256(review_path)

    try:
        manifest = validate_package_manifest(
            root,
            ManifestEvidence(candidate_hashes, review_hashes, corpus_hashes),
        )
        clean_hashes = validate_frozen_artifacts(root, evaluation_root, corpus_hashes)
        if any(manifest.tasks[task].index_hashes != {"clean": clean_hashes[task]} for task in TASKS):
            raise NewMcqRagError("NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
    except ValueError as error:
        code = getattr(error, "code", "NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE")
        raise NewMcqRagError(code) from error
    return NewMcqRagReport(
        status="NOT_READY",
        reason="NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN",
        reviewed_candidates=reviewed,
        source_classes=source_classes,
        candidate_corpus_hashes=corpus_hashes,
        candidate_artifact_hashes=candidate_hashes,
        review_artifact_hashes=review_hashes,
        clean_index_hashes=clean_hashes,
        remaining_objects=manifest.promotion.remaining_objects,
        promotion_ready=False,
    )


def load_new_mcq_clean_rag_state(
    root: Path,
    evaluation_root: Path,
    task: str,
    embedder: EmbeddingProvider,
    *,
    allow_test_embedder: bool = False,
) -> FrozenRagState:
    validate_new_mcq_rag_package(root, evaluation_root)
    return load_frozen_clean_state(
        root,
        task,
        embedder,
        allow_test_embedder=allow_test_embedder,
    )


def _validate_task(
    candidates: tuple[Candidate, ...],
    review: Review,
    context: _TaskValidationContext,
) -> None:
    task = context.task
    ids = tuple(row.document_id for row in candidates)
    normalized = {" ".join(row.text.casefold().split()) for row in candidates}
    counts = Counter(row.semantic_stratum for row in candidates)
    expected_sources = context.registry.task_sources[task]
    if (
        len(candidates) != 24
        or counts != Counter({stratum: 6 for stratum in STRATA})
        or len(set(ids)) != 24
        or len(normalized) != 24
        or context.all_texts & normalized
        or any(
            row.task_id != task
            or row.document_id
            != f"ragproc_v1::{task}::{row.semantic_stratum}::{row.document_ordinal:02d}"
            or not 1 <= row.document_ordinal <= 6
            or row.source_registry_ids != expected_sources
            or any(source_id not in context.sources for source_id in row.source_registry_ids)
            for row in candidates
        )
    ):
        raise NewMcqRagError("NEW_MCQ_RAG_DOCUMENT_REGISTRY_INVALID")
    _validate_evaluation_overlap(
        candidates,
        context.evaluation_root / f"{task}.jsonl",
    )
    if (
        review.task_id != task
        or review.author_session_id == review.reviewer_session_id
        or review.reviewer_session_id == "pending_parent_binding"
        or review.verdict != "PASS"
        or review.reasons
        or set(review.reviewed_document_ids) != set(ids)
        or review.document_verdicts != {document_id: "PASS" for document_id in ids}
    ):
        raise NewMcqRagError("NEW_MCQ_RAG_REVIEW_INVALID")
    context.all_texts.update(normalized)


def _validate_evaluation_overlap(
    candidates: tuple[Candidate, ...],
    evaluation_path: Path,
) -> None:
    candidate_texts = {" ".join(row.text.casefold().split()) for row in candidates}
    for line in evaluation_path.read_text(encoding="utf-8").splitlines():
        row = _EvaluationRow.model_validate_json(line)
        evaluation_texts = (row.input.question, *row.input.options)
        if any(" ".join(text.casefold().split()) in candidate_texts for text in evaluation_texts):
            raise NewMcqRagError("NEW_MCQ_RAG_EVALUATION_OVERLAP")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "NewMcqRagError",
    "NewMcqRagReport",
    "load_new_mcq_clean_rag_state",
    "validate_new_mcq_rag_package",
]
