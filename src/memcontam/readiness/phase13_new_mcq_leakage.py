from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from fractions import Fraction
from typing import Mapping, Protocol

from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.readiness.phase13_new_mcq_candidate import mcq_normalize, mcq_tokens
from memcontam.readiness.phase13_new_mcq_leakage_metrics import (
    answer_free_item_text,
    cosine,
    levenshtein,
    mcq_identity,
    structural_representation,
)
from memcontam.readiness.phase13_new_mcq_leakage_models import (
    ARTIFACT_SCHEMA,
    LEXICAL_THRESHOLD,
    NORMALIZER_ID,
    SEMANTIC_THRESHOLD,
    STRUCTURAL_ID,
    STRUCTURAL_THRESHOLD,
    AuditDocument,
    DocumentEvidence,
    EvaluationItem,
    LeakageArtifact,
    LeakageArtifactError,
    McqContent,
    MetricEvidence,
    PairEvidence,
)
class EmbeddingProvider(Protocol):
    @property
    def metadata(self) -> Mapping[str, str | int | bool]: ...

    def encode_document(self, text: str) -> list[float]: ...


def failed_thresholds(metrics: MetricEvidence) -> tuple[str, ...]:
    failures: list[str] = []
    if metrics.semantic_cosine >= SEMANTIC_THRESHOLD:
        failures.append("semantic")
    lexical = Fraction(metrics.lexical_intersection, metrics.lexical_union or 1)
    if lexical >= LEXICAL_THRESHOLD:
        failures.append("lexical")
    structural = Fraction(
        metrics.structural_length - metrics.structural_distance,
        metrics.structural_length or 1,
    )
    if structural >= STRUCTURAL_THRESHOLD:
        failures.append("structural")
    return tuple(failures)


def compare_document_to_item(
    document: AuditDocument,
    item: EvaluationItem,
    document_vector: tuple[float, ...],
    item_vector: tuple[float, ...],
) -> PairEvidence:
    item_text = answer_free_item_text(item.stem, item.options)
    exact = document.text in {item.stem, *item.options, item_text}
    canonical_parts = {mcq_normalize(item.stem), *(mcq_normalize(value) for value in item.options)}
    canonical = mcq_normalize(document.text) in canonical_parts | {item_text}
    permutation = document.mcq is not None and mcq_identity(document.mcq) == mcq_identity(
        McqContent(item.stem, item.options)
    )
    identity = bool(set(document.identity_keys) & set(item.identity_keys))
    excluded = tuple(sorted(set(document.source_span_ids) & set(item.source_span_ids)))
    document_structure = structural_representation(document.mcq or document.text)
    item_structure = structural_representation(McqContent(item.stem, item.options))
    document_tokens = frozenset(mcq_tokens(document.text))
    item_tokens = frozenset(mcq_tokens(item_text))
    metrics = MetricEvidence(
        semantic_cosine=cosine(document_vector, item_vector),
        lexical_intersection=len(document_tokens & item_tokens),
        lexical_union=len(document_tokens | item_tokens),
        structural_distance=levenshtein(document_structure, item_structure),
        structural_length=max(len(document_structure), len(item_structure)),
    )
    failures = [
        name
        for name, matched in (
            ("exact", exact),
            ("canonical", canonical or identity),
            ("permutation", permutation),
            ("source_span", bool(excluded)),
        )
        if matched
    ]
    failures.extend(failed_thresholds(metrics))
    return PairEvidence(
        item.evaluation_id,
        exact,
        canonical or identity,
        permutation,
        excluded,
        metrics,
        tuple(failures),
    )


def audit_documents(
    documents: tuple[AuditDocument, ...],
    evaluation_items: tuple[EvaluationItem, ...],
    provider: EmbeddingProvider,
    input_hashes: Mapping[str, str],
) -> LeakageArtifact:
    identity = _embedding_identity(provider)
    item_vectors = {
        item.evaluation_id: tuple(provider.encode_document(answer_free_item_text(item.stem, item.options)))
        for item in evaluation_items
    }
    evidence: list[DocumentEvidence] = []
    for document in documents:
        vector = tuple(provider.encode_document(mcq_normalize(document.text)))
        pairs = tuple(
            compare_document_to_item(document, item, vector, item_vectors[item.evaluation_id])
            for item in evaluation_items
            if item.task_id == document.task_id
        )
        evidence.append(_summarize(document, pairs))
    unsigned = LeakageArtifact(
        ARTIFACT_SCHEMA,
        "FAIL" if any(row.failed for row in evidence) else "PASS",
        NORMALIZER_ID,
        STRUCTURAL_ID,
        SEMANTIC_THRESHOLD,
        float(LEXICAL_THRESHOLD),
        float(STRUCTURAL_THRESHOLD),
        identity,
        tuple(sorted(input_hashes.items())),
        tuple(evidence),
        "",
    )
    return replace(unsigned, artifact_hash=_artifact_hash(unsigned))


def validate_leakage_artifact(artifact: LeakageArtifact) -> None:
    expected_status = "FAIL" if any(row.failed for row in artifact.document_evidence) else "PASS"
    valid = (
        artifact.schema_version == ARTIFACT_SCHEMA
        and artifact.status == expected_status
        and artifact.normalizer_id == NORMALIZER_ID
        and artifact.structural_representation_id == STRUCTURAL_ID
        and artifact.semantic_threshold == SEMANTIC_THRESHOLD
        and artifact.lexical_threshold == float(LEXICAL_THRESHOLD)
        and artifact.structural_threshold == float(STRUCTURAL_THRESHOLD)
        and artifact.artifact_hash == _artifact_hash(artifact)
    )
    if not valid:
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_ARTIFACT_HASH_MISMATCH")


def _summarize(document: AuditDocument, pairs: tuple[PairEvidence, ...]) -> DocumentEvidence:
    if not pairs:
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_TASK_EVALUATION_MISSING")
    semantic = max(pairs, key=lambda row: (row.metrics.semantic_cosine, row.evaluation_id))
    lexical = max(pairs, key=lambda row: (row.metrics.lexical_jaccard, row.evaluation_id))
    structural = max(pairs, key=lambda row: (row.metrics.structural_similarity, row.evaluation_id))
    return DocumentEvidence(
        document.document_id,
        document.task_id,
        document.source_span_ids,
        len(pairs),
        semantic.metrics.semantic_cosine,
        semantic.evaluation_id,
        lexical.metrics.lexical_jaccard,
        lexical.evaluation_id,
        lexical.metrics.lexical_intersection,
        lexical.metrics.lexical_union,
        structural.metrics.structural_similarity,
        structural.evaluation_id,
        structural.metrics.structural_distance,
        structural.metrics.structural_length,
        tuple(row.evaluation_id for row in pairs if row.exact_identity),
        tuple(row.evaluation_id for row in pairs if row.canonical_identity),
        tuple(row.evaluation_id for row in pairs if row.permutation_identity),
        tuple(sorted({span for row in pairs for span in row.excluded_source_span_ids})),
        tuple(row.evaluation_id for row in pairs if row.failed),
        tuple(sorted({code for row in pairs for code in row.failed_components})),
    )


def _embedding_identity(provider: EmbeddingProvider) -> str:
    metadata = provider.metadata
    valid = (
        metadata.get("model_id") == BgeM3EmbeddingProvider.MODEL_ID
        and metadata.get("revision") == BgeM3EmbeddingProvider.REVISION
        and metadata.get("vector_dimension") == BgeM3EmbeddingProvider.VECTOR_DIMENSION
        and metadata.get("normalize_embeddings") is True
    )
    if not valid:
        raise LeakageArtifactError("NEW_MCQ_LEAKAGE_BGE_M3_CONTRACT_REQUIRED")
    return f"{BgeM3EmbeddingProvider.MODEL_ID}@{BgeM3EmbeddingProvider.REVISION}"


def _artifact_hash(artifact: LeakageArtifact) -> str:
    payload = asdict(replace(artifact, artifact_hash=""))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AuditDocument",
    "EvaluationItem",
    "LeakageArtifact",
    "LeakageArtifactError",
    "McqContent",
    "MetricEvidence",
    "audit_documents",
    "compare_document_to_item",
    "failed_thresholds",
    "structural_representation",
    "validate_leakage_artifact",
]
