from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

SEMANTIC_THRESHOLD: Final = 0.90
LEXICAL_THRESHOLD: Final = Fraction(1, 2)
STRUCTURAL_THRESHOLD: Final = Fraction(9, 10)
NORMALIZER_ID: Final = "MCQ-NORM-NFKC-CASEFOLD-WS-v1"
STRUCTURAL_ID: Final = "MCQ-STRUCT-TOKEN-MASK-LEVENSHTEIN-v1"
ARTIFACT_SCHEMA: Final = "new_mcq_rag_leakage_artifact_v1"


@dataclass(frozen=True, slots=True)
class LeakageArtifactError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class McqContent:
    stem: str
    options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditDocument:
    document_id: str
    task_id: str
    text: str
    source_span_ids: tuple[str, ...]
    identity_keys: tuple[str, ...]
    mcq: McqContent | None = None


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    task_id: str
    evaluation_id: str
    stem: str
    options: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    identity_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    semantic_cosine: float
    lexical_intersection: int
    lexical_union: int
    structural_distance: int
    structural_length: int

    @property
    def lexical_jaccard(self) -> float:
        return self.lexical_intersection / self.lexical_union if self.lexical_union else 0.0

    @property
    def structural_similarity(self) -> float:
        if not self.structural_length:
            return 1.0
        return 1.0 - (self.structural_distance / self.structural_length)


@dataclass(frozen=True, slots=True)
class PairEvidence:
    evaluation_id: str
    exact_identity: bool
    canonical_identity: bool
    permutation_identity: bool
    excluded_source_span_ids: tuple[str, ...]
    metrics: MetricEvidence
    failed_components: tuple[str, ...]

    @property
    def failed(self) -> bool:
        return bool(self.failed_components)


@dataclass(frozen=True, slots=True)
class DocumentEvidence:
    document_id: str
    task_id: str
    source_span_ids: tuple[str, ...]
    evaluated_item_count: int
    maximum_semantic_cosine: float
    maximum_semantic_evaluation_id: str
    maximum_lexical_jaccard: float
    maximum_lexical_evaluation_id: str
    maximum_lexical_intersection: int
    maximum_lexical_union: int
    maximum_structural_similarity: float
    maximum_structural_evaluation_id: str
    maximum_structural_distance: int
    maximum_structural_length: int
    exact_identity_ids: tuple[str, ...]
    canonical_identity_ids: tuple[str, ...]
    permutation_identity_ids: tuple[str, ...]
    excluded_source_span_ids: tuple[str, ...]
    offending_evaluation_ids: tuple[str, ...]
    failed_components: tuple[str, ...]

    @property
    def failed(self) -> bool:
        return bool(self.failed_components)


@dataclass(frozen=True, slots=True)
class LeakageArtifact:
    schema_version: str
    status: str
    normalizer_id: str
    structural_representation_id: str
    semantic_threshold: float
    lexical_threshold: float
    structural_threshold: float
    embedding_identity: str
    input_hashes: tuple[tuple[str, str], ...]
    document_evidence: tuple[DocumentEvidence, ...]
    artifact_hash: str


__all__ = [
    "ARTIFACT_SCHEMA",
    "LEXICAL_THRESHOLD",
    "NORMALIZER_ID",
    "SEMANTIC_THRESHOLD",
    "STRUCTURAL_ID",
    "STRUCTURAL_THRESHOLD",
    "AuditDocument",
    "DocumentEvidence",
    "EvaluationItem",
    "LeakageArtifact",
    "LeakageArtifactError",
    "McqContent",
    "MetricEvidence",
    "PairEvidence",
]
