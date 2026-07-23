from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from memcontam.contamination.phase12.models import CandidateTriplet, canonical_json_hash


BRANCH_CORPUS_VERSION = "branch-corpus-v3"
BRANCHES = ("clean", "contam", "correct", "filter", "irrelevant")


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Document:
        document_id = value.get("id", value.get("document_id"))
        text = value.get("text", value.get("content"))
        if not isinstance(document_id, str) or not document_id or not isinstance(text, str) or not text:
            raise ValueError("INVALID_RAG_DOCUMENT")
        return cls(document_id=document_id, text=text)

    def payload(self) -> dict[str, str]:
        return {"id": self.document_id, "text": self.text}


@dataclass(frozen=True)
class CleanCorpus:
    corpus_id: str
    documents: tuple[Document, ...]

    @classmethod
    def from_documents(
        cls, documents: list[Mapping[str, Any]], *, corpus_id: str
    ) -> CleanCorpus:
        parsed = tuple(Document.from_mapping(document) for document in documents)
        if not corpus_id or len({document.document_id for document in parsed}) != len(parsed):
            raise ValueError("INVALID_CLEAN_CORPUS")
        return cls(corpus_id=corpus_id, documents=parsed)

    @property
    def content_hash(self) -> str:
        return canonical_json_hash([document.payload() for document in self.documents])


@dataclass(frozen=True)
class BranchCorpus:
    branch: str
    documents: tuple[Document, ...]
    active_document_ids: tuple[str, ...]
    serialization_id: str
    corpus_version: str = BRANCH_CORPUS_VERSION

    @property
    def active_documents(self) -> tuple[Document, ...]:
        active = set(self.active_document_ids)
        return tuple(document for document in self.documents if document.document_id in active)

    @property
    def content_hash(self) -> str:
        return canonical_json_hash([document.payload() for document in self.active_documents])


@dataclass(frozen=True)
class BranchCorpusSet:
    clean: CleanCorpus
    branches: dict[str, BranchCorpus]
    serialization_id: str

    @property
    def canonical_content_id(self) -> str:
        return canonical_json_hash(
            {branch: corpus.content_hash for branch, corpus in self.branches.items()}
        )


@dataclass(frozen=True)
class MetadataVariantCorpus:
    reference: BranchCorpusSet
    serialization_id: str
    input_surfaces: Mapping[str, Mapping[str, Any]] | None = None

    @classmethod
    def from_reference(
        cls, reference: BranchCorpusSet, *, serialization_id: str
    ) -> MetadataVariantCorpus:
        if not serialization_id or serialization_id == reference.serialization_id:
            raise ValueError("INV03_SERIALIZATION_ID_REQUIRED")
        return cls(reference=reference, serialization_id=serialization_id)

    @property
    def clean(self) -> CleanCorpus:
        return self.reference.clean

    @property
    def branches(self) -> dict[str, BranchCorpus]:
        return self.reference.branches

    @property
    def canonical_content_id(self) -> str:
        return self.reference.canonical_content_id


def build_branch_corpora(
    clean: CleanCorpus, triplet: CandidateTriplet | Mapping[str, Any]
) -> BranchCorpusSet:
    if not isinstance(clean, CleanCorpus):
        raise ValueError("INVALID_CLEAN_CORPUS")
    false = _triplet_document(triplet, "false")
    correct = _triplet_document(triplet, "correct")
    irrelevant = _triplet_document(triplet, "irrelevant")
    clean_documents = clean.documents
    branches = {
        "clean": _branch(clean, "clean", clean_documents, clean_documents),
        "contam": _branch(clean, "contam", (*clean_documents, false), (*clean_documents, false)),
        "correct": _branch(clean, "correct", (*clean_documents, correct), (*clean_documents, correct)),
        "filter": _branch(clean, "filter", (*clean_documents, false), clean_documents),
        "irrelevant": _branch(
            clean, "irrelevant", (*clean_documents, irrelevant), (*clean_documents, irrelevant)
        ),
    }
    return BranchCorpusSet(clean=clean, branches=branches, serialization_id=f"{clean.corpus_id}|base")


def _branch(
    clean: CleanCorpus,
    name: str,
    documents: tuple[Document, ...],
    active_documents: tuple[Document, ...],
) -> BranchCorpus:
    if len({document.document_id for document in documents}) != len(documents):
        raise ValueError("DUPLICATE_RAG_DOCUMENT")
    return BranchCorpus(
        branch=name,
        documents=documents,
        active_document_ids=tuple(document.document_id for document in active_documents),
        serialization_id=f"{clean.corpus_id}|{name}|{BRANCH_CORPUS_VERSION}",
    )


def _triplet_document(triplet: CandidateTriplet | Mapping[str, Any], role: str) -> Document:
    if isinstance(triplet, CandidateTriplet):
        variant = {
            "false": triplet.false_candidate,
            "correct": triplet.correct_twin,
            "irrelevant": triplet.irrelevant_control,
        }[role]
        return Document(document_id=variant.candidate_id, text=variant.content)
    value = triplet.get(role)
    if not isinstance(value, Mapping):
        raise ValueError("INVALID_CANDIDATE_TRIPLET")
    return Document.from_mapping(value)
