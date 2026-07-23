from __future__ import annotations

from dataclasses import replace

import pytest

from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.inv03_equivalence import (
    Inv03EquivalenceError,
    build_inv03_equivalence_registry,
    validate_inv03_equivalence_registry,
)
from memcontam.rag.phase12_corpus import CleanCorpus, MetadataVariantCorpus, build_branch_corpora


class Embedder:
    embedding_contract = {
        "dimension": 2,
        "normalized": True,
        "production_identity": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
        "provider": "deterministic-test-double",
    }

    def encode_document(self, text: str) -> list[float]:
        return [1.0, 0.0] if text == "clean" else [0.0, 1.0]

    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_requires_mechanical_reference_variant_equivalence() -> None:
    reference = build_branch_corpora(
        CleanCorpus.from_documents([{"id": "clean", "text": "clean"}], corpus_id="clean-v1"),
        {
            "false": {"id": "false", "text": "false"},
            "correct": {"id": "correct", "text": "correct"},
            "irrelevant": {"id": "irrelevant", "text": "irrelevant"},
        },
    )
    variant = MetadataVariantCorpus.from_reference(
        reference,
        serialization_id="game24|behavior-inv03|metadata-variant-v1",
    )
    embedder = Embedder()
    reference_indices = build_branch_indices(reference, embedder, filter_policy=None)
    variant_indices = build_branch_indices(variant, embedder, filter_policy=None)
    registry = build_inv03_equivalence_registry(
        reference, (variant,), embedder, filter_policy=None, ranking_contract_hash="ranking-v1"
    )

    validate_inv03_equivalence_registry(registry, reference_indices, (variant_indices,))
    contract = registry.contracts[0]
    assert contract.reference_artifact_id != contract.variant_artifact_id
    assert contract.reference_index_hash == contract.variant_index_hash
    assert contract.id_correspondence[contract.variant_artifact_id] == contract.canonical_content_id

    changed = replace(
        variant_indices.branches["clean"],
        vectors={"clean": (0.0, 1.0)},
    )
    broken = replace(variant_indices, branches={**variant_indices.branches, "clean": changed})
    with pytest.raises(Inv03EquivalenceError, match="INV03_EMBEDDING_VECTOR_CHANGED"):
        validate_inv03_equivalence_registry(registry, reference_indices, (broken,))
