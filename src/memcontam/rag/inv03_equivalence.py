from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.rag.branch_index import BranchIndexSet, EmbeddingProvider, build_branch_indices
from memcontam.rag.phase12_corpus import BranchCorpusSet, MetadataVariantCorpus


class Inv03EquivalenceError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Inv03EquivalenceContract:
    branch: str
    reference_artifact_id: str
    variant_artifact_id: str
    canonical_content_id: str
    renderer_input_hash: str
    embedding_input_hash: str
    embedding_vector_hash: str
    query_transform_hash: str
    filter_input_hash: str
    verifier_input_hash: str
    reference_index_hash: str
    variant_index_hash: str
    ranking_contract_hash: str
    id_correspondence: dict[str, str]


@dataclass(frozen=True)
class Inv03EquivalenceRegistry:
    contracts: tuple[Inv03EquivalenceContract, ...]
    registry_id: str = "inv03-equivalence-registry-v1"


def build_inv03_equivalence_registry(
    reference: BranchCorpusSet,
    variants: Sequence[MetadataVariantCorpus],
    embedder: EmbeddingProvider,
    filter_policy: object | None,
    ranking_contract_hash: str,
) -> Inv03EquivalenceRegistry:
    reference_indices = build_branch_indices(reference, embedder, filter_policy)
    contracts = []
    for variant in variants:
        if variant.canonical_content_id != reference.canonical_content_id:
            raise Inv03EquivalenceError("INV03_CANONICAL_CONTENT_MISMATCH")
        variant_indices = build_branch_indices(variant, embedder, filter_policy)
        for branch in ("clean", "contam", "filter"):
            reference_index = reference_indices.branches[branch]
            variant_index = variant_indices.branches[branch]
            document_hash = canonical_json_hash(
                [document.payload() for document in reference_index.documents]
            )
            vector_hash = canonical_json_hash(
                {key: list(value) for key, value in reference_index.vectors.items()}
            )
            contracts.append(
                Inv03EquivalenceContract(
                    branch=branch,
                    reference_artifact_id=f"{reference.serialization_id}|{branch}",
                    variant_artifact_id=f"{variant.serialization_id}|{branch}",
                    canonical_content_id=reference_index.artifact_hash,
                    renderer_input_hash=document_hash,
                    embedding_input_hash=document_hash,
                    embedding_vector_hash=vector_hash,
                    query_transform_hash=document_hash,
                    filter_input_hash=document_hash,
                    verifier_input_hash=document_hash,
                    reference_index_hash=reference_index.artifact_hash,
                    variant_index_hash=variant_index.artifact_hash,
                    ranking_contract_hash=ranking_contract_hash,
                    id_correspondence={
                        f"{reference.serialization_id}|{branch}": reference_index.artifact_hash,
                        f"{variant.serialization_id}|{branch}": reference_index.artifact_hash,
                    },
                )
            )
    return Inv03EquivalenceRegistry(contracts=tuple(contracts))


def validate_inv03_equivalence_registry(
    registry: Inv03EquivalenceRegistry,
    reference: BranchIndexSet,
    variants: Sequence[BranchIndexSet],
) -> None:
    variants_by_id = {variant.serialization_id: variant for variant in variants}
    for contract in registry.contracts:
        reference_index = reference.branches[contract.branch]
        variant_id = contract.variant_artifact_id.rsplit("|", 1)[0]
        variant = variants_by_id.get(variant_id)
        if variant is None:
            raise Inv03EquivalenceError("INV03_VARIANT_INDEX_MISSING")
        variant_index = variant.branches[contract.branch]
        if contract.reference_artifact_id == contract.variant_artifact_id:
            raise Inv03EquivalenceError("INV03_SERIALIZATION_ID_COLLISION")
        if reference_index.vectors != variant_index.vectors:
            raise Inv03EquivalenceError("INV03_EMBEDDING_VECTOR_CHANGED")
        if reference_index.artifact_hash != variant_index.artifact_hash:
            raise Inv03EquivalenceError("INV03_RANKING_CHANGED")
        if tuple(document.document_id for document in reference_index.documents) != tuple(
            document.document_id for document in variant_index.documents
        ):
            raise Inv03EquivalenceError("INV03_RETRIEVED_CONTENT_MISMATCH")
        if contract.reference_index_hash != reference_index.artifact_hash or (
            contract.variant_index_hash != variant_index.artifact_hash
        ):
            raise Inv03EquivalenceError("INV03_EQUIVALENCE_CONTRACT_MISMATCH")
        expected = reference_index.artifact_hash
        if set(contract.id_correspondence.values()) != {expected}:
            raise Inv03EquivalenceError("INV03_CONTENT_ID_CORRESPONDENCE_MISMATCH")
