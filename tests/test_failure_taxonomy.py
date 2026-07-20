from __future__ import annotations

import importlib
import importlib.util
from typing import get_args

import pytest


EXPECTED_FAILURE_TAXONOMY = {
    "no_memory_invalid_final_answer": ("BaselineOutputError", "invalid_final_answer"),
    "full_history_invalid_final_answer": ("BaselineOutputError", "invalid_final_answer"),
    "rag_invalid_final_answer": ("BaselineOutputError", "invalid_final_answer"),
    "rag_retrieval_failed": ("RetrievalContractError", "retrieval_failed"),
    "rag_embedding_failed": ("EmbeddingContractError", "embedding_failed"),
    "rag_manifest_invalid": ("CorpusContractError", "manifest_invalid"),
    "rag_embedding_dimension_mismatch": (
        "EmbeddingContractError",
        "embedding_dimension_mismatch",
    ),
    "rag_embedding_provider_unpinned": (
        "EmbeddingContractError",
        "embedding_provider_unpinned",
    ),
    "bot_invalid_problem_distillation": (
        "BaselineOutputError",
        "invalid_problem_distillation",
    ),
    "bot_invalid_solve_result": ("BaselineOutputError", "invalid_solve_result"),
    "bot_invalid_thought_distillation": (
        "BaselineOutputError",
        "invalid_thought_distillation",
    ),
    "reflexion_invalid_generation": (
        "BaselineOutputError",
        "invalid_reflexion_generation",
    ),
    "reflexion_invalid_reflection": ("BaselineOutputError", "invalid_reflection"),
    "provider_call_failed": ("ProviderCallFailure", "provider_call_failed"),
    "verifier_contract_failed": ("VerifierContractError", "verifier_contract_failed"),
}


def test_closed_failure_taxonomy_has_every_canonical_row_and_rejects_unknown_values() -> None:
    assert importlib.util.find_spec("memcontam.baselines.contracts"), (
        "closed failure taxonomy belongs in memcontam.baselines.contracts"
    )
    contracts = importlib.import_module("memcontam.baselines.contracts")

    taxonomy = getattr(contracts, "FAILURE_TAXONOMY", None)
    assert taxonomy is not None
    assert taxonomy == EXPECTED_FAILURE_TAXONOMY
    assert get_args(contracts.ErrorType) == (
        "BaselineOutputError",
        "RetrievalContractError",
        "EmbeddingContractError",
        "CorpusContractError",
        "ProviderCallFailure",
        "VerifierContractError",
    )
    assert get_args(contracts.ScientificIneligibilityReason) == (
        "invalid_final_answer",
        "retrieval_failed",
        "embedding_failed",
        "manifest_invalid",
        "embedding_dimension_mismatch",
        "embedding_provider_unpinned",
        "invalid_problem_distillation",
        "invalid_solve_result",
        "invalid_thought_distillation",
        "invalid_reflexion_generation",
        "invalid_reflection",
        "provider_call_failed",
        "verifier_contract_failed",
    )
    assert callable(getattr(contracts, "validate_failure_triple", None))
    contracts.validate_failure_triple(
        "ProviderCallFailure", "provider_call_failed", "provider_call_failed"
    )
    with pytest.raises(ValueError, match="unknown failure disposition"):
        contracts.validate_failure_triple("ProviderCallFailure", "unknown", "provider_call_failed")


def test_logging_validation_rejects_unknown_failure_triples() -> None:
    validation = importlib.import_module("memcontam.logging.validation")
    validate = getattr(validation, "validate_failure_metadata", None)
    assert callable(validate)

    with pytest.raises(ValueError, match="failure triple"):
        validate(
            {
                "error_type": "BaselineOutputError",
                "failure_disposition": "provider_call_failed",
                "scientific_ineligibility_reason": "provider_call_failed",
            }
        )


def test_rag_retrieval_failure_row_is_accepted_only_with_its_exact_triple() -> None:
    contracts = importlib.import_module("memcontam.baselines.contracts")

    contracts.validate_failure_triple(
        "RetrievalContractError", "rag_retrieval_failed", "retrieval_failed"
    )
    with pytest.raises(ValueError, match="failure triple"):
        contracts.validate_failure_triple(
            "EmbeddingContractError", "rag_retrieval_failed", "retrieval_failed"
        )
