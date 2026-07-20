from __future__ import annotations

import importlib
import importlib.util

import pytest


EXPECTED_FAILURE_DISPOSITIONS = {
    "no_memory_invalid_final_answer",
    "full_history_invalid_final_answer",
    "rag_invalid_final_answer",
    "rag_retrieval_failed",
    "rag_embedding_failed",
    "rag_manifest_invalid",
    "rag_embedding_dimension_mismatch",
    "rag_embedding_provider_unpinned",
    "bot_invalid_problem_distillation",
    "bot_invalid_solve_result",
    "bot_invalid_thought_distillation",
    "reflexion_invalid_generation",
    "reflexion_invalid_reflection",
    "provider_call_failed",
    "verifier_contract_failed",
}


def test_closed_failure_taxonomy_has_every_canonical_row_and_rejects_unknown_values() -> None:
    assert importlib.util.find_spec("memcontam.baselines.contracts"), (
        "closed failure taxonomy belongs in memcontam.baselines.contracts"
    )
    contracts = importlib.import_module("memcontam.baselines.contracts")

    taxonomy = getattr(contracts, "FAILURE_TAXONOMY", None)
    assert taxonomy is not None
    assert set(taxonomy) == EXPECTED_FAILURE_DISPOSITIONS
    assert callable(getattr(contracts, "validate_failure_triple", None))
    contracts.validate_failure_triple("provider", "provider_call_failed", "provider_failure")
    with pytest.raises(ValueError, match="unknown failure disposition"):
        contracts.validate_failure_triple("provider", "unknown", "provider_failure")
