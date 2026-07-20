from __future__ import annotations

import importlib
import importlib.util

import pytest

from memcontam.baselines.contracts import BaselineExecutionOutcome


def test_baseline_execution_outcome_retains_failed_evidence_and_valid_incorrect_success() -> None:
    assert importlib.util.find_spec("memcontam.baselines.contracts"), (
        "BaselineExecutionOutcome must live in the shared contracts module"
    )
    contracts = importlib.import_module("memcontam.baselines.contracts")

    outcome = getattr(contracts, "BaselineExecutionOutcome", None)
    assert outcome is not None
    assert getattr(outcome, "__dataclass_fields__", {}).keys() >= {
        "status",
        "final_response",
        "parsed_answer",
        "verifier_result",
        "answer_call_id",
        "method_calls",
        "memory_before",
        "memory_after",
        "retrieved_memory",
        "retrieved_scores",
        "memory_write_event",
        "error_type",
        "failure_disposition",
        "metadata",
    }

    incorrect = outcome(status="succeeded", verifier_result=False)
    assert incorrect.verifier_result is False
    assert incorrect.error_type is None
    assert incorrect.failure_disposition is None

    with pytest.raises(ValueError, match="succeeded"):
        outcome(status="succeeded", error_type="ProviderCallFailure")

    with pytest.raises(ValueError, match="complete failure triple"):
        outcome(status="failed")

    failed = outcome(
        status="failed",
        method_calls=(),
        memory_before=(),
        memory_after=(),
        retrieved_memory=(),
        retrieved_scores=(),
        error_type="ProviderCallFailure",
        failure_disposition="provider_call_failed",
        scientific_ineligibility_reason="provider_call_failed",
    )
    assert failed.method_calls == ()
    assert failed.memory_before == ()
    with pytest.raises(ValueError, match="failure triple"):
        outcome(
            status="failed",
            error_type="BaselineOutputError",
            failure_disposition="provider_call_failed",
            scientific_ineligibility_reason="provider_call_failed",
        )

    provider_failure = contracts.ProviderCallFailure(error_type="ProviderCallFailure")
    assert provider_failure.failure_disposition == "provider_call_failed"
    with pytest.raises(ValueError, match="failure triple"):
        contracts.ProviderCallFailure(error_type="BaselineOutputError")


def test_failed_outcome_requires_scientific_ineligibility_metadata() -> None:
    validation = importlib.import_module("memcontam.logging.validation")
    validate = getattr(validation, "validate_outcome_metadata", None)
    assert callable(validate)

    failed = BaselineExecutionOutcome(
        status="failed",
        error_type="ProviderCallFailure",
        failure_disposition="provider_call_failed",
        scientific_ineligibility_reason="provider_call_failed",
    )
    with pytest.raises(ValueError, match="scientific_ineligibility_reason"):
        validate(failed, {})


def test_full_history_parse_failure_uses_the_closed_output_failure_row() -> None:
    failed = BaselineExecutionOutcome(
        status="failed",
        error_type="BaselineOutputError",
        failure_disposition="full_history_invalid_final_answer",
        scientific_ineligibility_reason="invalid_final_answer",
    )

    assert failed.failure_disposition == "full_history_invalid_final_answer"


def test_retrieval_rag_valid_incorrect_answer_is_a_success() -> None:
    from memcontam.baselines.retrieval_rag import RetrievalRagAdapter
    from memcontam.clients.base import LLMResponse
    from memcontam.memory.embeddings import FakeEmbeddingProvider
    from memcontam.memory.stores import MemoryState
    from memcontam.tasks.base import TaskInstance

    class Client:
        def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
            return LLMResponse(content="final: wrong", raw={}, token_usage={}, latency_ms=0)

    task = TaskInstance(sample_id="sample-1", task_name="game24", input={"numbers": [1, 3, 4, 6]})
    outcome = RetrievalRagAdapter().execute(
        task,
        MemoryState(),
        client=Client(),
        model="replay",
        embedding_provider=FakeEmbeddingProvider(),
        verifier=lambda answer, seen_task: False,
    )

    assert outcome.status == "succeeded"
    assert outcome.verifier_result is False
    assert outcome.error_type is None
    assert outcome.failure_disposition is None
    assert outcome.memory_write_event is None
