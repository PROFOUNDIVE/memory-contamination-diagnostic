from __future__ import annotations

import importlib
import importlib.util

import pytest

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.logging.schema import VerifierResult
from memcontam.memory.embeddings import FakeEmbeddingProvider


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


def test_no_memory_adapter_fails_closed_on_empty_output_and_keeps_memory_read_only() -> None:
    from memcontam.baselines.no_memory import NoMemoryAdapter
    from memcontam.clients.base import LLMResponse
    from memcontam.memory.stores import MemoryState
    from memcontam.tasks.base import TaskInstance

    class Client:
        def chat(self, messages, model, config):
            del messages, model, config
            return LLMResponse(content="   ", raw={}, token_usage={}, latency_ms=0)

    memory = MemoryState()
    outcome = NoMemoryAdapter().execute(
        TaskInstance(sample_id="sample-1", task_name="game24", input={}),
        memory,
        client=Client(),
        model="replay",
    )

    assert outcome.status == "failed"
    assert outcome.failure_disposition == "no_memory_invalid_final_answer"
    assert outcome.memory_before == outcome.memory_after == ()


def test_retrieval_rag_valid_incorrect_answer_is_a_success() -> None:
    from memcontam.baselines.retrieval_rag import RetrievalRagAdapter
    from memcontam.baselines.contracts import CorpusIdentity
    from memcontam.clients.base import LLMResponse
    from memcontam.memory.embeddings import FakeEmbeddingProvider
    from memcontam.memory.stores import MemoryEntry, MemoryState
    from memcontam.tasks.base import TaskInstance

    class Client:
        def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
            return LLMResponse(content="final: wrong", raw={}, token_usage={}, latency_ms=0)

    task = TaskInstance(sample_id="sample-1", task_name="game24", input={"numbers": [1, 3, 4, 6]})
    provider = FakeEmbeddingProvider()
    outcome = RetrievalRagAdapter().execute(
        task,
        MemoryState(
            entries=[
                MemoryEntry(
                    entry_id="rag-strategy",
                    content="Keep exact intermediate values.",
                    memory_type="strategy",
                    clean_or_contaminated="clean",
                    metadata={"source": "fixture"},
                )
            ]
        ),
        client=Client(),
        model="replay",
        config={"_require_corpus_identity": True},
        embedding_provider=provider,
        corpus_identity=CorpusIdentity(
            manifest_id="fixture-corpus",
            corpus_version="v1",
            task_family="game24",
            embedding_provider_identity="fake-deterministic-embedding@local",
        ),
        verifier=lambda answer, seen_task: False,
    )

    assert outcome.status == "succeeded"
    assert outcome.verifier_result is False
    assert outcome.error_type is None
    assert outcome.failure_disposition is None
    assert outcome.memory_write_event is None


def test_reflexion_terminal_incorrect_answer_remains_a_success_after_reflection() -> None:
    from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
    from memcontam.clients.replay import ReplayClient
    from memcontam.memory.stores import MemoryEntry
    from memcontam.tasks.base import TaskInstance

    outcome = ReflexionAdapter().execute(
        TaskInstance(sample_id="sample-1", task_name="game24", input={}),
        ReflexionState(),
        client=ReplayClient(
            responses_by_sample={
                "sample-1": {
                    "reflexion_generate": "final: wrong",
                    "reflexion_reflect": (
                        '{"mode":"corrective","failure_class":"incorrect_answer",'
                        '"reflection_text":"retry","explicitly_used_memory_ids":[]}'
                    ),
                }
            }
        ),
        model="replay",
        config={"run_id": "run-1", "max_attempts": 1},
        verifier=lambda _answer, _task: False,
    )

    assert outcome.status == "succeeded"
    assert outcome.verifier_result is False
    assert outcome.failure_disposition is None
    assert len(outcome.memory_after) == 1
    assert MemoryEntry.model_validate(outcome.memory_after[0]).memory_type == "verbal_reflection"


def test_bot_invalid_solve_returns_closed_failure_without_verifier_or_write() -> None:
    from memcontam.baselines.bot_runtime import BotRuntime
    from memcontam.clients.replay import ReplayClient
    from memcontam.memory.bot_buffer import BotBufferIdentity
    from memcontam.memory.stores import MemoryEntry
    from memcontam.tasks.base import TaskInstance

    task = TaskInstance(
        sample_id="sample-1", task_name="game24", input={"numbers": [1, 2, 3, 4], "target": 24}
    )
    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_problem_distill": (
                    '{"key_information":"numbers = [1, 2, 3, 4]",'
                    '"restrictions":"Use each number once.",'
                    '"distilled_task":"Construct 24."}'
                ),
                "bot_instantiate_solve": '{"final_answer":"24"}',
                "bot_thought_distill": "must not be consumed",
            }
        }
    )
    verifier_calls = []

    def verifier_must_not_run(answer: str) -> VerifierResult:
        verifier_calls.append(answer)
        return VerifierResult(is_correct=False, parsed_answer=answer, reason="must_not_run")

    entry = MemoryEntry(
        entry_id="tpl-1",
        content="Build pairs.",
        memory_type="thought_template",
        clean_or_contaminated="clean",
        metadata={"description": "Build pairs.", "category": "procedure-based"},
    )

    outcome = BotRuntime().run(
        identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
        task=task,
        buffer_snapshot=[entry],
        client=client,
        model="replay",
        config={"sample_id": "sample-1", "embedding_provider": FakeEmbeddingProvider()},
        verifier=verifier_must_not_run,
    )

    assert outcome.status == "failed"
    assert outcome.error_type == "BaselineOutputError"
    assert outcome.failure_disposition == "bot_invalid_solve_result"
    assert outcome.scientific_ineligibility_reason == "invalid_solve_result"
    assert verifier_calls == []
    assert [call.stage for call in outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
    ]
    assert outcome.memory_after == outcome.memory_before
    assert outcome.memory_write_event is None


def test_bot_invalid_thought_distillation_returns_closed_failure_without_verifier_or_write() -> None:
    from memcontam.baselines.bot_runtime import BotRuntime
    from memcontam.clients.replay import ReplayClient
    from memcontam.memory.bot_buffer import BotBufferIdentity
    from memcontam.tasks.base import TaskInstance

    task = TaskInstance(
        sample_id="sample-1", task_name="game24", input={"numbers": [1, 2, 3, 4], "target": 24}
    )
    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_problem_distill": (
                    '{"key_information":"numbers = [1, 2, 3, 4]",'
                    '"restrictions":"Use each number once.",'
                    '"distilled_task":"Construct 24."}'
                ),
                "bot_instantiate_solve": (
                    '{"selected_structure":"procedure-based",'
                    '"solution_trace":"Build pairs.","final_answer":"final: 24"}'
                ),
                "bot_thought_distill": '{"description":""}',
            }
        }
    )
    verifier_calls: list[str] = []

    outcome = BotRuntime().run(
        identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
        task=task,
        buffer_snapshot=[],
        client=client,
        model="replay",
        config={"sample_id": "sample-1", "embedding_provider": FakeEmbeddingProvider()},
        verifier=lambda answer: verifier_calls.append(answer) or True,
    )

    assert outcome.status == "failed"
    assert outcome.error_type == "BaselineOutputError"
    assert outcome.failure_disposition == "bot_invalid_thought_distillation"
    assert outcome.scientific_ineligibility_reason == "invalid_thought_distillation"
    assert outcome.verifier_result is None
    assert verifier_calls == []
    assert outcome.memory_after == outcome.memory_before
    assert outcome.memory_write_event is not None
    assert outcome.memory_write_event["status"] == "rejected_invalid_distillation"
    assert [call.stage for call in outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
    ]
