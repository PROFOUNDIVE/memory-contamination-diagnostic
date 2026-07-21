from __future__ import annotations

import pytest

from memcontam.baselines.common import parse_final_answer
from memcontam.memory.embeddings import FakeEmbeddingProvider


def test_parses_one_case_insensitive_final_line_after_reasoning() -> None:
    assert parse_final_answer("Work through the arithmetic.\nFINAL: 24\n") == "24"


def test_reflexion_accepts_the_shared_terminal_final_answer_form() -> None:
    from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
    from memcontam.clients.replay import ReplayClient
    from memcontam.tasks.base import TaskInstance

    outcome = ReflexionAdapter().execute(
        TaskInstance(sample_id="sample-1", task_name="game24", input={}),
        ReflexionState(),
        client=ReplayClient(responses=["Work through the arithmetic.\nFINAL: 24\n"]),
        model="replay",
        verifier=lambda answer, task: answer == "24" and task.task_name == "game24",
    )

    assert outcome.status == "succeeded"
    assert outcome.parsed_answer == "24"


@pytest.mark.parametrize(
    "response",
    [
        "final: 24\nfinal: 25",
        "final: 24\nMore reasoning",
    ],
)
def test_reflexion_rejects_the_shared_invalid_final_answer_forms(response: str) -> None:
    from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
    from memcontam.clients.replay import ReplayClient
    from memcontam.tasks.base import TaskInstance

    state = ReflexionState()
    outcome = ReflexionAdapter().execute(
        TaskInstance(sample_id="sample-1", task_name="game24", input={}),
        state,
        client=ReplayClient(responses=[response]),
        model="replay",
        verifier=lambda answer, task: answer == "24" and task.task_name == "game24",
    )

    assert outcome.status == "failed"
    assert outcome.failure_disposition == "reflexion_invalid_generation"
    assert outcome.parsed_answer is None
    assert len(outcome.method_calls) == 1
    assert state.reflections == []


@pytest.mark.parametrize(
    "response",
    [
        "answer: 24",
        "final:   ",
        "The answer is final: 24",
        "final: 24\nfinal: 25",
        "final: 24\nMore reasoning",
    ],
)
def test_rejects_responses_without_one_nonempty_terminal_final_line(response: str) -> None:
    with pytest.raises(ValueError):
        parse_final_answer(response)


def test_no_memory_keeps_invalid_final_answer_failure_disposition() -> None:
    from memcontam.baselines.no_memory import NoMemoryAdapter
    from memcontam.clients.base import LLMResponse
    from memcontam.memory.stores import MemoryState
    from memcontam.tasks.base import TaskInstance

    class Client:
        def chat(self, messages, model, config):
            del messages, model, config
            return LLMResponse(content="24", raw={}, token_usage={}, latency_ms=0)

    outcome = NoMemoryAdapter().execute(
        TaskInstance(sample_id="sample-1", task_name="game24", input={}),
        MemoryState(),
        client=Client(),
        model="replay",
    )

    assert outcome.failure_disposition == "no_memory_invalid_final_answer"


def test_rag_keeps_invalid_final_answer_failure_disposition() -> None:
    from memcontam.baselines.retrieval_rag_adapter import RetrievalRagAdapter
    from memcontam.clients.base import LLMResponse
    from memcontam.memory.embeddings import FakeEmbeddingProvider
    from memcontam.memory.stores import MemoryState
    from memcontam.tasks.base import TaskInstance

    class Client:
        def chat(self, messages, model, config):
            del messages, model, config
            return LLMResponse(content="24", raw={}, token_usage={}, latency_ms=0)

    outcome = RetrievalRagAdapter().execute(
        TaskInstance(sample_id="sample-1", task_name="game24", input={}),
        MemoryState(),
        client=Client(),
        model="replay",
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert outcome.failure_disposition == "rag_invalid_final_answer"


def test_bot_keeps_invalid_solve_failure_disposition_for_invalid_final_answer() -> None:
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
                    '"solution_trace":"Build pairs.","final_answer":"24"}'
                ),
            }
        }
    )

    outcome = BotRuntime().run(
        identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
        task=task,
        buffer_snapshot=[],
        client=client,
        model="replay",
        config={"sample_id": "sample-1", "embedding_provider": FakeEmbeddingProvider()},
    )

    assert outcome.failure_disposition == "bot_invalid_solve_result"
    assert outcome.metadata["selected_structure"] == "procedure-based"
