from __future__ import annotations

from memcontam.baselines.no_memory import NoMemoryAdapter
from memcontam.clients.base import LLMResponse
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


def test_rejects_unmarked_final_answer_without_mutating_memory() -> None:
    class Client:
        def chat(self, messages, model, config):
            del messages, model, config
            return LLMResponse(content="24", raw={}, token_usage={}, latency_ms=0)

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
