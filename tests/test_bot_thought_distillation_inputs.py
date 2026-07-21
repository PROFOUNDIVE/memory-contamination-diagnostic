from __future__ import annotations

import importlib
import json

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.clients.replay import ReplayClient
from memcontam.logging.schema import VerifierResult
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import canonical_task_json


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
    )


def _problem():
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    return bot_read.DistilledProblem(
        key_information="numbers = [1, 2, 3, 4], target = 24",
        restrictions="Use every number exactly once.",
        distilled_task="Construct an expression equal to 24.",
    )


def _entry(entry_id: str, *, description: str, template: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        content=template,
        memory_type="thought_template",
        clean_or_contaminated="clean",
        metadata={"description": description, "category": "procedure-based"},
    )


def _thought_output(used_ids: list[str]) -> str:
    return json.dumps(
        {
            "description": "Build useful arithmetic subexpressions before combining them.",
            "template": "Create useful intermediate values, then combine them into the required form.",
            "category": "procedure-based",
            "explicitly_used_memory_ids": used_ids,
        }
    )


class _CapturingClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[dict[str, str]] | None = None

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del model, config
        self.messages = messages
        return LLMResponse(content=self.response, raw={}, token_usage={}, latency_ms=0)


def test_thought_distillation_prompt_contains_only_grounded_inputs() -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    bot_write = importlib.import_module("memcontam.baselines.bot_write")
    visible_memory_type = getattr(bot_write, "VisibleBoTMemory", None)
    assert visible_memory_type is not None
    entry = _entry(
        "template-1",
        description="Build factor pairs before combining them.",
        template="Create factors before combining intermediate values.",
    )
    decision = bot_read.BoTRetrievalDecision("matched", entry, 0.9, 0.7)
    client = _CapturingClient(_thought_output([entry.entry_id]))

    result = bot_write.distill_thought_template(
        canonical_task=canonical_task_json(_task()),
        distilled_problem=_problem(),
        retrieval_decision=decision,
        selected_structure="retrieved-template",
        solution_trace="Make factors 4 and 6, then multiply them.",
        final_answer="final: (1 + 3) * (2 + 4) = 24",
        visible_memory=(
            visible_memory_type(entry.entry_id, entry.metadata["description"], entry.content),
        ),
        client=client,
        model="replay",
        config={},
    )

    assert result.explicitly_used_memory_ids == (entry.entry_id,)
    assert client.messages is not None
    instructions = client.messages[0]["content"]
    prompt = client.messages[1]["content"]
    assert "core-task summary" in instructions
    assert "reusable procedure" in instructions
    assert "general answer form" in instructions
    assert "instance-specific final answer" in instructions
    assert "invent IDs" in instructions
    assert canonical_task_json(_task()) in prompt
    assert json.dumps(_problem().model_dump(), sort_keys=True, separators=(",", ":")) in prompt
    assert 'Retrieval decision JSON:\n{"decision":"matched"}' in prompt
    assert "retrieved-template" in prompt
    assert "Make factors 4 and 6, then multiply them." in prompt
    assert "final: (1 + 3) * (2 + 4) = 24" in prompt
    assert (
        '{"description":"Build factor pairs before combining them.",'
        '"entry_id":"template-1",'
        '"template":"Create factors before combining intermediate values."}'
    ) in prompt


def test_thought_distillation_fallback_renders_empty_memory_and_rejects_ids() -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    bot_write = importlib.import_module("memcontam.baselines.bot_write")
    visible_memory_type = getattr(bot_write, "VisibleBoTMemory", None)
    assert visible_memory_type is not None
    client = _CapturingClient(_thought_output(["not-visible"]))

    with pytest.raises(ValueError, match="unknown explicitly used memory IDs"):
        bot_write.distill_thought_template(
            canonical_task=canonical_task_json(_task()),
            distilled_problem=_problem(),
            retrieval_decision=bot_read.BoTRetrievalDecision("miss", None, 0.4, 0.7),
            selected_structure="procedure-based",
            solution_trace="Apply the selected procedure.",
            final_answer="final: 24",
            visible_memory=(),
            client=client,
            model="replay",
            config={},
        )

    assert client.messages is not None
    instructions = client.messages[0]["content"]
    prompt = client.messages[1]["content"]
    assert "must be []" in instructions
    assert 'Retrieval decision JSON:\n{"decision":"miss"}' in prompt
    assert "Visible memory JSON:\n[]" in prompt
    assert "not-visible" not in prompt


def test_thought_distillation_rejects_all_buffer_ids_not_rendered() -> None:
    runtime = importlib.import_module("memcontam.baselines.bot_runtime").BotRuntime()
    visible = _entry(
        "a-visible",
        description="Build factor pairs before combining them.",
        template="Create factors before combining intermediate values.",
    )
    hidden = _entry(
        "z-hidden",
        description="Do not expose this template.",
        template="This hidden template must not reach the distiller.",
    )

    class _Provider:
        def encode_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

        def encode_document(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_problem_distill": _problem().model_dump_json(),
                "bot_instantiate_solve": json.dumps(
                    {
                        "selected_structure": "retrieved-template",
                        "solution_trace": "Use the visible factor-pair procedure.",
                        "final_answer": "final: 24",
                    }
                ),
                "bot_thought_distill": _thought_output([hidden.entry_id]),
            }
        }
    )
    verifier_calls: list[str] = []

    outcome = runtime.run(
        identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
        task=_task(),
        buffer_snapshot=[visible, hidden],
        client=client,
        model="replay",
        config={"embedding_provider": _Provider()},
        verifier=lambda answer: verifier_calls.append(answer) or True,
    )

    thought_prompt = outcome.method_calls[-1].messages[1]["content"]
    assert outcome.failure_disposition == "bot_invalid_thought_distillation"
    assert verifier_calls == []
    assert outcome.memory_after == outcome.memory_before
    assert "a-visible" in thought_prompt
    assert "z-hidden" not in thought_prompt


def test_thought_distillation_prompt_omits_verifier_data() -> None:
    runtime = importlib.import_module("memcontam.baselines.bot_runtime").BotRuntime()
    task = TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
        verifier_spec={"gold": "TASK_VERIFIER_GOLD"},
    )
    entry = _entry(
        "template-1",
        description="Build factor pairs before combining them.",
        template="Create factors before combining intermediate values.",
    )
    verifier_calls: list[str] = []

    class _Provider:
        def encode_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

        def encode_document(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

    client = ReplayClient(
        responses_by_sample={
            "sample-1": {
                "bot_problem_distill": _problem().model_dump_json(),
                "bot_instantiate_solve": json.dumps(
                    {
                        "selected_structure": "retrieved-template",
                        "solution_trace": "Use the factor-pair procedure.",
                        "final_answer": "final: 24",
                    }
                ),
                "bot_thought_distill": _thought_output([entry.entry_id]),
            }
        }
    )

    def verifier(answer: str) -> VerifierResult:
        verifier_calls.append(answer)
        return VerifierResult(
            is_correct=False,
            parsed_answer="VERIFIER_PARSED_ANSWER",
            reason="VERIFIER_REASON_SECRET",
        )

    outcome = runtime.run(
        identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
        task=task,
        buffer_snapshot=[entry],
        client=client,
        model="replay",
        config={"embedding_provider": _Provider()},
        verifier=verifier,
    )

    thought_prompt = outcome.method_calls[-1].messages[1]["content"]
    assert outcome.status == "succeeded"
    assert verifier_calls == ["24"]
    assert canonical_task_json(task) in thought_prompt
    assert '"entry_id":"template-1"' in thought_prompt
    assert "TASK_VERIFIER_GOLD" not in thought_prompt
    assert "VERIFIER_PARSED_ANSWER" not in thought_prompt
    assert "VERIFIER_REASON_SECRET" not in thought_prompt
