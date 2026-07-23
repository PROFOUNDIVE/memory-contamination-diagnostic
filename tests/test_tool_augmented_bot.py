from __future__ import annotations

import json
from pathlib import Path

from memcontam.baselines.bot_runtime import BotRuntime
from memcontam.clients.replay import ReplayClient
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.game24 import build_instance
from memcontam.tools import SubprocessTestDouble, load_tool_runtime_contract
from memcontam.verifiers.game24 import verify_expression


ROOT = Path(__file__).resolve().parents[1]


class _EmbeddingProvider:
    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _RetrievedEmbeddingProvider(_EmbeddingProvider):
    def encode_document(self, text: str) -> list[float]:
        return [1.0, 0.0] if text.startswith("Validate") else [0.0, 1.0]


def _task():
    return build_instance({"sample_id": "game24-tool", "numbers": [1, 3, 4, 6], "target": 24})


def _problem() -> str:
    return json.dumps(
        {
            "key_information": "numbers = [1, 3, 4, 6], target = 24",
            "restrictions": "Use every number exactly once.",
            "distilled_task": "Construct an expression equal to 24.",
        }
    )


def _solve_result() -> str:
    return json.dumps(
        {
            "selected_structure": "programming-based",
            "solution_trace": "The executed calculation produced 24.",
            "final_answer": "final: 6 / (1 - 3 / 4)",
        }
    )


def _thought_result() -> str:
    return json.dumps(
        {
            "description": "Validate a candidate arithmetic expression before returning it.",
            "template": "Execute the expression, then return the checked expression in final form.",
            "category": "programming-based",
            "explicitly_used_memory_ids": [],
        }
    )


def _tool_config(executor: SubprocessTestDouble) -> dict:
    return {
        "embedding_provider": _EmbeddingProvider(),
        "tool_mode": "python_sandbox",
        "tool_executor": executor,
        "tool_runtime_contract": load_tool_runtime_contract(
            ROOT / "containers" / "python-sandbox" / "image.lock.json", scientific=False
        ),
    }


def _runtime() -> BotRuntime:
    return BotRuntime()


def _identity() -> BotBufferIdentity:
    return BotBufferIdentity("tool-run", "game24", "bot_style", "clean", "replay")


class _CapturingReplayClient(ReplayClient):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.configs: list[dict] = []

    def chat(self, messages, model, config):  # noqa: ANN001
        self.configs.append(dict(config))
        return super().chat(messages, model, config)


def test_executes_and_distills_reusable_programming_template() -> None:
    executor = SubprocessTestDouble()
    code = "print(6 / (1 - 3 / 4))"
    outcome = _runtime().run(
        identity=_identity(),
        task=_task(),
        buffer_snapshot=[],
        client=ReplayClient(
            responses_by_sample={
                "game24-tool": {
                    "bot_problem_distill": _problem(),
                    "bot_instantiate_solve": [
                        json.dumps({"action": "execute_python", "code": code}),
                        json.dumps({"action": "final", "answer": _solve_result()}),
                    ],
                    "bot_thought_distill": _thought_result(),
                }
            }
        ),
        model="replay",
        config=_tool_config(executor),
        verifier=lambda answer: verify_expression(answer, [1, 3, 4, 6]),
    )

    assert outcome.status == "succeeded"
    assert [call.stage for call in outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_instantiate_solve",
        "bot_thought_distill",
    ]
    assert outcome.answer_call_id == outcome.method_calls[2].call_id
    assert executor.execution_count == 1
    assert len(outcome.metadata["tool_events"]) == 1
    assert outcome.metadata["tool_events"][0].output == "24.0\n"
    assert outcome.metadata["executed_trajectory"] == [
        {
            "code": code,
            "code_hash": outcome.metadata["tool_events"][0].code_hash,
            "exit_code": 0,
            "stderr": "",
            "stdout": "24.0\n",
        }
    ]
    assert "Python sandbox" in outcome.method_calls[1].messages[0]["content"]
    thought_prompt = outcome.method_calls[-1].messages[1]["content"]
    assert 'Executed trajectory JSON:\n[{"code":"print(6 / (1 - 3 / 4))"' in thought_prompt
    assert "The executed calculation produced 24." in thought_prompt
    assert "Python sandbox" not in outcome.method_calls[-1].messages[0]["content"]


def test_rejects_unexecuted_validation_and_hidden_verifier_evidence() -> None:
    verifier_calls: list[str] = []
    outcome = _runtime().run(
        identity=_identity(),
        task=_task(),
        buffer_snapshot=[],
        client=ReplayClient(
            responses_by_sample={
                "game24-tool": {
                    "bot_problem_distill": _problem(),
                    "bot_instantiate_solve": json.dumps(
                        {"action": "final", "answer": _solve_result()}
                    ),
                    "bot_thought_distill": _thought_result(),
                }
            }
        ),
        model="replay",
        config=_tool_config(SubprocessTestDouble()),
        verifier=lambda answer: verifier_calls.append(answer)
        or verify_expression(answer, [1, 3, 4, 6]),
    )

    assert outcome.failure_disposition == "bot_invalid_thought_distillation"
    assert outcome.metadata["tool_contract_error"] == "BOT_UNEXECUTED_VALIDATION"
    assert outcome.metadata["tool_events"] == ()
    assert verifier_calls == []
    thought_prompt = outcome.method_calls[-1].messages[1]["content"]
    assert "Executed trajectory JSON:\n[]" in thought_prompt
    assert "target" in thought_prompt
    assert "VERIFIER" not in thought_prompt


def test_code_distillation_uses_only_the_retrieved_memory_id() -> None:
    executor = SubprocessTestDouble()
    retrieved = MemoryEntry(
        entry_id="retrieved-template",
        content="Validate the expression before responding.",
        memory_type="thought_template",
        metadata={"description": "Validate an arithmetic expression.", "category": "programming-based"},
    )
    hidden = MemoryEntry(
        entry_id="hidden-template",
        content="Never attribute this hidden template.",
        memory_type="thought_template",
        metadata={"description": "Avoid a hidden shortcut.", "category": "procedure-based"},
    )
    client = _CapturingReplayClient(
        responses_by_sample={
            "game24-tool": {
                "bot_problem_distill": _problem(),
                "bot_instantiate_solve": [
                    json.dumps({"action": "execute_python", "code": "print(24)"}),
                    json.dumps(
                        {
                            "action": "final",
                            "answer": json.dumps(
                                {
                                    "selected_structure": "retrieved-template",
                                    "solution_trace": "Use the retrieved validation procedure.",
                                    "final_answer": "final: 6 / (1 - 3 / 4)",
                                }
                            ),
                        }
                    ),
                ],
                "bot_thought_distill": json.dumps(
                    {
                        "description": "Validate an arithmetic expression before responding.",
                        "template": "Execute the expression before giving the final form.",
                        "category": "programming-based",
                        "explicitly_used_memory_ids": [retrieved.entry_id],
                    }
                ),
            }
        }
    )
    config = _tool_config(executor)
    config["embedding_provider"] = _RetrievedEmbeddingProvider()
    config["visible_memory_ids"] = [hidden.entry_id]

    outcome = _runtime().run(
        identity=_identity(),
        task=_task(),
        buffer_snapshot=[retrieved, hidden],
        client=client,
        model="replay",
        config=config,
        verifier=lambda answer: verify_expression(answer, [1, 3, 4, 6]),
    )

    assert outcome.status == "succeeded"
    thought_config = next(config for config in client.configs if config["method_stage"] == "bot_thought_distill")
    assert thought_config["visible_memory_ids"] == [retrieved.entry_id]
