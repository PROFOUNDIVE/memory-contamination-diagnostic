from __future__ import annotations

import importlib

import pytest

from memcontam.baselines.bot_style import BotStylePolicy
from memcontam.clients.replay import ReplayClient
from memcontam.logging.schema import VerifierResult
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


BotRuntime = importlib.import_module("memcontam.baselines.bot_runtime").BotRuntime


_DISTILLATION_OUTPUT = """Distilled Information:

1. Key information:
numbers = [1, 2, 3, 4], target = 24

2. Restriction:
Use each given number exactly once.

3. Distilled task:
Construct an arithmetic expression using all numbers that evaluates to the target.

4. Python transformation:
numbers = [1, 2, 3, 4]
target = 24

5. Answer form:
Output a single arithmetic expression prefixed with 'final: '.
"""


_SOLUTION_OUTPUT = "final: (1 + 3) * (2 + 4) = 24"


def test_bot_runs_distill_retrieve_instantiate_solve() -> None:
    task = TaskInstance(
        sample_id="game24_001",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
    )
    client = ReplayClient(
        responses_by_sample={
            "game24_001": {
                "bot_problem_distill": _DISTILLATION_OUTPUT,
                "bot_instantiate_solve": _SOLUTION_OUTPUT,
            }
        }
    )
    memory = MemoryState(
        entries=[
            MemoryEntry(
                entry_id="tpl_001",
                content="Look for factor pairs of 24 and build subexpressions that create them.",
                memory_type="thought_template",
                clean_or_contaminated="clean",
                source_trial_id="prev_trial_1",
            ),
            MemoryEntry(
                entry_id="tpl_002",
                content="Sort words alphabetically and preserve duplicates.",
                memory_type="thought_template",
                clean_or_contaminated="clean",
                source_trial_id="prev_trial_2",
            ),
        ]
    )
    policy = BotStylePolicy()
    base_config = {"sample_id": "game24_001", "temperature": 0}

    distilled = policy.problem_distillation(task, client, "gpt-4o", dict(base_config))

    assert distilled["key_information"] == "numbers = [1, 2, 3, 4], target = 24"
    assert "exactly once" in distilled["restriction"]
    assert "arithmetic expression" in distilled["distilled_task"]
    assert "numbers = [1, 2, 3, 4]" in distilled["python_transformation"]
    assert "final:" in distilled["answer_form"]

    solution = policy.template_instantiation_solve(
        task, distilled, memory, client, "gpt-4o", dict(base_config)
    )

    assert solution == _SOLUTION_OUTPUT


def test_bot_rejects_malformed_problem_distillation() -> None:
    task = TaskInstance(
        sample_id="math_001",
        task_name="math_equation_balancer",
        input={"input": "1 + 2 * 3"},
    )
    client = ReplayClient(
        responses_by_sample={
            "math_001": {
                "bot_problem_distill": (
                    "Distilled Information:\n\n"
                    "1. Key information:\nexpression = '1 + 2 * 3'\n\n"
                    "3. Distilled task:\nCompute the value.\n\n"
                    "5. Answer form:\nfinal: <number>\n"
                ),
            }
        }
    )
    policy = BotStylePolicy()

    with pytest.raises(ValueError, match="Restriction|Python transformation"):
        policy.problem_distillation(
            task, client, "gpt-4o", {"sample_id": "math_001"}
        )


def test_bot_runtime_runs_reference_order_and_updates() -> None:
    task = TaskInstance(
        sample_id="game24_001",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
    )
    identity = BotBufferIdentity("run_t12", "game24", "bot_style", "clean", "gpt-4o")
    memory_before = [
        MemoryEntry(
            entry_id="tpl_001",
            content="Look for factor pairs of 24 and build subexpressions that create them.",
            memory_type="thought_template",
            clean_or_contaminated="clean",
            source_trial_id="prev_trial_1",
        )
    ]
    client = ReplayClient(
        responses_by_sample={
            "game24_001": {
                "bot_problem_distill": _DISTILLATION_OUTPUT,
                "bot_instantiate_solve": _SOLUTION_OUTPUT,
                "bot_thought_distill": "Build factor-pair subexpressions before combining all numbers.",
                "bot_novelty_decide": "True",
            }
        }
    )

    result = BotRuntime().run(
        identity=identity,
        task=task,
        buffer_snapshot=memory_before,
        client=client,
        model="gpt-4o",
        config={"sample_id": "game24_001", "temperature": 0},
        verifier=lambda response: VerifierResult(
            is_correct=True, parsed_answer="(1 + 3) * (2 + 4)", reason="ok"
        ),
    )

    assert result["final_response"] == _SOLUTION_OUTPUT
    assert result["parsed_answer"] == "(1 + 3) * (2 + 4)"
    assert result["verifier_result"].is_correct is True
    assert result["retrieved_template"]["entry_id"] == "tpl_001"
    assert [call.stage for call in result["method_calls"]] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
        "bot_novelty_decide",
    ]
    assert [entry["entry_id"] for entry in result["memory_before"]] == ["tpl_001"]
    assert len(result["memory_after"]) == 2
    assert result["memory_write_event"]["status"] == "accepted"
    assert result["memory_write_event"]["new_entry_id"] == result["memory_after"][-1]["entry_id"]
    assert result["metadata"]["bot_buffer_identity"] == identity.__dict__


def test_bot_runtime_failed_verifier_stops_update() -> None:
    task = TaskInstance(
        sample_id="game24_001",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
    )
    identity = BotBufferIdentity("run_t12", "game24", "bot_style", "clean", "gpt-4o")
    memory_before = [
        MemoryEntry(
            entry_id="tpl_001",
            content="Look for factor pairs of 24 and build subexpressions that create them.",
            memory_type="thought_template",
            clean_or_contaminated="clean",
            source_trial_id="prev_trial_1",
        )
    ]
    client = ReplayClient(
        responses_by_sample={
            "game24_001": {
                "bot_problem_distill": _DISTILLATION_OUTPUT,
                "bot_instantiate_solve": "final: model claims success but verifier rejects it",
                "bot_thought_distill": "must not be consumed",
            }
        }
    )

    result = BotRuntime().run(
        identity=identity,
        task=task,
        buffer_snapshot=memory_before,
        client=client,
        model="gpt-4o",
        config={"sample_id": "game24_001", "temperature": 0},
        verifier=lambda response: VerifierResult(
            is_correct=False, parsed_answer=response, reason="verifier_failed"
        ),
    )

    assert result["verifier_result"].is_correct is False
    assert result["memory_after"] == result["memory_before"]
    assert result["memory_write_event"] is None
    assert [call.stage for call in result["method_calls"]] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
    ]
