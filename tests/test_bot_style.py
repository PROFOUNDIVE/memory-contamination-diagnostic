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
            clean_or_contaminated="contaminated",
            source_trial_id="prev_trial_1",
            metadata={
                "parent_entry_ids": ["template-parent"],
                "source_entry_ids": ["template-source"],
            },
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
    answer_call = result["method_calls"][1]
    assert result["answer_call_id"] == answer_call.call_id
    assert answer_call.stage == "bot_instantiate_solve"
    assert [call.source_spans for call in result["method_calls"] if call is not answer_call] == [[], [], []]
    assert len(answer_call.source_spans) == 1
    span = answer_call.source_spans[0]
    assert answer_call.messages[1]["content"][span.start : span.end] == (
        "entry_id=tpl_001\nLook for factor pairs of 24 and build subexpressions that create them."
    )
    assert span.source_ids == ["template-source"]
    assert span.parent_ids == ["template-parent"]


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


def test_bot_template_answer_and_accepted_write_keep_exact_lineage() -> None:
    task = TaskInstance(
        sample_id="game24_001",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
    )
    identity = BotBufferIdentity("run_t12", "game24", "bot_style", "contaminated", "gpt-4o")
    injected = MemoryEntry(
        entry_id="injected-template",
        content="Use an injected template.",
        memory_type="thought_template",
        clean_or_contaminated="contaminated",
        metadata={
            "contamination_class": "injected",
            "injected_root_ids": ["injected-template"],
            "lineage_status": "exact",
            "lineage_basis": "seed",
            "direct_parent_ids": [],
            "target_set_id": "controlled_injected_derived_v1",
            "is_target_contamination": True,
        },
    )
    client = ReplayClient(
        responses_by_sample={
            "game24_001": {
                "bot_problem_distill": _DISTILLATION_OUTPUT,
                "bot_instantiate_solve": _SOLUTION_OUTPUT,
                "bot_thought_distill": "A distinct reusable template.",
                "bot_novelty_decide": "True",
            }
        }
    )

    result = BotRuntime().run(
        identity=identity,
        task=task,
        buffer_snapshot=[injected],
        client=client,
        model="gpt-4o",
        config={
            "sample_id": "game24_001",
            "_logging_target_set_id": "controlled_injected_derived_v1",
        },
        verifier=lambda _response: VerifierResult(is_correct=True, parsed_answer="24"),
    )

    answer_span = result["method_calls"][1].source_spans[0]
    written = result["memory_after"][-1]
    assert answer_span.contamination_class == "injected"
    assert answer_span.injected_root_ids == ["injected-template"]
    assert answer_span.lineage_status == "exact"
    assert answer_span.target_set_id == "controlled_injected_derived_v1"
    assert answer_span.is_target_contamination is True
    assert written["metadata"]["direct_parent_ids"] == ["injected-template"]
    assert written["metadata"]["injected_root_ids"] == ["injected-template"]
    assert written["metadata"]["contamination_class"] == "derived"


def test_bot_runtime_retrieves_after_distillation_and_reuses_template() -> None:
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
            }
        }
    )
    events = []

    class RecordingProvider:
        def encode_query(self, text):
            events.append(("retrieve", text))
            return [1.0, 0.0]

        def encode_document(self, _text):
            return [1.0, 0.0]

    class RecordingPolicy(BotStylePolicy):
        def problem_distillation(self, *args, **kwargs):
            events.append(("distill", None))
            return super().problem_distillation(*args, **kwargs)

        def template_instantiation_solve(self, *args, **kwargs):
            events.append(("solve", kwargs["retrieved"]["entry_id"]))
            return super().template_instantiation_solve(*args, **kwargs)

    result = BotRuntime(policy=RecordingPolicy()).run(
        identity=identity,
        task=task,
        buffer_snapshot=memory_before,
        client=client,
        model="gpt-4o",
        config={
            "sample_id": "game24_001",
            "temperature": 0,
            "embedding_provider": RecordingProvider(),
        },
        verifier=lambda response: VerifierResult(
            is_correct=False, parsed_answer=response, reason="skip_update"
        ),
    )

    assert result["retrieved_template"]["entry_id"] == "tpl_001"
    assert events == [
        ("distill", None),
        ("retrieve", str(task.input)),
        ("solve", "tpl_001"),
    ]
