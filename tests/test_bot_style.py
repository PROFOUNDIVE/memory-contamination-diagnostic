from __future__ import annotations

import importlib
import importlib.util
import json

import pytest

from memcontam.baselines.bot_style import BotStylePolicy
from memcontam.clients.replay import ReplayClient
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.base import TaskInstance


BotRuntime = importlib.import_module("memcontam.baselines.bot_runtime").BotRuntime


def test_bot_contract_requires_structured_thought_before_verifier() -> None:
    runtime = importlib.import_module("memcontam.baselines.bot_runtime")

    assert callable(getattr(runtime, "freeze_native_transition", None))
    assert callable(getattr(runtime, "materialize_frozen_transition", None))


_DISTILLATION_OUTPUT = json.dumps(
    {
        "key_information": "numbers = [1, 2, 3, 4], target = 24",
        "restrictions": "Use each given number exactly once.",
        "distilled_task": "Construct an arithmetic expression using all numbers that evaluates to the target.",
    }
)


_SOLUTION_OUTPUT = json.dumps(
    {
        "selected_structure": "retrieved-template",
        "solution_trace": "Pair 1 + 3 and 2 + 4, then multiply the pair sums.",
        "final_answer": "final: (1 + 3) * (2 + 4) = 24",
    }
)


def _thought_output(*, used_ids: list[str] | None = None) -> str:
    return json.dumps(
        {
            "description": "Build factor-pair subexpressions before combining all numbers.",
            "template": "Create useful factors before combining intermediate values.",
            "category": "procedure-based",
            "explicitly_used_memory_ids": used_ids or [],
        }
    )


class _AdmittingEmbeddingProvider:
    metadata: dict[str, object] = {}

    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if text.startswith("{") else [0.0, 1.0]

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def test_bot_thought_distillation_compatibility_shim_uses_structured_solution_trace() -> None:
    bot_style = importlib.import_module("memcontam.baselines.bot_style")
    compatibility_shim = getattr(bot_style, "distill_thought_template", None)
    task = TaskInstance(sample_id="game24_001", task_name="game24", input={"numbers": [1, 2, 3, 4]})

    class ExplodingVerifier:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("compatibility shim must not read verifier state")

    assert callable(compatibility_shim)
    assert compatibility_shim(task, _SOLUTION_OUTPUT, ExplodingVerifier(), None) == (
        "Pair 1 + 3 and 2 + 4, then multiply the pair sums."
    )


def test_bot_problem_distill_accepts_only_the_three_field_schema() -> None:
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
    assert importlib.util.find_spec("memcontam.baselines.bot_read") is not None
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    distilled = bot_read.distill_problem(task, client, "gpt-4o", {"sample_id": "game24_001"})

    assert distilled.key_information == "numbers = [1, 2, 3, 4], target = 24"
    assert "exactly once" in distilled.restrictions
    assert "arithmetic expression" in distilled.distilled_task


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
                    '{"key_information":"expression = 1 + 2 * 3",'
                    '"distilled_task":"Compute the value.",'
                    '"answer_form":"final: <number>"}'
                ),
            }
        }
    )
    assert importlib.util.find_spec("memcontam.baselines.bot_read") is not None
    bot_read = importlib.import_module("memcontam.baselines.bot_read")

    with pytest.raises(ValueError, match="restrictions"):
        bot_read.distill_problem(task, client, "gpt-4o", {"sample_id": "math_001"})


def test_bot_retrieve_uses_distilled_query_and_accepts_threshold_equality() -> None:
    assert importlib.util.find_spec("memcontam.baselines.bot_read") is not None
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    problem = bot_read.DistilledProblem(
        key_information="numbers = [1, 2, 3, 4]",
        restrictions="Use each number once.",
        distilled_task="Construct an expression for 24.",
    )
    entries = [
        MemoryEntry(
            entry_id="tpl_001",
            content="Build useful pairs before combining them.",
            memory_type="thought_template",
            clean_or_contaminated="clean",
            metadata={
                "description": "Build useful pairs before combining them.",
                "category": "procedure-based",
            },
        )
    ]
    queries = []

    class ThresholdProvider:
        def encode_query(self, query):
            queries.append(query)
            return [1.0, 0.0]

        def encode_document(self, _description):
            return [0.7, (1 - 0.7**2) ** 0.5]

    retrieved = bot_read.retrieve_top_template(problem, entries, ThresholdProvider())

    assert retrieved.decision == "matched"
    assert retrieved.matched_entry is not None
    assert retrieved.matched_entry.entry_id == "tpl_001"
    assert retrieved.top_similarity == pytest.approx(0.7)
    assert queries == [bot_read.build_distilled_query(problem)]


def test_bot_retrieve_below_threshold_uses_fixed_fallback() -> None:
    assert importlib.util.find_spec("memcontam.baselines.bot_read") is not None
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    bot_solve = importlib.import_module("memcontam.baselines.bot_solve")
    problem = bot_read.DistilledProblem(
        key_information="x = 3",
        restrictions="Return a number.",
        distilled_task="Double x.",
    )
    entries = [
        MemoryEntry(
            entry_id="tpl_001",
            content="Ignore this weak match.",
            memory_type="thought_template",
            clean_or_contaminated="clean",
            metadata={"description": "Ignore this weak match.", "category": "procedure-based"},
        )
    ]

    class BelowThresholdProvider:
        def encode_query(self, _query):
            return [1.0, 0.0]

        def encode_document(self, _description):
            return [0.699, (1 - 0.699**2) ** 0.5]

    retrieval_decision = bot_read.retrieve_top_template(problem, entries, BelowThresholdProvider())
    prompt, source_spans = bot_solve.render_bot_solve_prompt(
        TaskInstance(sample_id="sample", task_name="math_equation_balancer", input={"input": "3 + 3"}),
        problem,
        retrieval_decision,
    )

    assert retrieval_decision.decision == "miss"
    assert "prompt-based" in prompt
    assert "procedure-based" in prompt
    assert "programming-based" in prompt
    assert source_spans == []


def test_bot_solve_requires_trace_and_final_answer_without_verifier() -> None:
    assert importlib.util.find_spec("memcontam.baselines.bot_solve") is not None
    bot_solve = importlib.import_module("memcontam.baselines.bot_solve")

    result = bot_solve.parse_bot_solve_result(_SOLUTION_OUTPUT)

    assert result.solution_trace.startswith("Pair 1 + 3")
    assert result.final_answer == "final: (1 + 3) * (2 + 4) = 24"
    with pytest.raises(ValueError):
        bot_solve.parse_bot_solve_result('{"final_answer":"24"}')


def test_bot_build_prompt_uses_structured_solve_contract() -> None:
    prompt = BotStylePolicy().build_prompt(
        TaskInstance(sample_id="sample", task_name="math_equation_balancer", input={"input": "3 + 3"}),
        MemoryState(),
        embedding_provider=_AdmittingEmbeddingProvider(),
    )

    assert prompt[0]["role"] == "user"
    assert "Distilled problem JSON:" in prompt[0]["content"]
    assert "solution_trace, final_answer" in prompt[0]["content"]


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
                "description": "Build factor-pair subexpressions before combining all numbers.",
                "category": "procedure-based",
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
                "bot_thought_distill": _thought_output(used_ids=["tpl_001"]),
            }
        }
    )

    result = BotRuntime().run(
        identity=identity,
        task=task,
        buffer_snapshot=memory_before,
        client=client,
        model="gpt-4o",
        config={
            "sample_id": "game24_001",
            "temperature": 0,
            "embedding_provider": _AdmittingEmbeddingProvider(),
        },
        verifier=lambda response: response == "(1 + 3) * (2 + 4) = 24",
    )

    assert result.final_response == "final: (1 + 3) * (2 + 4) = 24"
    assert result.parsed_answer == "(1 + 3) * (2 + 4) = 24"
    assert result.verifier_result is True
    assert result.retrieved_memory[0]["entry_id"] == "tpl_001"
    assert [call.stage for call in result.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
    ]
    assert [entry["entry_id"] for entry in result.memory_before] == ["tpl_001"]
    assert len(result.memory_after) == 2
    assert result.memory_write_event["status"] == "accepted"
    assert result.memory_write_event["new_entry_id"] == result.memory_after[-1]["entry_id"]
    assert result.memory_write_event["source_outcome"] is True
    assert result.metadata["bot_buffer_identity"] == identity.__dict__
    written = result.memory_after[-1]
    assert written["metadata"]["direct_parent_ids"] == ["tpl_001"]
    assert written["metadata"]["memory_support_ids"] == ["tpl_001"]
    answer_call = result.method_calls[1]
    assert result.answer_call_id == answer_call.call_id
    assert answer_call.stage == "bot_instantiate_solve"
    assert [call.source_spans for call in result.method_calls if call is not answer_call] == [[], []]
    assert len(answer_call.source_spans) == 1
    span = answer_call.source_spans[0]
    assert answer_call.messages[1]["content"][span.start : span.end] == (
        "entry_id=tpl_001\nLook for factor pairs of 24 and build subexpressions that create them."
    )
    assert span.source_ids == ["template-source"]
    assert span.parent_ids == ["template-parent"]
    thought_call = result.method_calls[2]
    assert "Pair 1 + 3 and 2 + 4" in thought_call.messages[1]["content"]
    assert "final: (1 + 3) * (2 + 4) = 24" in thought_call.messages[1]["content"]


def test_bot_runtime_valid_incorrect_answer_keeps_frozen_admission() -> None:
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
            metadata={
                "description": "Build factor-pair subexpressions before combining all numbers.",
                "category": "procedure-based",
            },
        )
    ]
    client = ReplayClient(
        responses_by_sample={
            "game24_001": {
                "bot_problem_distill": _DISTILLATION_OUTPUT,
                "bot_instantiate_solve": _SOLUTION_OUTPUT,
                "bot_thought_distill": _thought_output(),
                }
            }
    )

    result = BotRuntime().run(
        identity=identity,
        task=task,
        buffer_snapshot=memory_before,
        client=client,
        model="gpt-4o",
        config={
            "sample_id": "game24_001",
            "temperature": 0,
            "embedding_provider": _AdmittingEmbeddingProvider(),
        },
        verifier=lambda _response: False,
    )

    assert result.status == "succeeded"
    assert result.verifier_result is False
    assert len(result.memory_after) == len(result.memory_before) + 1
    assert result.memory_write_event["status"] == "accepted"
    assert result.memory_write_event["source_outcome"] is False
    assert [call.stage for call in result.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
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
            "description": "Use the injected template.",
            "category": "procedure-based",
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
                    "bot_thought_distill": _thought_output(used_ids=["injected-template"]),
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
            "embedding_provider": _AdmittingEmbeddingProvider(),
        },
        verifier=lambda _response: True,
    )

    answer_span = result.method_calls[1].source_spans[0]
    written = result.memory_after[-1]
    assert answer_span.contamination_class == "injected"
    assert answer_span.injected_root_ids == ["injected-template"]
    assert answer_span.lineage_status == "exact"
    assert answer_span.target_set_id == "controlled_injected_derived_v1"
    assert answer_span.is_target_contamination is True
    assert written["metadata"]["direct_parent_ids"] == ["injected-template"]
    assert written["metadata"]["memory_support_ids"] == ["injected-template"]
    assert written["metadata"]["injected_root_ids"] == ["injected-template"]
    assert written["metadata"]["contamination_class"] == "derived"


def test_bot_novelty_rejects_threshold_equality_without_model_call() -> None:
    runtime = importlib.import_module("memcontam.baselines.bot_runtime")
    bot_write = importlib.import_module("memcontam.baselines.bot_write")
    candidate = bot_write.BoTTemplatePayload(
        description="candidate description",
        template="candidate template",
        category="procedure-based",
        explicitly_used_memory_ids=(),
    )
    existing = MemoryEntry(
        entry_id="existing-template",
        content="existing template",
        memory_type="thought_template",
        clean_or_contaminated="clean",
        metadata={"description": "existing description", "category": "procedure-based"},
    )

    class EqualityProvider:
        metadata = {}

        def encode_query(self, text: str) -> list[float]:
            del text
            return [1.0, 0.0]

        def encode_document(self, text: str) -> list[float]:
            del text
            return [0.7, (1 - 0.7**2) ** 0.5]

    decision = runtime.evaluate_native_novelty(candidate, [existing], EqualityProvider())

    assert decision.admitted is False
    assert decision.top_similarity == pytest.approx(0.7)
    assert decision.compared_entry_id == "existing-template"


def test_bot_verifier_contract_failure_keeps_admitted_transition() -> None:
    task = TaskInstance(
        sample_id="game24_001",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4], "target": 24},
    )
    client = ReplayClient(
        responses_by_sample={
            "game24_001": {
                "bot_problem_distill": _DISTILLATION_OUTPUT,
                "bot_instantiate_solve": json.dumps(
                    {
                        "selected_structure": "procedure-based",
                        "solution_trace": "Pair 1 + 3 and 2 + 4, then multiply the pair sums.",
                        "final_answer": "final: (1 + 3) * (2 + 4) = 24",
                    }
                ),
                "bot_thought_distill": _thought_output(),
            }
        }
    )

    def failing_verifier(_response: str) -> bool:
        raise RuntimeError("verifier unavailable")

    result = BotRuntime().run(
        identity=BotBufferIdentity("run_t12", "game24", "bot_style", "clean", "gpt-4o"),
        task=task,
        buffer_snapshot=[],
        client=client,
        model="gpt-4o",
        config={"sample_id": "game24_001", "embedding_provider": _AdmittingEmbeddingProvider()},
        verifier=failing_verifier,
    )

    assert result.status == "failed"
    assert result.error_type == "VerifierContractError"
    assert result.failure_disposition == "verifier_contract_failed"
    assert result.verifier_result is None
    assert len(result.memory_after) == 1
    assert result.memory_write_event["status"] == "accepted"
    assert result.memory_write_event["source_outcome"] is None
    assert result.memory_after[0]["metadata"]["source_outcome"] is None


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
            metadata={
                "description": "Build factor-pair subexpressions before combining all numbers.",
                "category": "procedure-based",
            },
        )
    ]
    client = ReplayClient(
        responses_by_sample={
            "game24_001": {
                "bot_problem_distill": _DISTILLATION_OUTPUT,
                "bot_instantiate_solve": _SOLUTION_OUTPUT,
                "bot_thought_distill": _thought_output(),
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
            events.append(("solve", kwargs["retrieval_decision"].matched_entry.entry_id))
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
        verifier=lambda _response: True,
    )

    bot_read = importlib.import_module("memcontam.baselines.bot_read")

    assert result.retrieved_memory[0]["entry_id"] == "tpl_001"
    assert events == [
        ("distill", None),
        (
            "retrieve",
            bot_read.build_distilled_query(
                bot_read.DistilledProblem(
                    key_information="numbers = [1, 2, 3, 4], target = 24",
                    restrictions="Use each given number exactly once.",
                    distilled_task=(
                        "Construct an arithmetic expression using all numbers that evaluates to the target."
                    ),
                )
            ),
        ),
        ("solve", "tpl_001"),
        ("retrieve", "Build factor-pair subexpressions before combining all numbers."),
    ]
