from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def test_baseline_execution_adapter_enforces_semantic_call_order_and_prompt_bytes() -> None:
    assert importlib.util.find_spec("memcontam.baselines.execution"), (
        "BASELINE-FIDELITY-V1 requires the shared baseline execution adapter"
    )
    execution = importlib.import_module("memcontam.baselines.execution")

    assert callable(getattr(execution, "execute_baseline", None))
    assert callable(getattr(execution, "assert_prompt_bytes", None))


def test_execute_baseline_delegates_to_existing_executor_methods() -> None:
    from memcontam.baselines.execution import execute_baseline

    class Adapter:
        def execute(self, value: str) -> str:
            return f"execute:{value}"

    class Policy:
        def run(self, value: str) -> str:
            return f"run:{value}"

    assert execute_baseline(Adapter(), "task") == "execute:task"
    assert execute_baseline(Policy(), "task") == "run:task"


def test_no_memory_prompt_matches_the_committed_prompt_fixture() -> None:
    from memcontam.baselines.execution import assert_prompt_bytes
    from memcontam.baselines.no_memory import NoMemoryPolicy
    from memcontam.memory.stores import MemoryState
    from memcontam.tasks.base import TaskInstance
    from memcontam.tasks.dispatch import canonical_task_json

    task = TaskInstance(
        sample_id="sample-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
    )
    assert_prompt_bytes(
        Path(__file__).parent / "fixtures" / "prompts" / "no_memory_generate.json",
        stage="no_memory_generate",
        messages=NoMemoryPolicy().build_prompt(task, MemoryState()),
        replacements={
            "{{task_family}}": task.task_name,
            "{{task_canonical}}": canonical_task_json(task),
        },
    )


def test_full_history_adapter_uses_only_the_full_history_generate_semantic_stage() -> None:
    from memcontam.baselines.full_history import FullHistoryAdapter

    assert callable(FullHistoryAdapter().execute)
    assert not hasattr(FullHistoryAdapter(), "run")


def test_retrieval_rag_adapter_uses_only_the_rag_generate_semantic_stage() -> None:
    from memcontam.baselines.retrieval_rag import RetrievalRagAdapter

    adapter = RetrievalRagAdapter()
    assert callable(adapter.execute)
    assert not hasattr(adapter, "run")
    assert not hasattr(adapter, "build_prompt")


def test_reflexion_policy_build_prompt_delegates_to_the_adapter_renderer() -> None:
    from memcontam.baselines.reflexion_adapter import (
        ReflexionState,
        _generation_messages,
        visible_reflections,
    )
    from memcontam.baselines.reflexion_style import ReflexionStylePolicy
    from memcontam.memory.stores import MemoryEntry, MemoryState
    from memcontam.tasks.base import TaskInstance

    task = TaskInstance(sample_id="sample-1", task_name="game24", input={})
    reflection = MemoryEntry(
        entry_id="reflection-1",
        content="Reflection: verify arithmetic.",
        memory_type="verbal_reflection",
    )
    memory = MemoryState(
        entries=[
            MemoryEntry(entry_id="seed", content="do not render", memory_type="seed"),
            reflection,
        ]
    )

    assert (
        ReflexionStylePolicy().build_prompt(task, memory)
        == _generation_messages(
            task, visible_reflections(ReflexionState(reflections=[reflection]))
        )[0]
    )


def test_bot_problem_and_instantiate_stages_have_strict_read_and_solve_contracts() -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    bot_solve = importlib.import_module("memcontam.baselines.bot_solve")

    assert callable(bot_read.distill_problem)
    assert callable(bot_read.build_distilled_query)
    assert callable(bot_read.retrieve_top_template)
    assert callable(bot_solve.render_bot_solve_prompt)
    assert callable(bot_solve.parse_bot_solve_result)


def test_bot_thought_distillation_contract_requires_structured_template_payload() -> None:
    bot_write = importlib.import_module("memcontam.baselines.bot_write")

    assert callable(bot_write.distill_thought_template)
    assert callable(bot_write.validate_explicitly_used_memory_ids)
    result = bot_write.TemplateDistillationResult.model_validate(
        {
            "description": "Arithmetic factorization method.",
            "template": "Create factors before combining terms.",
            "category": "procedure-based",
            "explicitly_used_memory_ids": ["template-1"],
        }
    )
    assert result.explicitly_used_memory_ids == ("template-1",)
