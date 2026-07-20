from __future__ import annotations

import importlib
import importlib.util


def test_baseline_execution_adapter_enforces_semantic_call_order_and_prompt_bytes() -> None:
    assert importlib.util.find_spec("memcontam.baselines.execution"), (
        "BASELINE-FIDELITY-V1 requires the shared baseline execution adapter"
    )
    execution = importlib.import_module("memcontam.baselines.execution")

    assert callable(getattr(execution, "execute_baseline", None))
    assert callable(getattr(execution, "assert_prompt_bytes", None))


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


def test_bot_problem_and_instantiate_stages_have_strict_read_and_solve_contracts() -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    bot_solve = importlib.import_module("memcontam.baselines.bot_solve")

    assert callable(bot_read.distill_problem)
    assert callable(bot_read.build_distilled_query)
    assert callable(bot_read.retrieve_top_template)
    assert callable(bot_solve.render_bot_solve_prompt)
    assert callable(bot_solve.parse_bot_solve_result)
