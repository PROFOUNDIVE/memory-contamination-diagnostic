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
