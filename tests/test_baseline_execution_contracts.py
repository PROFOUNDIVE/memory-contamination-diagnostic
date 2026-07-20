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
