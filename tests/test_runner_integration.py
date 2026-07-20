from __future__ import annotations

import inspect

import memcontam.cli as cli


def test_main_baselines_dispatch_through_one_adapter_registry() -> None:
    assert set(cli.BASELINE_ADAPTERS) == {
        "no_memory",
        "full_history",
        "retrieval_rag",
        "reflexion_style",
        "bot_style",
    }
    assert "filtered_state" not in inspect.getsource(cli)
    assert "memory.admission" not in inspect.getsource(cli)
