from __future__ import annotations

import inspect

import memcontam.cli as cli


def test_native_contract_main_baselines_dispatch_through_one_adapter_registry() -> None:
    assert set(cli.BASELINE_ADAPTERS) == {
        "no_memory",
        "full_history",
        "retrieval_rag",
        "reflexion_style",
        "bot_style",
    }
    assert {adapter.__name__ for adapter in cli.BASELINE_ADAPTERS.values()} == {
        "NoMemoryAdapter",
        "FullHistoryAdapter",
        "RetrievalRagAdapter",
        "ReflexionAdapter",
        "BotRuntime",
    }
    assert "_ReplayBotSolveCompatibilityClient" not in inspect.getsource(cli)
    assert "_bot_memory_writeback" not in inspect.getsource(cli)
    assert "execute_baseline(" in inspect.getsource(cli)
    assert "filtered_state" not in inspect.getsource(cli)
    assert "memory.admission" not in inspect.getsource(cli)
