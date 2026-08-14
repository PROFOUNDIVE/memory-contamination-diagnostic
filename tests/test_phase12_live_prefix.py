from __future__ import annotations

import importlib
import importlib.util
from typing import Literal

import pytest

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.contracts import BaselineConditionSpec, MemoryArmExecutionKey
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.runtime_registry import RuntimeEntry, RuntimeTrialResult
from memcontam.memory.checkpoint_v3 import NativeState
from memcontam.tasks.game24 import build_instance


MEMORY_BASELINES = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")


def _modules():
    assert importlib.util.find_spec("memcontam.experiment.phase12.live_prefix") is not None
    assert importlib.util.find_spec("memcontam.experiment.phase12.checkpoint_selection") is not None
    return (
        importlib.import_module("memcontam.experiment.phase12.live_prefix"),
        importlib.import_module("memcontam.experiment.phase12.checkpoint_selection"),
    )


def _conditions() -> dict[str, BaselineConditionSpec]:
    families: dict[str, Literal["full_history", "rag", "bot", "reflexion"]] = {
        "fh_bounded": "full_history",
        "rag_frozen": "rag",
        "bot_style": "bot",
        "reflexion_style": "reflexion",
    }
    return {
        baseline: BaselineConditionSpec(
            condition_id=f"{baseline}-clean",
            baseline_family=family,
            fidelity_label="bounded",
            rag_mode="frozen" if family == "rag" else "not_applicable",
            fh_mode="bounded",
            execution_key_example=MemoryArmExecutionKey(kind="memory_arm", arm="clean"),
        )
        for baseline, family in families.items()
    }


def _context(index: int) -> Game24RuntimeContext:
    return Game24RuntimeContext(
        task=build_instance({"sample_id": f"game24-{index}", "numbers": [1, 3, 4, 6]}),
        client=_Client(),
        model="test-model",
        verifier=lambda _answer, _task: True,
        decoding={"temperature": 0},
        branch="clean",
        identities=RuntimeIdentities("run-1", f"trial-{index}", index),
    )


class _Client:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model, config
        raise AssertionError("registry double does not call the client")


def _native_state(baseline: str, index: int) -> NativeState:
    native_state: dict[str, object]
    if baseline == "fh_bounded":
        native_state = {"records": [{"trial": index}], "first_eviction_trial_id": None}
    elif baseline == "rag_frozen":
        native_state = {
            "branch": "clean",
            "corpus_id": "clean-corpus",
            "index_id": "clean-index",
            "read_only": True,
        }
    elif baseline == "bot_style":
        native_state = {
            "templates": ["bot-a", "bot-b"],
            "clean_competitor_ids": ["bot-a", "bot-b"],
        }
    else:
        native_state = {"reflections": ["reflection-a"], "active_capacity": 2}
    return NativeState(baseline, (), native_state)


def _registry(calls: list[tuple[str, int, str, str]]) -> dict[str, RuntimeEntry]:
    registry: dict[str, RuntimeEntry] = {}
    for baseline in MEMORY_BASELINES:
        registry[baseline] = RuntimeEntry(
            initial_state=lambda _context: 0,
            execute_trial=lambda context, state, baseline=baseline: _execute(
                calls, baseline, context, _state_index(state)
            ),
            serialize_state=lambda state, baseline=baseline: _native_state(
                baseline, _state_index(state)
            ),
            restore_state=lambda snapshot, _context: snapshot,
            maturity_view=lambda _state, _context: None,
        )
    registry["nomem"] = RuntimeEntry(
        initial_state=lambda _context: None,
        execute_trial=lambda _context, _state: pytest.fail("NoMem must not run in a memory prefix"),
        serialize_state=lambda state: state,
        restore_state=lambda state, _context: state,
        maturity_view=lambda _state, _context: None,
    )
    return registry


def _execute(
    calls: list[tuple[str, int, str, str]], baseline: str, context: Game24RuntimeContext, state: int
) -> RuntimeTrialResult:
    assert type(context.identities.order_key) is int
    calls.append((baseline, context.identities.order_key, context.branch, context.identities.condition_id))
    return RuntimeTrialResult(outcome=BaselineExecutionOutcome(status="succeeded"), state=state + 1)


def _state_index(state: object) -> int:
    assert type(state) is int
    return state


def test_runs_clean_memory_prefixes_sequentially_and_aligns_nomem_suffix(monkeypatch) -> None:
    live_prefix, _selection = _modules()
    calls: list[tuple[str, int, str, str]] = []
    monkeypatch.setattr(live_prefix, "LIVE_BASELINE_REGISTRY", _registry(calls))

    result = live_prefix.run_live_clean_prefix(
        seed=17,
        contexts=tuple(_context(index) for index in range(1, 6)),
        conditions=_conditions(),
        suffix_horizon=1,
    )

    assert calls == [
        (baseline, index, "clean", f"{baseline}-clean")
        for baseline in MEMORY_BASELINES
        for index in range(1, 6)
    ]
    assert {
        baseline: [
            checkpoint.state.native_state["checkpoint_index"]
            for checkpoint in checkpoints
        ]
        for baseline, checkpoints in result.checkpoints_by_baseline.items()
    } == {baseline: [1, 2, 3, 4, 5] for baseline in MEMORY_BASELINES}
    assert result.selection.selected_trial_index == 2
    assert set(result.selection.selected_checkpoints) == set(MEMORY_BASELINES)
    assert {
        checkpoint.state.native_state["checkpoint_index"]
        for checkpoint in result.selection.selected_checkpoints.values()
    } == {2}
    assert [task.sample_id for task in result.suffix_tasks] == ["game24-3"]
    assert result.nomem_suffix_tasks == result.suffix_tasks


def test_rejects_a_nonclean_prefix_before_execution() -> None:
    live_prefix, _selection = _modules()
    context = _context(1)
    contaminated = Game24RuntimeContext(
        task=context.task,
        client=context.client,
        model=context.model,
        verifier=context.verifier,
        decoding=context.decoding,
        branch="contam",
        identities=context.identities,
    )

    with pytest.raises(live_prefix.LivePrefixError, match="CLEAN_PREFIX_REQUIRED"):
        live_prefix.run_live_clean_prefix(
            seed=17,
            contexts=(contaminated,),
            conditions=_conditions(),
            suffix_horizon=1,
        )


def test_runs_an_explicit_registered_memory_panel(monkeypatch) -> None:
    live_prefix, selection = _modules()
    calls: list[tuple[str, int, str, str]] = []
    monkeypatch.setattr(live_prefix, "LIVE_BASELINE_REGISTRY", _registry(calls))
    panel = selection.BaselinePanel(
        baselines=("fh_bounded", "rag_frozen"),
        expected_families={"fh_bounded": "full_history", "rag_frozen": "rag"},
    )

    result = live_prefix.run_live_clean_prefix(
        seed=17,
        contexts=tuple(_context(index) for index in range(1, 5)),
        conditions={baseline: _conditions()[baseline] for baseline in panel.baselines},
        suffix_horizon=1,
        panel=panel,
    )

    assert {baseline for baseline, *_rest in calls} == set(panel.baselines)
    assert set(result.checkpoints_by_baseline) == set(panel.baselines)
    assert set(result.selection.selected_checkpoints) == set(panel.baselines)


def test_rejects_invalid_panel_before_executing_any_prefix_trial(monkeypatch) -> None:
    live_prefix, selection = _modules()
    calls: list[tuple[str, int, str, str]] = []
    monkeypatch.setattr(live_prefix, "LIVE_BASELINE_REGISTRY", _registry(calls))
    invalid_panel = selection.BaselinePanel(
        baselines=("fh_bounded", "rag_frozen"),
        expected_families={"fh_bounded": "full_history", "rag_frozen": "full_history"},
    )

    with pytest.raises(selection.CheckpointSelectionError, match="INVALID_BASELINE_PANEL"):
        live_prefix.run_live_clean_prefix(
            seed=17,
            contexts=tuple(_context(index) for index in range(1, 5)),
            conditions={baseline: _conditions()[baseline] for baseline in invalid_panel.baselines},
            suffix_horizon=1,
            panel=invalid_panel,
        )

    assert calls == []
