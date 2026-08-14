from __future__ import annotations

import importlib
import importlib.util
from typing import Literal

from memcontam.experiment.phase12.contracts import BaselineConditionSpec, MemoryArmExecutionKey
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint


MEMORY_BASELINES = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")


def _selection_module():
    assert importlib.util.find_spec("memcontam.experiment.phase12.checkpoint_selection") is not None
    return importlib.import_module("memcontam.experiment.phase12.checkpoint_selection")


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


def _state(baseline: str, index: int) -> NativeState:
    native_state: dict[str, object] = {"checkpoint_index": index}
    if baseline == "fh_bounded":
        native_state.update(
            {"records": [{"trial": index}], "first_eviction_trial_id": None}
        )
        if index == 1:
            native_state["first_eviction_trial_id"] = "prior-injection-trial"
    elif baseline == "rag_frozen":
        native_state.update(
            {
                "branch": "contam" if index == 1 else "clean",
                "corpus_id": "clean-corpus",
                "index_id": "clean-index",
                "read_only": True,
            }
        )
    elif baseline == "bot_style":
        templates = ["bot-a", "bot-b"] if index != 1 else ["bot-a"]
        native_state.update(
            {
                "templates": templates,
                "clean_competitor_ids": templates,
            }
        )
    else:
        native_state.update(
            {
                "reflections": ["reflection-a"] if index != 1 else [],
                "active_capacity": 2,
            }
        )
    return NativeState(baseline, (), native_state)


def _checkpoints() -> dict[str, tuple]:
    return {
        baseline: tuple(serialize_checkpoint(_state(baseline, index)) for index in range(1, 6))
        for baseline in MEMORY_BASELINES
    }


def test_selects_lower_median_shared_clean_checkpoint_and_records_rejections() -> None:
    selection = _selection_module()

    result = selection.select_common_checkpoint(
        seed=17,
        checkpoints_by_baseline=_checkpoints(),
        conditions=_conditions(),
        trial_indices=(1, 2, 3, 4, 5),
        suffix_horizon=2,
    )

    assert result.joint_eligibility.joint_eligible_indices == (2, 3)
    assert result.selected_trial_index == 2
    assert result.suffix_trial_indices == (3, 4)
    assert result.blocked is False
    assert {
        checkpoint.state.native_state["checkpoint_index"]
        for checkpoint in result.selected_checkpoints.values()
    } == {2}
    rejected = {
        (item.baseline, item.checkpoint_index): item.reason_codes for item in result.rejections
    }
    assert "FH_POST_INJECTION_VISIBILITY_UNAVAILABLE" in rejected[("fh_bounded", 1)]
    assert "RAG_CLEAN_CORPUS_REQUIRED" in rejected[("rag_frozen", 1)]
    assert "BOT_CLEAN_COMPETITORS_UNAVAILABLE" in rejected[("bot_style", 1)]
    assert "REFLEXION_REFLECTIONS_UNAVAILABLE" in rejected[("reflexion_style", 1)]
    assert all(
        "INSUFFICIENT_SUFFIX_HORIZON" in rejected[(baseline, index)]
        for baseline in MEMORY_BASELINES
        for index in (4, 5)
    )


def test_empty_joint_eligibility_blocks_without_replacement() -> None:
    selection = _selection_module()
    checkpoints = _checkpoints()
    checkpoints["rag_frozen"] = (checkpoints["rag_frozen"][0],)

    result = selection.select_common_checkpoint(
        seed=18,
        checkpoints_by_baseline=checkpoints,
        conditions=_conditions(),
        trial_indices=(1, 2, 3, 4, 5),
        suffix_horizon=2,
    )

    assert result.joint_eligibility.joint_eligible_indices == ()
    assert result.selected_trial_index is None
    assert result.selected_checkpoints == {}
    assert result.suffix_trial_indices == ()
    assert result.blocked is True
    assert result.block_reason == "EMPTY_JOINT_ELIGIBILITY"


def test_checkpoint_plumbing_accepts_an_explicit_registered_panel() -> None:
    selection = _selection_module()
    checkpoints = _checkpoints()
    conditions = _conditions()
    panel = selection.BaselinePanel(
        baselines=("fh_bounded", "rag_frozen"),
        expected_families={"fh_bounded": "full_history", "rag_frozen": "rag"},
    )

    result = selection.select_checkpoint_for_panel(
        seed=19,
        checkpoints_by_baseline={baseline: checkpoints[baseline] for baseline in panel.baselines},
        conditions={baseline: conditions[baseline] for baseline in panel.baselines},
        trial_indices=(1, 2, 3, 4, 5),
        suffix_horizon=2,
        panel=panel,
    )

    assert result.joint_eligibility.joint_eligible_indices == (2, 3)
    assert result.selected_trial_index == 2
    assert set(result.selected_checkpoints) == set(panel.baselines)
