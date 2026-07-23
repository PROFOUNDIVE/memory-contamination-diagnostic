from __future__ import annotations

import importlib
import importlib.util
from typing import Literal

from memcontam.experiment.phase12.contracts import BaselineConditionSpec, MemoryArmExecutionKey
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint


def _maturity_module():
    assert importlib.util.find_spec("memcontam.experiment.phase12.maturity") is not None
    return importlib.import_module("memcontam.experiment.phase12.maturity")


def _condition(
    family: Literal["full_history", "rag", "bot", "reflexion"], *, fh_mode: str = "bounded"
) -> BaselineConditionSpec:
    return BaselineConditionSpec(
        condition_id=f"{family}-condition",
        baseline_family=family,
        fidelity_label="bounded",
        rag_mode="frozen" if family == "rag" else "not_applicable",
        fh_mode=fh_mode,
        execution_key_example=MemoryArmExecutionKey(kind="memory_arm", arm="clean"),
    )


def _checkpoint(baseline: str, index: int, native_state: dict[str, object]):
    return serialize_checkpoint(
        NativeState(
            baseline=baseline,
            entries=(f"{baseline}-clean-1", f"{baseline}-clean-2"),
            native_state={"checkpoint_index": index, "maturity_horizon": 3, **native_state},
        )
    )


def test_evaluates_primary_and_optional_maturity_from_clean_checkpoint_state() -> None:
    maturity = _maturity_module()
    decisions = (
        maturity.evaluate_maturity(
            _condition("full_history"),
            _checkpoint("fh_bounded", 3, {"records": [{"input": "x"}]}),
            horizon=3,
        ),
        maturity.evaluate_maturity(
            _condition("rag"),
            _checkpoint(
                "rag_frozen", 3, {"read_only": True, "corpus_id": "clean", "index_id": "clean"}
            ),
            horizon=3,
        ),
        maturity.evaluate_maturity(
            _condition("bot"),
            _checkpoint("bot_style", 3, {"templates": [{"id": "a"}, {"id": "b"}]}),
            horizon=3,
        ),
        maturity.evaluate_maturity(
            _condition("reflexion"),
            _checkpoint(
                "reflexion_style",
                3,
                {"reflections": [{"id": "a"}, {"id": "b"}], "active_capacity": 3},
            ),
            horizon=3,
        ),
        maturity.evaluate_optional_dc_maturity(
            _checkpoint(
                "dynamic_cheatsheet_rs_optional",
                3,
                {"archive": [{"id": "a"}], "strategy": "clean strategy"},
            ),
            horizon=3,
            condition_id="dc-optional",
        ),
    )

    assert all(decision.eligible for decision in decisions)
    assert [decision.checkpoint_index for decision in decisions] == [3, 3, 3, 3, 3]
    assert decisions[-1].baseline_family == "dc"


def test_maturity_recomputes_for_horizon_without_using_outcomes() -> None:
    maturity = _maturity_module()
    checkpoint = _checkpoint(
        "fh_bounded",
        3,
        {
            "records": [{"input": "x"}],
            "maturity_horizon": 3,
            "outcome": "incorrect",
            "verifier_result": False,
        },
    )

    eligible = maturity.evaluate_maturity(_condition("full_history"), checkpoint, horizon=3)
    ineligible = maturity.evaluate_maturity(_condition("full_history"), checkpoint, horizon=4)

    assert eligible.eligible is True
    assert ineligible.eligible is False
    assert ineligible.reason_codes == ("INSUFFICIENT_HORIZON",)
