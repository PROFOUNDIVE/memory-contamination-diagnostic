from __future__ import annotations

import inspect

from memcontam.baselines import bot_runtime
from memcontam.baselines.contracts import (
    BASELINE_EXECUTION_CONTRACT_V2,
    BASELINE_FIDELITY_V2,
    FAILURE_TAXONOMY_V2,
)


def test_bot_policy_uses_native_novelty_without_legacy_model_novelty_stage() -> None:
    assert callable(getattr(bot_runtime, "evaluate_native_novelty", None))
    assert callable(getattr(bot_runtime, "freeze_native_transition", None))
    assert "bot_novelty_decide" not in inspect.getsource(bot_runtime)


def test_baseline_fidelity_v2_contract_constants_share_the_v2_boundary() -> None:
    assert BASELINE_FIDELITY_V2 == "baseline_fidelity_v2"
    assert BASELINE_EXECUTION_CONTRACT_V2 == BASELINE_FIDELITY_V2
    assert FAILURE_TAXONOMY_V2 == BASELINE_FIDELITY_V2
