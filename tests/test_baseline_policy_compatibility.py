from __future__ import annotations

import inspect

from memcontam.baselines import bot_runtime


def test_bot_policy_uses_native_novelty_without_legacy_model_novelty_stage() -> None:
    assert callable(getattr(bot_runtime, "evaluate_native_novelty", None))
    assert callable(getattr(bot_runtime, "freeze_native_transition", None))
    assert "bot_novelty_decide" not in inspect.getsource(bot_runtime)
