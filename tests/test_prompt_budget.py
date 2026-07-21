from __future__ import annotations

import pytest

from memcontam.baselines.prompt_budget import (
    PromptBudgetSpec,
    count_prompt_tokens,
    effective_prompt_budget,
)
from memcontam.config.resolution import validate_fidelity_contract


def test_counts_role_and_content_with_documented_deterministic_serialization() -> None:
    messages = [
        {"role": "system", "content": "Answer precisely."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    first_count = count_prompt_tokens(messages, "cl100k_base")
    second_count = count_prompt_tokens(messages, "cl100k_base")

    assert first_count == second_count
    assert first_count > 0


def test_effective_prompt_budget_reserves_current_task_tokens() -> None:
    spec = PromptBudgetSpec(
        context_window_tokens=200,
        max_output_tokens=50,
        fixed_prompt_overhead_tokens=25,
        safety_margin_tokens=25,
    )

    assert effective_prompt_budget(spec, current_task_tokens=40) == 60


@pytest.mark.parametrize(
    "spec,current_task_tokens",
    [
        (
            PromptBudgetSpec(
                context_window_tokens=100,
                max_output_tokens=50,
                fixed_prompt_overhead_tokens=25,
                safety_margin_tokens=25,
            ),
            0,
        ),
        (
            PromptBudgetSpec(
                context_window_tokens=100,
                max_output_tokens=50,
                fixed_prompt_overhead_tokens=25,
                safety_margin_tokens=24,
            ),
            1,
        ),
    ],
)
def test_rejects_nonpositive_effective_prompt_budget(
    spec: PromptBudgetSpec, current_task_tokens: int
) -> None:
    with pytest.raises(ValueError, match="effective prompt budget"):
        effective_prompt_budget(spec, current_task_tokens)


def test_v2_config_rejects_nonpositive_full_history_budget() -> None:
    config = {
        "run": {
            "retry_policy_version": "baseline_fidelity_v2",
            "baseline_execution_contract_version": "baseline_fidelity_v2",
            "failure_taxonomy_version": "baseline_fidelity_v2",
            "fidelity_gate_layer": "structural",
        },
        "logging": {
            "memory_policy_version": "baseline_fidelity_v2",
            "prompt_version": "baseline_fidelity_v2",
        },
        "full_history": {
            "mode": "context_bounded_pair_atomic",
            "token_encoding": "cl100k_base",
            "context_window_tokens": 100,
            "max_output_tokens": 50,
            "fixed_prompt_overhead_tokens": 25,
            "safety_margin_tokens": 25,
        },
    }

    with pytest.raises(ValueError, match="effective prompt budget"):
        validate_fidelity_contract(config)
