from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.clients.base import LLMResponse
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import (
    build_live_reduced_main_branches,
    build_live_three_arm_branches,
)
from memcontam.experiment.phase12.live_suffix import run_live_matched_suffix
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.tasks.base import TaskInstance

REGISTRY_PATH = Path("data/phase12/registries/candidate_registry_v1.json")


class _Client:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model, config
        return LLMResponse(
            content="final: 14 - 16 / 20 * 15 + 23 = 25",
            raw={"replay": True},
            token_usage={},
            latency_ms=0,
        )


@pytest.mark.parametrize(
    ("task", "triplet_id"),
    (
        (
            TaskInstance(
                sample_id="meb-main-1",
                task_name="math_equation_balancer",
                input={"input": "14 ? 16 ? 20 ? 15 ? 23 = 25"},
                verifier_spec={"target": "14 - 16 / 20 * 15 + 23 = 25", "target_value": 25},
            ),
            "meb-precedence-v1",
        ),
        (
            TaskInstance(
                sample_id="words-main-1",
                task_name="word_sorting",
                input={"words": ["syndrome", "therefrom"]},
                verifier_spec={"sorted_words": ["syndrome", "therefrom"]},
            ),
            "word-sorting-first-difference-v1",
        ),
    ),
)
def test_live_branch_resolves_registered_candidate_for_each_main_task(
    task: TaskInstance, triplet_id: str
) -> None:
    context = Game24RuntimeContext(
        task=task,
        client=_Client(),
        model="test-model",
        verifier=lambda _answer, _task: False,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", "trial-1", 1),
        initial_states={"fh_bounded": FullHistoryStateV3(records=[])},
    )
    entry = LIVE_BASELINE_REGISTRY["fh_bounded"]
    prefix = serialize_checkpoint(
        cast(NativeState, entry.serialize_state(FullHistoryStateV3(records=[])))
    )

    branches = build_live_three_arm_branches(
        prefix=prefix,
        context=context,
        candidate_registry=load_candidate_registry(REGISTRY_PATH),
        filter_policy=AdmissionContext(),
    )

    intervention = next(event for event in branches.events if event.kind == "intervention_applied")
    assert intervention.candidate_triplet_id == triplet_id


def test_reduced_main_suffix_executes_four_memory_arms_and_one_nomem_law() -> None:
    task = TaskInstance(
        sample_id="meb-main-1",
        task_name="math_equation_balancer",
        input={"input": "14 ? 16 ? 20 ? 15 ? 23 = 25"},
        verifier_spec={"target": "14 - 16 / 20 * 15 + 23 = 25", "target_value": 25},
    )
    context = Game24RuntimeContext(
        task=task,
        client=_Client(),
        model="test-model",
        verifier=lambda answer, seen: answer == seen.verifier_spec["target"],
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", "trial-1", 1),
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={"fh_bounded": FullHistoryStateV3(records=[])},
    )
    entry = LIVE_BASELINE_REGISTRY["fh_bounded"]
    prefix = serialize_checkpoint(
        cast(NativeState, entry.serialize_state(FullHistoryStateV3(records=[])))
    )
    branches = build_live_reduced_main_branches(
        prefix=prefix,
        context=context,
        candidate_registry=load_candidate_registry(REGISTRY_PATH),
    )

    result = run_live_matched_suffix(
        branches_by_baseline={"fh_bounded": branches},
        contexts=(context,),
    )

    assert tuple(branches.arms) == ("clean", "correct", "irrelevant", "contam")
    assert [trial.arm for trial in result.memory_runs["fh_bounded"].trials] == [
        "clean",
        "correct",
        "irrelevant",
        "contam",
    ]
    assert result.nomem.display_aliases == ("clean", "correct", "irrelevant", "contam")
    assert result.nomem.underlying_execution_count == 1
