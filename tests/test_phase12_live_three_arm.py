from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.clients.base import LLMResponse
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY, RuntimeEntry, RuntimeTrialResult
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.tasks.game24 import build_instance


REGISTRY_PATH = Path("data/phase12/registries/candidate_registry_v1.json")


class _Client:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model, config
        raise AssertionError("live suffix registry double does not call the client")


def _context(index: int) -> Game24RuntimeContext:
    return Game24RuntimeContext(
        task=build_instance({"sample_id": f"game24-{index}", "numbers": [1, 3, 4, 6]}),
        client=_Client(),
        model="test-model",
        verifier=lambda _answer, _task: False,
        decoding={"temperature": 0.0, "top_p": 1.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", f"trial-{index}", index),
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={"fh_bounded": FullHistoryStateV3(records=[])},
    )


def _registry(calls: list[tuple[str, str, str, tuple[tuple[str, object], ...]]]):
    fh = LIVE_BASELINE_REGISTRY["fh_bounded"]
    nomem = LIVE_BASELINE_REGISTRY["nomem"]

    def execute(context: Game24RuntimeContext, state: object) -> RuntimeTrialResult:
        calls.append(
            (
                context.branch,
                context.task.sample_id,
                context.model,
                tuple(sorted(context.decoding.items())),
            )
        )
        return RuntimeTrialResult(BaselineExecutionOutcome(status="succeeded", verifier_result=False), state)

    return {
        "fh_bounded": RuntimeEntry(
            fh.initial_state,
            execute,
            fh.serialize_state,
            fh.restore_state,
            fh.maturity_view,
        ),
        "nomem": RuntimeEntry(
            nomem.initial_state,
            execute,
            nomem.serialize_state,
            nomem.restore_state,
            nomem.maturity_view,
        ),
    }


def test_live_suffix_matches_clean_contam_and_filter_without_requiring_retrieval_or_success() -> None:
    from memcontam.experiment.phase12.live_branch import build_live_three_arm_branches
    from memcontam.experiment.phase12.live_suffix import run_live_matched_suffix

    calls: list[tuple[str, str, str, tuple[tuple[str, object], ...]]] = []
    registry = _registry(calls)
    prefix_context = _context(1)
    prefix = serialize_checkpoint(
        cast(NativeState, registry["fh_bounded"].serialize_state(FullHistoryStateV3(records=[])))
    )
    branches = build_live_three_arm_branches(
        prefix=prefix,
        context=prefix_context,
        candidate_registry=load_candidate_registry(REGISTRY_PATH),
        filter_policy=AdmissionContext(),
        registry=registry,
    )

    result = run_live_matched_suffix(
        branches_by_baseline={"fh_bounded": branches},
        contexts=(_context(2), _context(3)),
        registry=registry,
    )

    memory_run = result.memory_runs["fh_bounded"]
    assert result.suffix_ids == ("game24-2", "game24-3")
    assert memory_run.horizon == 2
    assert [(trial.arm, trial.suffix_id) for trial in memory_run.trials] == [
        (arm, suffix_id)
        for arm in ("clean", "contam", "filter")
        for suffix_id in result.suffix_ids
    ]
    assert {trial.model for trial in memory_run.trials} == {"test-model"}
    assert {tuple(sorted(trial.decoding.items())) for trial in memory_run.trials} == {
        (("temperature", 0.0), ("top_p", 1.0))
    }
    assert all(trial.outcome.verifier_result is False for trial in memory_run.trials)
    assert calls == [
        (arm, f"game24-{index}", "test-model", (("temperature", 0.0), ("top_p", 1.0)))
        for arm in ("clean", "contam", "filter", "clean")
        for index in (2, 3)
    ]


def test_live_branch_rejects_shared_mutable_arm_state() -> None:
    import pytest

    from memcontam.experiment.phase12.live_branch import LiveBranchError, build_live_three_arm_branches

    calls: list[tuple[str, str, str, tuple[tuple[str, object], ...]]] = []
    registry = _registry(calls)
    context = _context(1)
    prefix = serialize_checkpoint(
        cast(NativeState, registry["fh_bounded"].serialize_state(FullHistoryStateV3(records=[])))
    )
    branches = build_live_three_arm_branches(
        prefix=prefix,
        context=context,
        candidate_registry=load_candidate_registry(REGISTRY_PATH),
        filter_policy=AdmissionContext(),
        registry=registry,
    )

    with pytest.raises(LiveBranchError, match="CROSS_ARM_STATE_LEAKAGE"):
        replace(
            branches,
            arms={
                **branches.arms,
                "filter": replace(branches.arms["filter"], state=branches.arms["contam"].state),
            },
        )
