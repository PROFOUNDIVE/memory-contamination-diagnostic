from __future__ import annotations

from pathlib import Path

from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.clients.base import LLMResponse
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import (
    OrdinaryArm,
    ProspectiveOrdinaryRun,
    execute_prospective_ordinary,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.tasks.base import TaskInstance


ARMS: tuple[OrdinaryArm, ...] = ("clean", "correct", "irrelevant", "contam")


class _Client:
    def __init__(self) -> None:
        self.configs: list[dict[str, JsonValue]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, JsonValue],
    ) -> LLMResponse:
        del messages, model
        self.configs.append(dict(config))
        return LLMResponse(
            content="final: (6 / (1 - 3 / 4))",
            raw={"replay": True},
            token_usage={},
            latency_ms=0,
        )


def test_prospective_ordinary_executes_each_registered_arm_from_native_branch() -> None:
    task = TaskInstance(
        sample_id="game24:arm-test",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
        verifier_spec={"target": 24},
    )
    clean_state = FullHistoryStateV3(records=[])
    entry = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"]
    context = Game24RuntimeContext(
        task=task,
        client=_Client(),
        model="replay",
        verifier=lambda _answer, _task: True,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("branch-build", "branch-build:trial", 0),
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={"fh_bounded": clean_state},
    )
    snapshot = entry.serialize_state(clean_state)
    assert isinstance(snapshot, NativeState)
    branches = build_live_reduced_main_branches(
        prefix=serialize_checkpoint(snapshot),
        context=context,
        candidate_registry=load_candidate_registry(
            Path("data/phase12/registries/candidate_registry_v1.json")
        ),
        registry=PHASE13_CORE_BASELINE_REGISTRY,
    )

    results = []
    for arm in ARMS:
        client = _Client()
        branch = branches.arms[arm]
        result = execute_prospective_ordinary(
            ProspectiveOrdinaryRun(
                task_name="game24",
                baseline="fh_bounded",
                arm=arm,
                branch=branch,
                run_id=f"ordinary-{arm}",
                model="replay",
                client=client,
                verifier=lambda _answer, _task: True,
                decoding={"temperature": 0.0},
                tasks=(task,),
                baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
            )
        )
        assert result.arm == arm
        assert {config["arm"] for config in client.configs} == {arm}
        assert isinstance(branch.state, FullHistoryStateV3)
        assert len(branch.state.records) == branch.root_count
        results.append(result)

    assert len({id(result.trials[0].state) for result in results}) == 4
