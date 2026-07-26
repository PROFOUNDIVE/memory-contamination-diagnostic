from __future__ import annotations

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
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", f"trial-{index}", index),
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={"fh_bounded": FullHistoryStateV3(records=[])},
    )


def test_nomem_executes_once_for_the_shared_three_arm_suffix() -> None:
    from memcontam.experiment.phase12.live_branch import build_live_three_arm_branches
    from memcontam.experiment.phase12.live_suffix import run_live_matched_suffix

    calls: list[tuple[str, str, int]] = []
    fh = LIVE_BASELINE_REGISTRY["fh_bounded"]
    nomem = LIVE_BASELINE_REGISTRY["nomem"]

    def execute(context: Game24RuntimeContext, state: object) -> RuntimeTrialResult:
        calls.append((context.branch, context.task.sample_id, id(state)))
        return RuntimeTrialResult(BaselineExecutionOutcome(status="succeeded"), state)

    registry = {
        "fh_bounded": RuntimeEntry(
            fh.initial_state, execute, fh.serialize_state, fh.restore_state, fh.maturity_view
        ),
        "nomem": RuntimeEntry(
            nomem.initial_state,
            execute,
            nomem.serialize_state,
            nomem.restore_state,
            nomem.maturity_view,
        ),
    }
    prefix = serialize_checkpoint(
        cast(NativeState, registry["fh_bounded"].serialize_state(FullHistoryStateV3(records=[])))
    )
    branches = build_live_three_arm_branches(
        prefix=prefix,
        context=_context(1),
        candidate_registry=load_candidate_registry(
            Path("data/phase12/registries/candidate_registry_v1.json")
        ),
        filter_policy=AdmissionContext(),
        registry=registry,
    )

    result = run_live_matched_suffix(
        branches_by_baseline={"fh_bounded": branches},
        contexts=(_context(2), _context(3)),
        registry=registry,
    )

    assert result.nomem.underlying_execution_count == 1
    assert result.nomem.display_aliases == ("clean", "contam", "filter")
    assert [trial.suffix_id for trial in result.nomem.trials] == list(result.suffix_ids)
    nomem_calls = calls[-2:]
    assert [call[1] for call in nomem_calls] == list(result.suffix_ids)
    assert len({call[2] for call in nomem_calls}) == 1
