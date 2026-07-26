from __future__ import annotations

from pathlib import Path

from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.clients.base import LLMResponse
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.tasks.game24 import build_instance


class _Client:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model, config
        raise AssertionError("branch construction does not call the client")


def test_filter_uses_the_contaminated_source_and_exposes_only_active_state() -> None:
    from memcontam.experiment.phase12.live_branch import build_live_three_arm_branches

    context = Game24RuntimeContext(
        task=build_instance({"sample_id": "game24-1", "numbers": [1, 3, 4, 6]}),
        client=_Client(),
        model="test-model",
        verifier=lambda _answer, _task: False,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", "trial-1", 1),
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={"fh_bounded": FullHistoryStateV3(records=[])},
    )
    snapshot = LIVE_BASELINE_REGISTRY["fh_bounded"].serialize_state(FullHistoryStateV3(records=[]))
    assert isinstance(snapshot, NativeState)
    prefix = serialize_checkpoint(snapshot)
    candidate_registry = load_candidate_registry(
        Path("data/phase12/registries/candidate_registry_v1.json")
    )

    branches = build_live_three_arm_branches(
        prefix=prefix,
        context=context,
        candidate_registry=candidate_registry,
        filter_policy=AdmissionContext(),
    )

    root_id = candidate_registry.triplets[0].false_candidate.candidate_id
    clean, contam, filtered = (branches.arms[arm] for arm in ("clean", "contam", "filter"))
    assert filtered.filter_state is not None
    active_ids = {
        entry.entry_id for entry in filtered.filter_state.reader_entries if isinstance(entry, NativeEntry)
    }
    updater_ids = {
        entry.entry_id for entry in filtered.filter_state.updater_entries if isinstance(entry, NativeEntry)
    }

    assert [branches.arms[arm].root_count for arm in ("clean", "contam", "filter")] == [0, 1, 1]
    assert clean.prefix_identity == contam.prefix_identity == filtered.prefix_identity
    assert contam.source_identity == filtered.source_identity
    assert contam.injected_root_id == filtered.injected_root_id == root_id
    assert root_id not in active_ids
    assert updater_ids == active_ids
    assert [(event.kind, event.arm) for event in branches.events] == [
        ("branch_constructed", "clean"),
        ("branch_constructed", "contam"),
        ("branch_constructed", "filter"),
        ("intervention_applied", "contam"),
        ("intervention_applied", "filter"),
    ]
