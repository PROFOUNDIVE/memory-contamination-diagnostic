from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.clients.replay import ReplayClient
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
from memcontam.readiness.phase13_main_new_mcq_runtime import (
    build_new_mcq_live_branches,
    load_new_mcq_runtime_registry,
    new_mcq_native_entries,
)
from memcontam.readiness.phase13_new_mcq_rag_models import InterventionRegistry
from memcontam.tasks.base import TaskInstance


ROOT = Path(__file__).resolve().parents[1]
INTERVENTIONS = InterventionRegistry.model_validate_json(
    (ROOT / "data/phase13/rag/new_mcq/intervention_registry_v1.json").read_bytes()
)


@pytest.mark.parametrize(
    ("baseline", "kind", "semantic_kind", "native_component", "payload_keys"),
    (
        ("fh_bounded", "raw_interaction", "full_history_transcript", "history", {"kind", "query", "response"}),
        ("bot_style", "thought_template", "thought_template", "buffer", {"kind", "procedural_body", "retrieval_description"}),
        ("reflexion_style", "reflection", "verbal_reflection", "reflections", {"kind", "lesson"}),
        ("dc_rs", "raw_interaction", "dc_rs_io_pair", "archive", {"kind", "query", "response"}),
    ),
)
def test_new_mcq_treatments_use_each_baseline_native_carrier(
    baseline: str,
    kind: str,
    semantic_kind: str,
    native_component: str,
    payload_keys: set[str],
) -> None:
    entries = new_mcq_native_entries("mmlu_pro_engineering", baseline, INTERVENTIONS)

    assert set(entries) == {"correct", "irrelevant", "contam"}
    assert {entry.semantic_kind for entry in entries.values()} == {semantic_kind}
    assert {entry.native_component for entry in entries.values()} == {native_component}
    payloads = [json.loads(entry.content) for entry in entries.values()]
    assert all(payload_keys <= set(payload) and payload["kind"] == kind for payload in payloads)
    assert len({entry.content_hash for entry in entries.values()}) == 3
    assert all(entry.render_id for entry in entries.values())


def test_main_live_runtime_preflight_accepts_authority_complete_runtime() -> None:
    ProductionMainRuntime.preflight()


def test_new_mcq_live_branch_injects_selected_h2_carrier() -> None:
    state = FullHistoryStateV3(records=[])
    serialized = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"].serialize_state(state)
    assert isinstance(serialized, NativeState)
    context = Game24RuntimeContext(
        task=TaskInstance(
            sample_id="engineering-1",
            task_name="mmlu_pro_engineering",
            input={"question": "Which carrier is active?", "options": ["A", "B", "C", "D"]},
        ),
        client=ReplayClient(responses_by_sample={}),
        model="replay",
        verifier=lambda _answer, _task: True,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", "trial-1", 1),
        initial_states={"fh_bounded": state},
    )

    branches = build_new_mcq_live_branches(
        prefix=serialize_checkpoint(serialized),
        context=context,
        task="mmlu_pro_engineering",
        registry=load_new_mcq_runtime_registry(ROOT),
        runtime_registry=PHASE13_CORE_BASELINE_REGISTRY,
    )

    assert tuple(branches.arms) == ("clean", "correct", "irrelevant", "contam")
    assert branches.arms["clean"].root_count == 0
    assert all(branches.arms[arm].root_count == 1 for arm in ("correct", "irrelevant", "contam"))
    assert {
        event.candidate_triplet_id
        for event in branches.events
        if event.kind == "intervention_applied"
    } == {
        "phase13-new-mcq::mmlu_pro_engineering::"
        f"{INTERVENTIONS.tasks['mmlu_pro_engineering'].selected_candidate_id}"
    }
