from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.clients.replay import ReplayClient
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance

from .test_phase13_dc_rs_runtime import _EmbeddingProvider


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/phase12/registries/candidate_registry_v1.json"
INTERVENTIONS = ROOT / "data/phase13/main/legacy_dc_rs_intervention_registry_v1.json"


def _context() -> Game24RuntimeContext:
    task = TaskInstance(
        sample_id="phase13-dc-rs-render-test",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    )
    return Game24RuntimeContext(
        task=task,
        client=ReplayClient(responses_by_sample={}),
        model="replay",
        verifier=lambda _answer, _task: False,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities(
            "run-1",
            "run-1:trial:2:game24:phase13-dc-rs-render-test",
            2,
            "dc_rs",
        ),
        embedding_provider=_EmbeddingProvider(),
        baseline_configs={
            "dc_rs": {
                "embedding_mode": "test_double",
                "serialized_cheatsheet_budget_tokens": 8192,
                "tool_mode": "text_only",
            }
        },
        initial_states={},
    )


def _renderers(raw: bytes | None = None) -> RendererRegistry:
    candidates = load_candidate_registry(CANDIDATES)
    candidate_raw = CANDIDATES.read_bytes()
    factory = RendererRegistry.governed
    return factory(
        INTERVENTIONS.read_bytes() if raw is None else raw,
        candidates,
        hashlib.sha256(candidate_raw).hexdigest(),
    )


def test_governed_dc_rs_triplet_uses_identical_query_and_candidate_bound_responses() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    prefix = serialize_checkpoint(
        cast(NativeState, entry.serialize_state(DcRsStateV3([], allow_unparented_strategies=True))),
        checkpoint_index=1,
    )

    branches = build_live_reduced_main_branches(
        prefix=prefix,
        context=_context(),
        candidate_registry=load_candidate_registry(CANDIDATES),
        registry=PHASE13_CORE_BASELINE_REGISTRY,
        renderers=_renderers(),
    )

    records = {
        arm: json.loads(cast(NativeEntry, branch.checkpoint.state.entries[-1]).content)
        for arm, branch in (
            ("correct", branches.arms["correct"]),
            ("irrelevant", branches.arms["irrelevant"]),
            ("contam", branches.arms["contam"]),
        )
    }
    triplet = load_candidate_registry(CANDIDATES).triplets[0]
    assert len({record["input"] for record in records.values()}) == 1
    assert records["correct"]["raw_output"] == triplet.correct_twin.content
    assert records["irrelevant"]["raw_output"] == triplet.irrelevant_control.content
    assert records["contam"]["raw_output"] == triplet.false_candidate.content


def test_governed_dc_rs_registry_rejects_rehashed_surface_tampering() -> None:
    payload = json.loads(INTERVENTIONS.read_bytes())
    record = payload["tasks"][0]["records"][0]
    record["query"] += " tampered"
    record["query_sha256"] = hashlib.sha256(record["query"].encode()).hexdigest()
    content = json.dumps(
        {"input": record["query"], "raw_output": record["response"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    record["serialized_content_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    record["render_id"] = (
        f"legacy-dc-rs-render-v1::{record['candidate_id']}::"
        f"{record['serialized_content_sha256'][:16]}"
    )
    unhashed = dict(payload)
    unhashed.pop("registry_sha256")
    payload["registry_sha256"] = hashlib.sha256(
        (json.dumps(unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    tampered = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()

    with pytest.raises(ValueError, match="LEGACY_DC_RS_REGISTRY_INVALID"):
        _renderers(tampered)


def test_controlled_dc_rs_root_is_not_misclassified_as_ordinary_history() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()
    prefix = serialize_checkpoint(
        cast(NativeState, entry.serialize_state(DcRsStateV3([], allow_unparented_strategies=True))),
        checkpoint_index=1,
    )
    branches = build_live_reduced_main_branches(
        prefix=prefix,
        context=context,
        candidate_registry=load_candidate_registry(CANDIDATES),
        registry=PHASE13_CORE_BASELINE_REGISTRY,
        renderers=_renderers(),
    )
    contam = branches.arms["contam"].state
    expected = cast(NativeEntry, branches.arms["contam"].checkpoint.state.entries[-1])

    restored = entry.initial_state(
        replace(
            context,
            branch="contam",
            initial_states={"dc_rs": contam},
            expected_intervention=expected,
        )
    )

    assert isinstance(restored, DcRsStateV3)
    assert restored.injected_root_id == "candidate-game24-integer-intermediates-v1"


def test_controlled_dc_rs_root_rejects_self_consistent_forgery() -> None:
    entry = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    context = _context()
    prefix = serialize_checkpoint(
        cast(NativeState, entry.serialize_state(DcRsStateV3([], allow_unparented_strategies=True))),
        checkpoint_index=1,
    )
    branch = build_live_reduced_main_branches(
        prefix=prefix,
        context=context,
        candidate_registry=load_candidate_registry(CANDIDATES),
        registry=PHASE13_CORE_BASELINE_REGISTRY,
        renderers=_renderers(),
    ).arms["contam"]
    state = cast(DcRsStateV3, branch.state)
    root = next(item for item in state.archive if item.entry_id == state.injected_root_id)
    assert isinstance(root, MemoryEntry)
    root.content = '{"input":"forged","raw_output":"forged"}'
    root.metadata["render_id"] = "legacy-dc-rs-render-v1::forged"

    with pytest.raises(ValueError, match="INVALID_DC_RS_STATE"):
        entry.initial_state(
            replace(
                context,
                branch="contam",
                initial_states={"dc_rs": state},
                expected_intervention=cast(NativeEntry, branch.checkpoint.state.entries[-1]),
            )
        )
