from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from memcontam.contamination.phase12.registry import load_current_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("render", "candidate_name"),
    [
        ("render_correct", "correct_twin"),
        ("render_irrelevant", "irrelevant_control"),
        ("render_false", "false_candidate"),
    ],
)
def test_legacy_bot_arms_render_native_template_envelope(render, candidate_name: str) -> None:
    path = ROOT / "data/phase12/registries/candidate_registry_v2.json"
    registry = load_current_candidate_registry(path)
    triplet = next(item for item in registry.triplets if item.task == "game24")
    checkpoint = serialize_checkpoint(NativeState("bot_style", (), {"templates": []}))
    candidate = getattr(triplet, candidate_name)

    renderers = RendererRegistry.governed(
        (ROOT / "data/phase13/main/legacy_dc_rs_intervention_registry_v2.json").read_bytes(),
        registry, hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    native = getattr(renderers, render)("bot_style", triplet, checkpoint)

    assert native.retrieval_description == candidate.content
    assert native.template_body == native.content
    assert native.category == "procedure-based"
