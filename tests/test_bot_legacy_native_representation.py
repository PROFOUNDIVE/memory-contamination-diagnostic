from __future__ import annotations

from pathlib import Path

import pytest

from memcontam.contamination.phase12.registry import load_current_candidate_registry
from memcontam.contamination.phase12.renderers import (
    render_correct,
    render_false,
    render_irrelevant,
)
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("render", "candidate_name"),
    [
        (render_correct, "correct_twin"),
        (render_irrelevant, "irrelevant_control"),
        (render_false, "false_candidate"),
    ],
)
def test_legacy_bot_arms_render_native_template_envelope(render, candidate_name: str) -> None:
    registry = load_current_candidate_registry(
        ROOT / "data/phase12/registries/candidate_registry_v2.json"
    )
    triplet = next(item for item in registry.triplets if item.task == "game24")
    checkpoint = serialize_checkpoint(NativeState("bot_style", (), {"templates": []}))
    candidate = getattr(triplet, candidate_name)

    native = render("bot_style", triplet, checkpoint)

    assert native.retrieval_description == candidate.content
    assert native.template_body == native.content
    assert native.category == "procedure-based"
