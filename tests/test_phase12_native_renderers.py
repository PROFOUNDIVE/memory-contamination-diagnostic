from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "phase12" / "FX-BRANCH-001.json"
REGISTRY_PATH = ROOT / "data" / "phase12" / "registries" / "candidate_registry_v1.json"

NATIVE_KINDS = {
    "fh_bounded": ("full_history_transcript", "history"),
    "rag_frozen": ("rag_document", "corpus"),
    "bot_style": ("thought_template", "buffer"),
    "reflexion_style": ("verbal_reflection", "reflections"),
}


def _checkpoint_module():
    assert importlib.util.find_spec("memcontam.memory.checkpoint_v3") is not None
    return importlib.import_module("memcontam.memory.checkpoint_v3")


def _renderer_module():
    assert importlib.util.find_spec("memcontam.contamination.phase12.renderers") is not None
    return importlib.import_module("memcontam.contamination.phase12.renderers")


def _controls_module():
    assert importlib.util.find_spec("memcontam.contamination.phase12.controls") is not None
    return importlib.import_module("memcontam.contamination.phase12.controls")


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _triplet():
    registry = importlib.import_module("memcontam.contamination.phase12.registry")
    return registry.load_candidate_registry(REGISTRY_PATH).triplets[0]


def _checkpoint(baseline: str):
    module = _checkpoint_module()
    state = module.NativeState.from_mapping(_fixture()["baseline_prefixes"][baseline]["checkpoint"])
    return module.serialize_checkpoint(state)


def test_renders_triplet_variants_into_each_matching_native_domain() -> None:
    renderers = _renderer_module()
    controls = _controls_module()
    triplet = _triplet()

    for baseline, (semantic_kind, native_component) in NATIVE_KINDS.items():
        checkpoint = _checkpoint(baseline)
        false_entry = renderers.render_false(baseline, triplet, checkpoint)
        correct_entry = controls.construct_correct_control(baseline, triplet, checkpoint)
        irrelevant_entry = controls.construct_irrelevant_control(baseline, triplet, checkpoint)

        assert checkpoint == _checkpoint(baseline)
        assert [entry.entry_id for entry in (false_entry, correct_entry, irrelevant_entry)] == [
            triplet.false_candidate.candidate_id,
            triplet.correct_twin.candidate_id,
            triplet.irrelevant_control.candidate_id,
        ]
        assert {
            entry.semantic_kind for entry in (false_entry, correct_entry, irrelevant_entry)
        } == {semantic_kind}
        assert {
            entry.native_component for entry in (false_entry, correct_entry, irrelevant_entry)
        } == {native_component}
        assert {
            entry.schema_version for entry in (false_entry, correct_entry, irrelevant_entry)
        } == {"phase12_native_entry_v1"}
        assert all(
            entry.direct_parent_ids == ()
            for entry in (false_entry, correct_entry, irrelevant_entry)
        )

    dc_checkpoint = _checkpoint_module().serialize_checkpoint(
        _checkpoint_module().NativeState(
            baseline="dynamic_cheatsheet_rs_optional",
            entries=("dc-archive-clean",),
            native_state={"archive": [{"id": "dc-archive-clean"}], "strategy": "clean"},
        )
    )
    dc_entry = renderers.render_false("dynamic_cheatsheet_rs_optional", triplet, dc_checkpoint)
    assert (dc_entry.semantic_kind, dc_entry.native_component) == ("dc_rs_io_pair", "archive")


def test_rejects_forbidden_or_nonmatched_interventions() -> None:
    checkpoint_module = _checkpoint_module()
    renderers = _renderer_module()
    triplet = _triplet()
    checkpoint = _checkpoint("fh_bounded")

    with pytest.raises(renderers.RendererError, match="NOMEM_INJECTION_FORBIDDEN"):
        renderers.render_false("no_memory", triplet, checkpoint)

    false_entry = renderers.render_false("fh_bounded", triplet, checkpoint)
    with_false_root = checkpoint_module.append_native_entry(checkpoint, false_entry)
    with pytest.raises(renderers.RendererError, match="DUPLICATE_ROOT"):
        renderers.render_false("fh_bounded", triplet, with_false_root)

    changed_clean_entry = replace(
        checkpoint,
        state=replace(
            checkpoint.state, entries=("changed-clean-entry", *checkpoint.state.entries[1:])
        ),
    )
    with pytest.raises(renderers.RendererError, match="CHECKPOINT_HASH_MISMATCH"):
        renderers.render_false("fh_bounded", triplet, changed_clean_entry)

    wrong_component = replace(false_entry, native_component="buffer")
    with pytest.raises(checkpoint_module.CheckpointError, match="WRONG_NATIVE_COMPONENT"):
        checkpoint_module.append_native_entry(checkpoint, wrong_component)

    dc_checkpoint = checkpoint_module.serialize_checkpoint(
        checkpoint_module.NativeState(
            baseline="dynamic_cheatsheet_rs_optional",
            entries=(),
            native_state={"archive": [], "strategy": "clean"},
        )
    )
    direct_strategy = checkpoint_module.NativeEntry(
        entry_id="strategy-root",
        semantic_kind="dynamic_cheatsheet",
        schema_version="phase12_native_entry_v1",
        native_component="strategy",
        content="Do not inject this directly.",
        content_hash="ignored-by-the-checkpoint-test",
        direct_parent_ids=(),
    )
    with pytest.raises(checkpoint_module.CheckpointError, match="DIRECT_DC_STRATEGY_ROOT"):
        checkpoint_module.append_native_entry(dc_checkpoint, direct_strategy)
