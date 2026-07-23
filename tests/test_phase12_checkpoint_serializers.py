from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "phase12" / "FX-BRANCH-001.json"


def _checkpoint_module():
    assert importlib.util.find_spec("memcontam.memory.checkpoint_v3") is not None
    return importlib.import_module("memcontam.memory.checkpoint_v3")


def _serializer_registry_module():
    assert importlib.util.find_spec("memcontam.memory.serializer_registry") is not None
    return importlib.import_module("memcontam.memory.serializer_registry")


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_round_trips_every_primary_baseline_checkpoint() -> None:
    checkpoint_module = _checkpoint_module()
    registry = _serializer_registry_module().SerializerRegistry.native()
    fixture = _fixture()

    states = [
        (
        checkpoint_module.NativeState.from_mapping(prefix["checkpoint"]),
            checkpoint_module.Phase12CheckpointIdentity(
                checkpoint_id=prefix["expected_checkpoint_id"],
                baseline=prefix["checkpoint"]["baseline"],
                sha256=prefix["expected_checkpoint_sha256"],
            ),
        )
        for prefix in fixture["baseline_prefixes"].values()
    ]
    states.append(
        (
            checkpoint_module.NativeState(
            baseline="dynamic_cheatsheet_rs_optional",
            entries=(
                checkpoint_module.NativeEntry(
                    entry_id="dc-archive-1",
                    semantic_kind="dc_rs_io_pair",
                    schema_version="phase12_native_entry_v1",
                    native_component="archive",
                    content="input/output archive pair",
                    content_hash="archive-content-hash",
                    direct_parent_ids=(),
                ),
                checkpoint_module.NativeEntry(
                    entry_id="dc-strategy-1",
                    semantic_kind="dynamic_cheatsheet",
                    schema_version="phase12_native_entry_v1",
                    native_component="strategy",
                    content="derived strategy",
                    content_hash="strategy-content-hash",
                    direct_parent_ids=("dc-archive-1",),
                ),
            ),
            native_state={"archive": ["dc-archive-1"], "strategy": "derived strategy"},
            ),
            None,
        )
    )

    for state, expected_identity in states:
        checkpoint = checkpoint_module.serialize_checkpoint(state, registry=registry)
        restored = checkpoint_module.deserialize_checkpoint(checkpoint, registry=registry)
        repeated = checkpoint_module.serialize_checkpoint(restored, registry=registry)

        assert restored == state
        assert repeated.canonical_bytes == checkpoint.canonical_bytes
        assert repeated.identity == checkpoint.identity
        assert checkpoint.identity.baseline == state.baseline
        assert checkpoint.identity.sha256

        if expected_identity is not None:
            assert checkpoint.identity == expected_identity

    rag_state = states[2][0]
    assert rag_state.baseline == "rag_frozen"
    assert all(
        not isinstance(entry, checkpoint_module.NativeEntry) or entry.native_component != "index"
        for entry in rag_state.entries
    )
