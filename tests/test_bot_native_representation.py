from __future__ import annotations

import json

from memcontam.baselines.bot_phase12 import BoTStateV3, _as_memory_entry
from memcontam.baselines.bot_read import DistilledProblem, retrieve_top_template
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.stores import MemoryEntry


def test_ordinary_template_fields_survive_checkpoint_reopen_and_restore() -> None:
    entry = MemoryEntry(
        entry_id="ordinary-template",
        content="Apply the reusable procedural scaffold.",
        memory_type="thought_template",
        metadata={
            "description": "A distinct retrieval description.",
            "category": "programming-based",
        },
    )
    runtime = LIVE_BASELINE_REGISTRY["bot_style"]

    snapshot = runtime.serialize_state(BoTStateV3(entries=[entry]))
    assert isinstance(snapshot, NativeState)
    checkpoint = serialize_checkpoint(snapshot)
    reopened = NativeState.from_mapping(json.loads(checkpoint.canonical_bytes))
    restored = runtime.restore_state(reopened, None)
    restored_snapshot = runtime.serialize_state(restored)
    assert isinstance(restored_snapshot, NativeState)

    native = restored_snapshot.entries[0]
    assert isinstance(native, NativeEntry)
    assert native.retrieval_description == "A distinct retrieval description."
    assert native.template_body == "Apply the reusable procedural scaffold."
    assert native.category == "programming-based"
    memory = _as_memory_entry(native)
    assert memory.content == native.template_body
    assert memory.metadata["description"] == native.retrieval_description
    assert memory.metadata["category"] == native.category


def test_native_retrieval_changes_with_distilled_input_not_template_body() -> None:
    runtime = LIVE_BASELINE_REGISTRY["bot_style"]
    entries: list[MemoryEntry | NativeEntry] = [
        MemoryEntry(
            entry_id="template-arithmetic",
            content="Reusable scaffold one.",
            memory_type="thought_template",
            metadata={"description": "arithmetic retrieval description", "category": "procedure-based"},
        ),
        MemoryEntry(
            entry_id="template-alphabetic",
            content="Reusable scaffold two.",
            memory_type="thought_template",
            metadata={"description": "alphabetic retrieval description", "category": "procedure-based"},
        ),
    ]
    snapshot = runtime.serialize_state(BoTStateV3(entries=entries))
    assert isinstance(snapshot, NativeState)
    native_entries = [
        _as_memory_entry(entry) for entry in snapshot.entries if isinstance(entry, NativeEntry)
    ]

    class InputSensitiveProvider:
        def __init__(self) -> None:
            self.metadata: dict[str, object] = {}

        def encode_query(self, text: str) -> list[float]:
            return [1.0, 0.0] if "arithmetic" in text else [0.0, 1.0]

        def encode_document(self, text: str) -> list[float]:
            return [1.0, 0.0] if "arithmetic" in text else [0.0, 1.0]

    arithmetic = DistilledProblem(
        key_information="arithmetic",
        restrictions="use the numbers",
        distilled_task="solve arithmetic",
    )
    alphabetic = DistilledProblem(
        key_information="alphabetic",
        restrictions="compare letters",
        distilled_task="sort alphabetically",
    )

    assert retrieve_top_template(arithmetic, native_entries, InputSensitiveProvider()).matched_entry == (
        native_entries[0]
    )
    assert retrieve_top_template(alphabetic, native_entries, InputSensitiveProvider()).matched_entry == (
        native_entries[1]
    )
