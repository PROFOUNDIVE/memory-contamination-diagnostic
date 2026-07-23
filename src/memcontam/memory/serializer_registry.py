from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from memcontam.memory.checkpoint_v3 import CHECKPOINT_V3, NATIVE_ENTRY_V1, CheckpointError, NativeEntry, NativeState


@dataclass(frozen=True)
class NativeSchema:
    semantic_kind: str
    native_component: str


NATIVE_SCHEMAS: Mapping[str, NativeSchema] = {
    "full_history": NativeSchema("full_history_transcript", "history"),
    "fh_bounded": NativeSchema("full_history_transcript", "history"),
    "retrieval_rag": NativeSchema("rag_document", "corpus"),
    "rag_frozen": NativeSchema("rag_document", "corpus"),
    "bot_style": NativeSchema("thought_template", "buffer"),
    "reflexion_style": NativeSchema("verbal_reflection", "reflections"),
    "dynamic_cheatsheet_rs_optional": NativeSchema("dc_rs_io_pair", "archive"),
}


@dataclass(frozen=True)
class SerializerRegistry:
    schemas: Mapping[str, NativeSchema]

    @classmethod
    def native(cls) -> SerializerRegistry:
        return cls(NATIVE_SCHEMAS)

    def schema_for(self, baseline: str) -> NativeSchema:
        try:
            return self.schemas[baseline]
        except KeyError as error:
            raise CheckpointError("UNKNOWN_NATIVE_BASELINE") from error

    def validate(self, state: NativeState) -> None:
        if state.schema_version != CHECKPOINT_V3:
            raise CheckpointError("INVALID_CHECKPOINT_SCHEMA")
        schema = self.schema_for(state.baseline)
        entry_ids = set()
        for entry in state.entries:
            entry_id = entry.entry_id if isinstance(entry, NativeEntry) else entry
            if not isinstance(entry_id, str) or not entry_id or entry_id in entry_ids:
                raise CheckpointError("DUPLICATE_ROOT")
            entry_ids.add(entry_id)
            if isinstance(entry, NativeEntry):
                self._validate_entry(state.baseline, schema, entry)

    def _validate_entry(self, baseline: str, schema: NativeSchema, entry: NativeEntry) -> None:
        if entry.schema_version != NATIVE_ENTRY_V1:
            raise CheckpointError("INVALID_NATIVE_ENTRY_SCHEMA")
        if baseline in {"rag_frozen", "retrieval_rag"} and entry.native_component == "index":
            raise CheckpointError("RAG_INDEX_ENTRY_FORBIDDEN")
        if baseline == "dynamic_cheatsheet_rs_optional":
            if (entry.semantic_kind, entry.native_component) == ("dynamic_cheatsheet", "strategy"):
                if not entry.direct_parent_ids:
                    raise CheckpointError("DIRECT_DC_STRATEGY_ROOT")
                return
        if (entry.semantic_kind, entry.native_component) != (schema.semantic_kind, schema.native_component):
            raise CheckpointError("WRONG_NATIVE_COMPONENT")
