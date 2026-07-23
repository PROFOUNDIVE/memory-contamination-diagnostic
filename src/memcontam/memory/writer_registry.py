from __future__ import annotations

from dataclasses import dataclass

from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3


@dataclass(frozen=True)
class WriterPermission:
    baseline: str
    semantic_kind: str
    writer_id: str
    writer_stage: str
    native_component: str
    requires_source_trial: bool


NATIVE_WRITER_PERMISSIONS = (
    WriterPermission(
        "full_history",
        "full_history_transcript",
        "fh_appender",
        "full_history_generate",
        "history",
        True,
    ),
    WriterPermission(
        "bot_style",
        "thought_template",
        "bot_buffer_manager",
        "bot_thought_distill",
        "buffer",
        True,
    ),
    WriterPermission(
        "reflexion_style",
        "verbal_reflection",
        "reflexion_reflector",
        "reflexion_reflect",
        "reflections",
        True,
    ),
    WriterPermission(
        "retrieval_rag",
        "rag_document",
        "rag_corpus_loader",
        "rag_corpus_load",
        "corpus",
        False,
    ),
    WriterPermission(
        "dynamic_cheatsheet_optional",
        "dynamic_cheatsheet",
        "dc_strategy_writer",
        "dynamic_cheatsheet_curate",
        "strategy",
        True,
    ),
    WriterPermission(
        "dynamic_cheatsheet_rs_optional",
        "dc_rs_io_pair",
        "dc_archive_writer",
        "dc_rs_generate",
        "archive",
        True,
    ),
    WriterPermission(
        "dynamic_cheatsheet_rs_optional",
        "dynamic_cheatsheet",
        "dc_strategy_writer",
        "dc_rs_synthesize",
        "strategy",
        True,
    ),
)


@dataclass(frozen=True)
class WriterRegistry:
    permissions: tuple[WriterPermission, ...] = NATIVE_WRITER_PERMISSIONS

    @classmethod
    def native(cls) -> WriterRegistry:
        return cls()

    def permission_for(self, envelope: MemoryCardEnvelopeV3) -> WriterPermission | None:
        for permission in self.permissions:
            if (
                permission.baseline,
                permission.semantic_kind,
                permission.writer_id,
                permission.writer_stage,
                permission.native_component,
            ) == (
                envelope.baseline,
                envelope.semantic_kind,
                envelope.writer_id,
                envelope.writer_stage,
                envelope.native_component,
            ):
                return permission
        return None

    def permits(self, envelope: MemoryCardEnvelopeV3) -> bool:
        return self.permission_for(envelope) is not None
