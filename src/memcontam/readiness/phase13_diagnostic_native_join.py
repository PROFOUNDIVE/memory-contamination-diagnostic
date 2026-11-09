from __future__ import annotations

from memcontam.baselines.dynamic_cheatsheet_phase12 import (
    SourceAliasTable,
    _strategy_memory,
    core_synthesis_message,
    curate_pre_generation,
)
from memcontam.logging.schema import MethodCall
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.dispatch import render_model_visible_task

from .phase13_live_diagnostic_models import LiveDiagnosticError


def validate_curator_lineage(calls: tuple[MethodCall, ...], before: NativeState, task: TaskInstance) -> None:
    if before.baseline != "dc_rs" or len(calls) != 2 or calls[1].error_type is not None:
        return
    curator, answer = calls
    archive_ids = {entry.entry_id for entry in before.entries
                   if isinstance(entry, NativeEntry) and entry.native_component == "archive"}
    source_ids = tuple(span.entry_id for span in curator.source_spans if span.entry_id in archive_ids)
    if any(source_id is None for source_id in source_ids):
        raise LiveDiagnosticError("DIAGNOSTIC_CURATOR_SOURCE_INVALID")
    aliases = SourceAliasTable.from_source_ids(tuple(source_id for source_id in source_ids if source_id is not None))
    archive = {row.entry_id: row for row in (MemoryEntry.model_validate(raw) for raw in before.native_state["archive"])}
    strategies = [entry for entry in before.entries if isinstance(entry, NativeEntry) and entry.native_component == "strategy"]
    message, spans, expected_aliases = core_synthesis_message(render_model_visible_task(task),
        None if not strategies else _strategy_memory(strategies[-1]),
        [archive[source_id] for source_id in aliases.resolve(aliases.visible_ids)])
    if curator.messages != [message] or curator.source_spans != spans or aliases != expected_aliases:
        raise LiveDiagnosticError("DIAGNOSTIC_CURATOR_SOURCE_INVALID")
    candidate = curate_pre_generation(curator.raw_response or "", fallback_strategy="",
        retrieved_archive_ids=aliases.resolve(aliases.visible_ids), source_aliases=aliases, strict_whole_response=True)
    if not answer.source_spans or candidate.parser_status != "accepted":
        raise LiveDiagnosticError("DIAGNOSTIC_CURATOR_SOURCE_INVALID")
    for span in answer.source_spans:
        rendered = answer.messages[span.message_index]["content"][span.start:span.end]
        if (span.parent_call_id != curator.call_id or tuple(span.direct_parent_ids) != candidate.explicit_source_ids
            or span.lineage_status != candidate.lineage_status or rendered != candidate.content):
            raise LiveDiagnosticError("DIAGNOSTIC_CURATOR_PARENT_MISMATCH")
