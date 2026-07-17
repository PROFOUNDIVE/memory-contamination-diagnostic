from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

from memcontam.logging.schema import (
    ContaminationExposure,
    MemoryEvent,
    MemoryItemLog,
    PromptSourceSpan,
)
from memcontam.memory.stores import MemoryEntry


def memory_snapshot_hash(entries: list[MemoryEntry]) -> str:
    """Return a canonical SHA-256 hash of a normalized memory snapshot.

    The snapshot uses ``MemoryItemLog`` lineage fields with dict keys sorted,
    while preserving the list order of entries so append/replacement order
    remains auditable.
    """
    snapshot = [MemoryItemLog.from_memory_entry(entry).model_dump(mode="json") for entry in entries]
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_memory_event(
    baseline: str,
    source_trial_id: str,
    memory_before: list[MemoryEntry],
    memory_after: list[MemoryEntry],
    memory_write_event: dict[str, Any] | None,
) -> MemoryEvent | None:
    """Normalize a baseline-specific ``memory_write_event`` to a typed ``MemoryEvent``.

    Returns ``None`` for read-only baselines (``no_memory``, ``retrieval_rag``)
    and for events without a recognized mutation type.  The returned event has
    placeholder run-context fields (``run_metadata_id``, ``run_id``, ``stage``,
    ``event_seq``) that the strict writer will assign.
    """
    if memory_write_event is None:
        return None

    event_type = memory_write_event.get("type") or memory_write_event.get("event_type")
    if event_type is None:
        return None

    # Read-only baselines never claim a memory mutation.
    if baseline in {"no_memory", "retrieval_rag"}:
        return None

    before_ids = [entry.entry_id for entry in memory_before]
    after_ids = [entry.entry_id for entry in memory_after]
    before_set = set(before_ids)
    after_set = set(after_ids)
    new_ids = [entry_id for entry_id in after_ids if entry_id not in before_set]
    removed_ids = [entry_id for entry_id in before_ids if entry_id not in after_set]
    updated_ids: list[str] = []

    before_hash = memory_snapshot_hash(memory_before)
    after_hash = memory_snapshot_hash(memory_after)
    snapshot_changed = before_hash != after_hash

    declared_status = _canonical_status(str(memory_write_event.get("status", "unknown")))
    # A changed snapshot is the ground-truth signal that a mutation occurred;
    # the declared status may describe a sub-operation (e.g. DC-RS synthesis
    # preserved while an I/O pair is still appended).
    status = "accepted" if snapshot_changed else declared_status

    if status == "accepted":
        for entry_id in new_ids:
            if entry_id not in after_set:
                raise ValueError(f"accepted memory event claims missing new entry: {entry_id}")
    elif new_ids:
        raise ValueError(
            f"{status} memory event must not claim new entries: {new_ids}"
        )
    elif snapshot_changed:
        raise ValueError(
            f"{status} memory event must not change snapshot hash"
        )

    parent_entry_ids = _string_list(memory_write_event.get("parent_entry_ids"))
    source_entry_ids = _string_list(memory_write_event.get("source_entry_ids"))
    contaminated_source_ids = _string_list(
        memory_write_event.get("source_contaminated_entry_ids")
    )

    # BoT records ``source_entry_ids`` but not ``parent_entry_ids``; treat the
    # source IDs as parents for the normalized event.
    if not parent_entry_ids and source_entry_ids:
        parent_entry_ids = list(source_entry_ids)

    # Fallback to new-entry metadata for lineage fields the baseline event did
    # not carry explicitly.
    if not parent_entry_ids and new_ids:
        parent_entry_ids = _union_metadata_field(memory_after, new_ids, "parent_entry_ids")
    if not source_entry_ids and new_ids:
        source_entry_ids = _union_metadata_field(memory_after, new_ids, "source_entry_ids")
    if not contaminated_source_ids and new_ids:
        contaminated_source_ids = _union_metadata_field(
            memory_after, new_ids, "source_contaminated_entry_ids"
        )
        if not contaminated_source_ids:
            contaminated_source_ids = _contaminated_source_ids_from_entries(
                memory_after, new_ids
            )

    creation_origin: str | None = None
    memory_version: str | None = None
    if new_ids:
        new_entries = [entry for entry in memory_after if entry.entry_id in new_ids]
        if new_entries:
            item = MemoryItemLog.from_memory_entry(new_entries[0])
            creation_origin = item.creation_origin
            memory_version = item.version

    operation = _operation_from_event(event_type, baseline)

    # Context fields are writer-assigned; use minimal placeholders here.
    return MemoryEvent(
        memory_id="",
        run_metadata_id="",
        run_id="",
        trial_id=source_trial_id,
        trial_seq=0,
        event_seq=0,
        stage="",
        event_type="memory_write",
        operation=operation,
        baseline=baseline,
        source_trial_id=source_trial_id,
        parent_entry_ids=parent_entry_ids,
        source_entry_ids=source_entry_ids,
        contaminated_source_ids=contaminated_source_ids,
        before_entry_ids=before_ids,
        after_entry_ids=after_ids,
        before_snapshot_hash=before_hash,
        after_snapshot_hash=after_hash,
        new_entry_ids=new_ids,
        updated_entry_ids=updated_ids,
        removed_entry_ids=removed_ids,
        creation_origin=creation_origin,
        memory_version=memory_version,
        status=status,
        created_at=_now(),
    )


def _canonical_status(status: str) -> str:
    if status in {"accepted", "replaced"}:
        return "accepted"
    if status in {"rejected", "rejected_empty"}:
        return "rejected"
    if status in {"preserved", "preserved_missing_tag", "preserved_empty"}:
        return "preserved"
    if status == "reused":
        return "reused"
    if status == "incomplete":
        return "incomplete"
    return status


def _operation_from_event(event_type: str, baseline: str) -> str:
    if baseline == "full_history":
        return "append"
    if baseline == "reflexion_style":
        return "append"
    if baseline == "bot_style":
        return "insert"
    if baseline == "dynamic_cheatsheet_optional":
        return "replace"
    if baseline == "dynamic_cheatsheet_rs_optional":
        return "replace_and_append"
    return event_type


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _contaminated_source_ids_from_entries(
    entries: list[MemoryEntry], entry_ids: list[str]
) -> list[str]:
    result: list[str] = []
    for entry in entries:
        if entry.entry_id not in entry_ids or entry.clean_or_contaminated != "contaminated":
            continue
        sources = entry.metadata.get("source_entry_ids", [entry.entry_id])
        if not isinstance(sources, list):
            sources = [entry.entry_id]
        for value in sources:
            if isinstance(value, str) and value not in result:
                result.append(value)
    return result


def _union_metadata_field(
    entries: list[MemoryEntry], entry_ids: list[str], field: str
) -> list[str]:
    result: list[str] = []
    for entry in entries:
        if entry.entry_id not in entry_ids:
            continue
        values = entry.metadata.get(field)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value not in result:
                    result.append(value)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PromptSourcePart:
    text: str
    entry: MemoryEntry


def build_prompt_with_sources(
    parts: list[str | PromptSourcePart],
    *,
    message_index: int = 0,
) -> tuple[str, list[PromptSourceSpan]]:
    content_parts: list[str] = []
    spans: list[PromptSourceSpan] = []
    offset = 0
    for part in parts:
        if isinstance(part, str):
            content_parts.append(part)
            offset += len(part)
        else:
            content_parts.append(part.text)
            span = source_span_from_entry(
                part.entry,
                part.text,
                message_index=message_index,
                start=offset,
                end=offset + len(part.text),
            )
            spans.append(span)
            offset += len(part.text)
    return "".join(content_parts), spans


def source_span_from_entry(
    entry: MemoryEntry,
    text: str,
    message_index: int,
    start: int,
    end: int,
) -> PromptSourceSpan:
    item = MemoryItemLog.from_memory_entry(entry)
    return PromptSourceSpan(
        message_index=message_index,
        start=start,
        end=end,
        rendered_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        entry_id=item.entry_id,
        parent_call_id=_optional_string(item.metadata.get("parent_call_id")),
        source_ids=item.source_entry_ids,
        parent_ids=item.parent_entry_ids,
        lineage_id=item.lineage_id,
        version=item.version,
        origin=item.creation_origin,
        clean_or_contaminated=cast(Literal["clean", "contaminated"], item.clean_or_contaminated),
    )


def derived_source_span(
    text: str,
    *,
    message_index: int,
    start: int,
    end: int,
    entry_id: str,
    parent_call_id: str,
    source_ids: list[str],
    parent_ids: list[str],
    lineage_id: str,
    version: str,
    origin: str,
    clean_or_contaminated: Literal["clean", "contaminated"],
) -> PromptSourceSpan:
    return PromptSourceSpan(
        message_index=message_index,
        start=start,
        end=end,
        rendered_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        entry_id=entry_id,
        parent_call_id=parent_call_id,
        source_ids=source_ids,
        parent_ids=parent_ids,
        lineage_id=lineage_id,
        version=version,
        origin=origin,
        clean_or_contaminated=clean_or_contaminated,
    )


def source_lineage_from_spans(
    spans: list[PromptSourceSpan],
) -> tuple[list[str], list[str], Literal["clean", "contaminated"]]:
    source_ids: list[str] = []
    parent_ids: list[str] = []
    contaminated = False
    for span in spans:
        _extend_unique(source_ids, span.source_ids)
        _extend_unique(parent_ids, span.parent_ids)
        _extend_unique(parent_ids, [span.entry_id])
        if span.clean_or_contaminated == "contaminated":
            contaminated = True
            if not span.source_ids:
                _extend_unique(source_ids, [span.entry_id])
    return source_ids, parent_ids, "contaminated" if contaminated else "clean"


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def compute_exposure_from_spans(
    answer_call_id: str,
    spans: list[PromptSourceSpan],
    condition: Literal["clean", "contaminated", "contaminated_filter"],
) -> ContaminationExposure:
    if condition == "clean":
        return ContaminationExposure(
            condition=condition,
            status="not_applicable",
            is_exposed=None,
            answer_call_id=answer_call_id,
            exposure_mode="clean",
            reason="clean arm has no contaminated memory sources",
        )

    contaminated_spans = [span for span in spans if span.clean_or_contaminated == "contaminated"]
    if not contaminated_spans:
        return ContaminationExposure(
            condition=condition,
            status="supported",
            is_exposed=False,
            answer_call_id=answer_call_id,
            exposure_mode="not_in_final_prompt",
            reason="contaminated sources were available but not rendered into the final answer prompt",
        )

    exposed_source_ids: list[str] = []
    for span in contaminated_spans:
        if span.source_ids:
            for source_id in span.source_ids:
                if source_id not in exposed_source_ids:
                    exposed_source_ids.append(source_id)
        elif span.entry_id not in exposed_source_ids:
            exposed_source_ids.append(span.entry_id)

    source_entry_ids: list[str] = []
    for span in spans:
        if span.entry_id not in source_entry_ids:
            source_entry_ids.append(span.entry_id)
        for source_id in span.source_ids:
            if source_id not in source_entry_ids:
                source_entry_ids.append(source_id)

    return ContaminationExposure(
        condition=condition,
        status="supported",
        is_exposed=True,
        answer_call_id=answer_call_id,
        target_entry_ids=[span.entry_id for span in spans],
        source_entry_ids=source_entry_ids,
        exposed_source_ids=exposed_source_ids,
        exposure_mode="final_prompt",
        reason="contaminated source span rendered into final answer prompt",
    )
