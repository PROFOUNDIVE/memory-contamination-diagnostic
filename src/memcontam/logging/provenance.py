from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence, cast

from memcontam.logging.schema import (
    ContaminationClass,
    ContaminationExposure,
    LineageBasis,
    LineageEdge,
    LineageStatus,
    MemoryEvent,
    MemoryItemLog,
    PromptSourceSpan,
    TargetContaminationSetSpec,
)
from memcontam.logging.validation import normalize_direct_parent_ids, validate_outcome_metadata
from memcontam.memory.stores import MemoryEntry


def memory_snapshot_hash(entries: list[MemoryEntry]) -> str:
    """Return a canonical SHA-256 hash of a normalized memory snapshot.

    The snapshot uses ``MemoryItemLog`` lineage fields with dict keys sorted,
    while preserving the list order of entries so append/replacement order
    remains auditable.
    """
    snapshot = [
        MemoryItemLog.from_memory_entry(entry, entries).model_dump(mode="json") for entry in entries
    ]
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
    before_by_id = {entry.entry_id: entry for entry in memory_before}
    updated_ids = [
        entry.entry_id
        for entry in memory_after
        if entry.entry_id in before_by_id
        and MemoryItemLog.from_memory_entry(entry, memory_after).model_dump(mode="json")
        != MemoryItemLog.from_memory_entry(
            before_by_id[entry.entry_id], memory_before
        ).model_dump(mode="json")
    ]

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
        declared_new_id = memory_write_event.get("new_entry_id")
        if isinstance(declared_new_id, str) and declared_new_id and declared_new_id not in new_ids:
            raise ValueError(f"accepted memory event new_entry_id is not new: {declared_new_id}")
    elif new_ids:
        raise ValueError(
            f"{status} memory event must not claim new entries: {new_ids}"
        )
    elif snapshot_changed:
        raise ValueError(
            f"{status} memory event must not change snapshot hash"
        )

    parent_entry_ids = normalize_direct_parent_ids(memory_write_event)
    source_entry_ids = _string_list(memory_write_event.get("source_entry_ids"))
    contaminated_source_ids = _string_list(
        memory_write_event.get("source_contaminated_entry_ids")
    )

    # Only recorded direct-parent evidence can populate exact parents. Source
    # evidence and updater context remain source evidence.
    if not parent_entry_ids and new_ids:
        parent_entry_ids = _union_direct_parent_ids(memory_after, new_ids)
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
    changed_ids = [*new_ids, *updated_ids]
    after_by_id = {entry.entry_id: entry for entry in memory_after}
    lineage_edges = [
        edge
        for entry_id in changed_ids
        for edge in build_direct_parent_edges(
            MemoryItemLog.from_memory_entry(after_by_id[entry_id], memory_after)
        )
    ]

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
        lineage_edges=lineage_edges,
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


def _union_direct_parent_ids(entries: list[MemoryEntry], entry_ids: list[str]) -> list[str]:
    result: list[str] = []
    for entry in entries:
        if entry.entry_id in entry_ids:
            _extend_unique(result, normalize_direct_parent_ids(entry.metadata))
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CanonicalLineage:
    contamination_class: ContaminationClass
    injected_root_ids: list[str]
    lineage_status: LineageStatus
    lineage_basis: LineageBasis
    direct_parent_ids: list[str]


def target_set_membership(
    item: MemoryItemLog, target_set: TargetContaminationSetSpec
) -> bool:
    """Return whether a normalized item is eligible for a fixed target set."""
    return (
        item.contamination_class in target_set.included_classes
        and (not target_set.require_exact_lineage or item.lineage_status == "exact")
    )


def combine_lineage_status(statuses: list[LineageStatus]) -> LineageStatus:
    """Combine local evidence without upgrading approximate evidence to exact."""
    if statuses and all(status == "exact" for status in statuses):
        return "exact"
    if "approximate" in statuses:
        return "approximate"
    return "unavailable"


def canonical_lineage_for_entry(
    entry: MemoryEntry,
    entries: list[MemoryEntry] | None = None,
    _visited: frozenset[str] = frozenset(),
) -> CanonicalLineage:
    """Derive the v2 lineage view from typed seed and recorded-parent evidence.

    Source-ID unions and signatures remain non-authoritative: they never create
    exact roots or a derived class.
    """
    if entry.entry_id in _visited:
        return CanonicalLineage("clean", [], "unavailable", "none", [])

    metadata = entry.metadata
    direct_parent_ids = normalize_direct_parent_ids(metadata)
    declared_roots = _string_list(metadata.get("injected_root_ids"))
    declared_status = metadata.get("lineage_status")
    declared_basis = metadata.get("lineage_basis")
    declared_class = metadata.get("contamination_class")

    if _is_typed_seed(entry, declared_class, declared_status, declared_basis, direct_parent_ids, declared_roots):
        return CanonicalLineage(
            contamination_class=cast(ContaminationClass, declared_class),
            injected_root_ids=declared_roots,
            lineage_status="exact",
            lineage_basis="seed",
            direct_parent_ids=[],
        )

    if declared_basis == "signature" or declared_status == "approximate":
        return CanonicalLineage(
            contamination_class="clean",
            injected_root_ids=[],
            lineage_status="approximate",
            lineage_basis="signature" if declared_basis == "signature" else "none",
            direct_parent_ids=[],
        )

    parent_roots, parent_statuses = _parent_lineage(
        direct_parent_ids, entries, _visited | {entry.entry_id}
    )
    exact_roots = parent_roots or declared_roots
    status = combine_lineage_status(parent_statuses)
    if direct_parent_ids and exact_roots and (status == "exact" or entries is None):
        return CanonicalLineage(
            contamination_class="derived",
            injected_root_ids=exact_roots,
            lineage_status="exact",
            lineage_basis="recorded_parent",
            direct_parent_ids=direct_parent_ids,
        )

    if (
        entry.memory_type == "full_history_transcript"
        and metadata.get("memory_error_status") == "satisfied"
        and not exact_roots
    ):
        return CanonicalLineage(
            contamination_class="natural",
            injected_root_ids=[],
            lineage_status=status,
            lineage_basis="recorded_parent" if direct_parent_ids else "none",
            direct_parent_ids=direct_parent_ids,
        )

    return CanonicalLineage(
        contamination_class="clean",
        injected_root_ids=[],
        lineage_status=status,
        lineage_basis="recorded_parent" if direct_parent_ids and status == "exact" else "none",
        direct_parent_ids=direct_parent_ids if status == "exact" else [],
    )


def phase11_lineage_metadata(
    entry: MemoryEntry,
    entries: list[MemoryEntry],
    target_set: TargetContaminationSetSpec | dict[str, Any] | str | None,
) -> dict[str, Any]:
    """Return the v2 metadata view for a runtime-created memory entry."""
    if not target_set:
        return {}
    if isinstance(target_set, str):
        target_set = TargetContaminationSetSpec(
            target_set_id=target_set,
            definition_version="phase11_v1",
            included_classes=["injected", "derived"],
            require_exact_lineage=True,
        )
    elif isinstance(target_set, dict):
        target_set = TargetContaminationSetSpec.model_validate(target_set)
    lineage = canonical_lineage_for_entry(entry, entries)
    return {
        "contamination_class": lineage.contamination_class,
        "injected_root_ids": lineage.injected_root_ids,
        "lineage_status": lineage.lineage_status,
        "lineage_basis": lineage.lineage_basis,
        "direct_parent_ids": lineage.direct_parent_ids,
        "target_set_id": target_set.target_set_id,
        "is_target_contamination": (
            lineage.contamination_class in target_set.included_classes
            and (not target_set.require_exact_lineage or lineage.lineage_status == "exact")
        ),
    }


def build_direct_parent_edges(item: MemoryItemLog) -> list[LineageEdge]:
    """Build one compact direct edge per recorded parent for a changed entry."""
    return [
        LineageEdge(
            child_entry_id=item.entry_id,
            parent_entry_id=parent_entry_id,
            relation="recorded_parent",
            lineage_status=item.lineage_status or "unavailable",
            lineage_basis=item.lineage_basis or "none",
            injected_root_ids=item.injected_root_ids if item.lineage_status == "exact" else [],
        )
        for parent_entry_id in item.direct_parent_ids
    ]


def canonical_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove forbidden transitive lineage material from canonical rows."""
    return {
        key: _without_ancestor_material(value)
        for key, value in metadata.items()
        if key != "ancestor_ids" and not key.startswith("ancestor_")
    }


def baseline_outcome_to_logging_v2(outcome: Any) -> dict[str, Any]:
    """Map shared baseline evidence into existing ``logging_v2`` destinations."""
    from memcontam.baselines.contracts import BaselineExecutionOutcome

    if not isinstance(outcome, BaselineExecutionOutcome):
        raise TypeError("logging_v2 persistence requires BaselineExecutionOutcome")

    metadata = dict(outcome.metadata)
    trial: dict[str, Any] = {"status": outcome.status}
    for source_field, destination_field in (
        ("final_response", "raw_response"),
        ("parsed_answer", "parsed_answer"),
        ("verifier_result", "verifier_result"),
        ("answer_call_id", "answer_call_id"),
    ):
        value = getattr(outcome, source_field)
        if value is not None:
            trial[destination_field] = value
    for field_name in (
        "method_calls",
        "memory_before",
        "memory_after",
        "retrieved_memory",
        "retrieved_scores",
        "memory_write_event",
    ):
        value = getattr(outcome, field_name)
        if value not in ((), None):
            trial[field_name] = list(value) if isinstance(value, tuple) else value

    failure: dict[str, str] | None = None
    if outcome.status == "failed":
        if (
            outcome.error_type is None
            or outcome.failure_disposition is None
            or outcome.scientific_ineligibility_reason is None
        ):
            raise ValueError("failed outcome requires one complete failure triple")
        metadata["failure_disposition"] = outcome.failure_disposition
        metadata["scientific_ineligibility_reason"] = outcome.scientific_ineligibility_reason
        trial["error_type"] = outcome.error_type
        failure = {"error_type": outcome.error_type, "disposition": outcome.failure_disposition}
    validate_outcome_metadata(outcome, metadata)
    trial["metadata"] = metadata
    return {"trial": trial, "failure": failure}


def _parent_lineage(
    parent_ids: list[str], entries: list[MemoryEntry] | None, visited: frozenset[str]
) -> tuple[list[str], list[LineageStatus]]:
    if not entries:
        return [], []
    by_id = {entry.entry_id: entry for entry in entries}
    roots: list[str] = []
    statuses: list[LineageStatus] = []
    for parent_id in parent_ids:
        parent = by_id.get(parent_id)
        if parent is None:
            return [], ["unavailable"]
        parent_lineage = canonical_lineage_for_entry(parent, entries, visited)
        _extend_unique(roots, parent_lineage.injected_root_ids)
        statuses.append(parent_lineage.lineage_status)
    return roots, statuses


def _is_typed_seed(
    entry: MemoryEntry,
    contamination_class: Any,
    lineage_status: Any,
    lineage_basis: Any,
    direct_parent_ids: list[str],
    injected_root_ids: list[str],
) -> bool:
    if entry.source_trial_id is not None or lineage_status != "exact" or lineage_basis != "seed":
        return False
    if contamination_class == "clean":
        return not direct_parent_ids and not injected_root_ids and entry.clean_or_contaminated == "clean"
    return (
        contamination_class == "injected"
        and not direct_parent_ids
        and injected_root_ids == [entry.entry_id]
        and entry.clean_or_contaminated == "contaminated"
    )


def _without_ancestor_material(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_ancestor_material(child)
            for key, child in value.items()
            if key != "ancestor_ids" and not key.startswith("ancestor_")
        }
    if isinstance(value, list):
        return [_without_ancestor_material(child) for child in value]
    return value


@dataclass(frozen=True)
class PromptSourcePart:
    text: str
    entry: MemoryEntry


def build_prompt_with_sources(
    parts: list[str | PromptSourcePart],
    *,
    message_index: int = 0,
    entries: list[MemoryEntry] | None = None,
) -> tuple[str, list[PromptSourceSpan]]:
    entry_table = entries if entries is not None else [
        part.entry for part in parts if isinstance(part, PromptSourcePart)
    ]
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
                entries=entry_table,
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
    entries: list[MemoryEntry] | None = None,
) -> PromptSourceSpan:
    item = MemoryItemLog.from_memory_entry(entry, entries)
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
        contamination_class=item.contamination_class,
        injected_root_ids=item.injected_root_ids,
        lineage_status=item.lineage_status,
        lineage_basis=item.lineage_basis,
        direct_parent_ids=item.direct_parent_ids,
        target_set_id=item.target_set_id,
        is_target_contamination=item.is_target_contamination,
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
    contamination_class: ContaminationClass | None = None,
    injected_root_ids: list[str] | None = None,
    lineage_status: LineageStatus | None = None,
    lineage_basis: LineageBasis | None = None,
    direct_parent_ids: list[str] | None = None,
    target_set_id: str | None = None,
    is_target_contamination: bool | None = None,
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
        contamination_class=contamination_class,
        injected_root_ids=injected_root_ids or [],
        lineage_status=lineage_status,
        lineage_basis=lineage_basis,
        direct_parent_ids=direct_parent_ids or [],
        target_set_id=target_set_id,
        is_target_contamination=is_target_contamination,
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


def compute_exposure_from_spans_v1(
    answer_call_id: str,
    spans: list[PromptSourceSpan],
    condition: Literal["clean", "contaminated", "contaminated_filter"],
) -> ContaminationExposure:
    """Preserve the Phase-10 binary final-answer exposure contract."""
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


def compute_exposure_from_spans(
    answer_call_id: str,
    spans: list[PromptSourceSpan],
    condition: Literal["clean", "contaminated", "contaminated_filter"],
) -> ContaminationExposure:
    """Compatibility wrapper for logging_v1 callers."""
    return compute_exposure_from_spans_v1(answer_call_id, spans, condition)


def compute_exposure_from_spans_v2(
    answer_call_id: str,
    spans: list[PromptSourceSpan],
    condition: Literal["clean", "contaminated", "contaminated_filter"],
    memory_before: Sequence[dict[str, Any] | MemoryEntry],
    target_set: TargetContaminationSetSpec,
) -> ContaminationExposure:
    """Compute Phase-11 exposure from exact answer spans and a fixed target set."""
    source_entry_ids = _unique_entry_ids(spans)
    memory_items = _normalized_memory_items(memory_before)
    target_entry_ids = [
        item.entry_id for item in memory_items if target_set_membership(item, target_set)
    ]
    if condition == "clean":
        return ContaminationExposure(
            condition=condition,
            status="not_applicable",
            is_exposed=None,
            answer_call_id=answer_call_id,
            target_entry_ids=target_entry_ids,
            source_entry_ids=source_entry_ids,
            exposure_mode="clean",
            reason="clean arm has no controlled target exposure",
            target_set_id=target_set.target_set_id,
        )

    target_entry_id_set = set(target_entry_ids)
    exact_target_spans = [
        span
        for span in spans
        if span.entry_id in target_entry_id_set and _exact_target_span(span, target_set)
    ]

    if not target_entry_ids and _has_approximate_target_candidate(memory_items, spans, target_set):
        return ContaminationExposure(
            condition=condition,
            status="not_evaluable",
            is_exposed=None,
            answer_call_id=answer_call_id,
            target_entry_ids=[],
            source_entry_ids=source_entry_ids,
            exposure_mode="not_evaluable",
            reason="target membership has approximate-only lineage evidence",
            target_set_id=target_set.target_set_id,
            evidence_lineage_status="approximate",
        )

    if not exact_target_spans:
        reason = (
            "target memory was filtered before final answer rendering"
            if condition == "contaminated_filter" and not target_entry_ids
            else "target memory was available but not rendered into the final answer prompt"
        )
        return ContaminationExposure(
            condition=condition,
            status="supported",
            is_exposed=False,
            answer_call_id=answer_call_id,
            target_entry_ids=target_entry_ids,
            source_entry_ids=source_entry_ids,
            exposure_mode="not_in_final_prompt",
            reason=reason,
            target_set_id=target_set.target_set_id,
            evidence_lineage_status="exact",
        )

    exposed_entry_ids = _unique_entry_ids(exact_target_spans)
    exposed_root_ids: list[str] = []
    for span in exact_target_spans:
        _extend_unique(exposed_root_ids, span.injected_root_ids)
    return ContaminationExposure(
        condition=condition,
        status="supported",
        is_exposed=True,
        answer_call_id=answer_call_id,
        target_entry_ids=target_entry_ids,
        source_entry_ids=source_entry_ids,
        exposed_source_ids=exposed_entry_ids,
        exposure_mode="final_prompt",
        reason="exact target source span rendered into final answer prompt",
        target_set_id=target_set.target_set_id,
        exposed_entry_ids=exposed_entry_ids,
        exposed_injected_root_ids=exposed_root_ids,
        evidence_lineage_status="exact",
    )


def _normalized_memory_items(
    memory_before: Sequence[dict[str, Any] | MemoryEntry],
) -> list[MemoryItemLog]:
    entries = [
        entry if isinstance(entry, MemoryEntry) else MemoryEntry.model_validate(entry)
        for entry in memory_before
    ]
    return [MemoryItemLog.from_memory_entry(entry, entries) for entry in entries]


def _unique_entry_ids(spans: list[PromptSourceSpan]) -> list[str]:
    entry_ids: list[str] = []
    for span in spans:
        _extend_unique(entry_ids, [span.entry_id])
    return entry_ids


def _exact_target_span(span: PromptSourceSpan, target_set: TargetContaminationSetSpec) -> bool:
    return (
        span.contamination_class in target_set.included_classes
        and (not target_set.require_exact_lineage or span.lineage_status == "exact")
    )


def _has_approximate_target_candidate(
    memory_items: list[MemoryItemLog],
    spans: list[PromptSourceSpan],
    target_set: TargetContaminationSetSpec,
) -> bool:
    if not target_set.require_exact_lineage:
        return False
    for item in memory_items:
        declared_class = item.metadata.get("contamination_class", item.contamination_class)
        declared_status = item.metadata.get("lineage_status", item.lineage_status)
        if declared_class in target_set.included_classes and declared_status != "exact":
            return True
    return any(
        span.contamination_class in target_set.included_classes and span.lineage_status != "exact"
        for span in spans
    )
