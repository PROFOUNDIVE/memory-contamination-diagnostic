from __future__ import annotations

import hashlib
from typing import Literal, TypedDict, cast

from memcontam.memory.stores import MemoryEntry


class FilterDecisionItem(TypedDict):
    entry_id: str
    content_hash: str
    ground_truth: Literal["clean", "contaminated"]
    action: Literal["kept", "removed"]
    reason: str
    score: float | None


class FilterTelemetry(TypedDict):
    filter_name: str
    decisions: list[FilterDecisionItem]
    input_source_ids: list[str]
    kept_source_ids: list[str]
    removed_source_ids: list[str]
    input_count: int
    kept_count: int
    removed_count: int
    # ponytail: legacy aggregate key; Task 10 migrates consumers to removed_count
    dropped: int


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def filter_legacy_replay_entries(
    entries: list[MemoryEntry],
) -> tuple[list[MemoryEntry], FilterTelemetry]:
    decisions: list[FilterDecisionItem] = []
    kept: list[MemoryEntry] = []
    removed_source_ids: list[str] = []

    for entry in entries:
        if entry.clean_or_contaminated == "contaminated":
            action: Literal["kept", "removed"] = "removed"
            reason = "known_contaminated"
            removed_source_ids.append(entry.entry_id)
        else:
            action = "kept"
            reason = "clean"
            kept.append(entry)

        decisions.append(
            FilterDecisionItem(
                entry_id=entry.entry_id,
                content_hash=_content_hash(entry.content),
                ground_truth=cast(Literal["clean", "contaminated"], entry.clean_or_contaminated),
                action=action,
                reason=reason,
                score=None,
            )
        )

    removed_count = len(removed_source_ids)
    telemetry = FilterTelemetry(
        filter_name="drop_known_contaminated",
        decisions=decisions,
        input_source_ids=[entry.entry_id for entry in entries],
        kept_source_ids=[entry.entry_id for entry in kept],
        removed_source_ids=removed_source_ids,
        input_count=len(entries),
        kept_count=len(kept),
        removed_count=removed_count,
        dropped=removed_count,
    )
    return kept, telemetry
