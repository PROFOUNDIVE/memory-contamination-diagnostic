"""Test-only contamination QA helpers."""

from memcontam.memory.filters import FilterTelemetry, filter_legacy_replay_entries
from memcontam.memory.stores import MemoryEntry


def drop_known_contaminated(
    entries: list[MemoryEntry],
) -> tuple[list[MemoryEntry], FilterTelemetry]:
    return filter_legacy_replay_entries(entries)


__all__ = ["drop_known_contaminated"]
