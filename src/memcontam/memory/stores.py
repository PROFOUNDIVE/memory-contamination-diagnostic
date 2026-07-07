from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    entry_id: str
    content: str
    memory_type: str
    clean_or_contaminated: str = "clean"
    source_trial_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class MemoryState(BaseModel):
    entries: list[MemoryEntry] = Field(default_factory=list)
