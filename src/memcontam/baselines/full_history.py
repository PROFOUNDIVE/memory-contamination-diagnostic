from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from memcontam.memory.stores import MemoryEntry


@dataclass(frozen=True)
class FullHistoryPayload:
    task_input: str
    raw_response: str


@dataclass
class FullHistoryState:
    records: list[MemoryEntry] = field(default_factory=list)


def render_full_history(entry_id: str, payload: FullHistoryPayload) -> str:
    return (
        f'<BEGIN_HISTORY_RECORD id="{entry_id}">\n'
        f"TASK:\n{payload.task_input}\n\n"
        f"RESPONSE:\n{payload.raw_response}\n"
        "<END_HISTORY_RECORD>"
    )


def __getattr__(name: str) -> Any:
    if name in {"FullHistoryAdapter", "FullHistoryPolicy"}:
        return import_module("memcontam.baselines.full_history_adapter").FullHistoryAdapter
    raise AttributeError(name)
