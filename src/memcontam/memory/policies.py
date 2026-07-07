from __future__ import annotations

from typing import Protocol

from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


class BaselinePolicy(Protocol):
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        """Build model messages for one trial."""
