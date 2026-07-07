from __future__ import annotations

from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


class FullHistoryPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        history = "\n".join(entry.content for entry in memory.entries)
        return [{"role": "user", "content": f"History:\n{history}\n\nSolve: {task.input}"}]
