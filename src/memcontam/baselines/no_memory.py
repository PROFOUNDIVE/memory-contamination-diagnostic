from __future__ import annotations

from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


class NoMemoryPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"Solve this {task.task_name} instance: {task.input}"}]
