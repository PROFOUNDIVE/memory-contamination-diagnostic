from __future__ import annotations

from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


class ReflexionStylePolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        reflections = "\n".join(entry.content for entry in memory.entries[-3:])
        return [{"role": "user", "content": f"Reflections:\n{reflections}\n\nSolve: {task.input}"}]
