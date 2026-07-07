from __future__ import annotations

from memcontam.memory.retrieval import lexical_retrieve
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


class RetrievalRagPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        retrieved = lexical_retrieve(str(task.input), memory.entries)
        context = "\n".join(entry.content for entry, _score in retrieved)
        return [{"role": "user", "content": f"Retrieved memory:\n{context}\n\nSolve: {task.input}"}]
