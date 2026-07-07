from __future__ import annotations

from memcontam.memory.retrieval import lexical_retrieve
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


class BotStylePolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        templates = lexical_retrieve(str(task.input), memory.entries, k=1)
        template_text = templates[0][0].content if templates else ""
        return [{"role": "user", "content": f"Thought template:\n{template_text}\n\nSolve: {task.input}"}]
