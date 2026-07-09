from __future__ import annotations

from memcontam.memory.retrieval import render_retrieved_record, retrieve_records
from memcontam.memory.stores import MemoryState
from memcontam.tasks.base import TaskInstance


class RetrievalRagPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        retrieved = retrieve_records(str(task.input), memory.entries)
        context = "\n".join(render_retrieved_record(record) for record in retrieved)
        return [{"role": "user", "content": f"Retrieved memory:\n{context}\n\nSolve: {task.input}"}]
