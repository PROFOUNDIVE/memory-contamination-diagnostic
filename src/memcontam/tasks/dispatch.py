from __future__ import annotations

import json
from typing import Any, Mapping

from memcontam.tasks.base import TaskInstance


def canonical_task_json(task: TaskInstance | Mapping[str, Any]) -> str:
    payload = (
        task.model_dump(mode="json", exclude={"verifier_spec"})
        if isinstance(task, TaskInstance)
        else task
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_core_task_json(task: TaskInstance) -> str:
    payload = {"input": task.input, "task_name": task.task_name}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
