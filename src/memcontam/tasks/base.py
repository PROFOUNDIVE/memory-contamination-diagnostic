from __future__ import annotations

from pydantic import BaseModel, Field


class TaskInstance(BaseModel):
    sample_id: str
    task_name: str
    input: dict
    verifier_spec: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
