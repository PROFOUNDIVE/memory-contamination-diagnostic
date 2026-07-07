from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class VerifierResult(BaseModel):
    is_correct: bool
    parsed_answer: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrialLog(BaseModel):
    trial_id: str
    run_id: str
    task_name: str
    sample_id: str
    baseline: str
    arm: Literal["clean", "contaminated", "contaminated_filter"]
    backbone: str
    input: dict[str, Any]
    gold_or_verifier_spec: dict[str, Any]
    prompt_messages: list[dict[str, str]]
    memory_before: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_memory: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_scores: list[float] = Field(default_factory=list)
    filter_decision: dict[str, Any] | None = None
    raw_response: str
    parsed_answer: str | None = None
    verifier_result: VerifierResult
    memory_write_event: dict[str, Any] | None = None
    memory_after: list[dict[str, Any]] = Field(default_factory=list)
    contamination_exposure: dict[str, Any] = Field(default_factory=dict)
    bad_memory_uptake_label: str | None = None
    repeated_failure_label: str | None = None
    recovery_after_filter_label: str | None = None
    latency_ms: int | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_estimate: float | None = None
    retry_count: int = 0
    error_type: str | None = None
