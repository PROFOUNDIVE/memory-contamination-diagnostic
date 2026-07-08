from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EXPOSURE_KEYS = {
    "condition",
    "is_exposed",
    "source_entry_ids",
    "contamination_types",
    "memory_before_entry_ids",
    "retrieved_entry_ids",
    "exposure_mode",
    "reason",
}

BadMemoryUptakeLabel = Literal[
    "not_applicable", "not_evaluable", "no_uptake_detected", "uptake_detected"
]
RepeatedFailureLabel = Literal["not_applicable", "first_failure", "repeated_failure"]
RecoveryAfterFilterLabel = Literal["not_applicable", "recovered", "not_recovered"]


class ContaminationExposure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: Literal["clean", "contaminated", "contaminated_filter"] = "clean"
    is_exposed: bool = False
    source_entry_ids: list[str] = Field(default_factory=list)
    contamination_types: list[str] = Field(default_factory=list)
    memory_before_entry_ids: list[str] = Field(default_factory=list)
    retrieved_entry_ids: list[str] = Field(default_factory=list)
    exposure_mode: Literal["none", "memory_before", "retrieved_memory"] = "none"
    reason: str = "clean arm has no contaminated memory sources"


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
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_write_event: dict[str, Any] | None = None
    memory_after: list[dict[str, Any]] = Field(default_factory=list)
    contamination_exposure: ContaminationExposure = Field(default_factory=ContaminationExposure)
    bad_memory_uptake_label: BadMemoryUptakeLabel | None = None
    repeated_failure_label: RepeatedFailureLabel | None = None
    recovery_after_filter_label: RecoveryAfterFilterLabel | None = None
    latency_ms: int | None = Field(default=None, strict=True, ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_estimate: float | None = None
    retry_count: int = 0
    error_type: str | None = None

    @field_validator("contamination_exposure", mode="before")
    @classmethod
    def _require_exact_exposure_keys(cls, value: Any) -> Any:
        if isinstance(value, dict) and set(value) != EXPOSURE_KEYS:
            raise ValueError("contamination_exposure must use the exact controlled-exposure keys")
        return value
