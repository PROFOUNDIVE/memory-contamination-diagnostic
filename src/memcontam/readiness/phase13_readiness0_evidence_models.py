from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memcontam.logging.schema import PromptSourceSpan


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProviderReasoningContract(_FrozenModel):
    mode: Literal["standard"]
    effort: Literal["none"]
    context: Literal["current_turn"]


class ProviderRequestContract(_FrozenModel):
    model: Literal["gpt-5.6-luna"]
    input_sha256: Sha256
    temperature: float
    top_p: float | int
    reasoning: ProviderReasoningContract
    previous_response_id: None
    service_tier: Literal["default"]
    store: Literal[False]
    tools: list[dict[str, Any]]
    max_output_tokens: int = Field(ge=1)


class ProviderAuthorityContract(_FrozenModel):
    maximum_input_tokens: int = Field(ge=1)
    maximum_output_tokens: int = Field(ge=1)
    execution_envelope_id: Literal["CORE_EXECUTION_ENVELOPE_REGISTRY_V2"]
    execution_envelope_sha256: Sha256
    failure_contract_id: Literal["CORE_TRANSPORT_ATTEMPT_CONTRACT_V2"]
    failure_contract_sha256: Sha256
    terminal_failure_contract_id: Literal["CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"]
    terminal_failure_contract_sha256: Sha256
    rate_card_sha256: Sha256


class ProviderCallEvidence(_FrozenModel):
    call_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    raw_response: str | None
    transport_attempts: int = Field(ge=0)
    token_usage: dict[str, int]
    latency_ms: int | None = Field(ge=0)
    provider_cost_usd: float | None = Field(ge=0)
    provider_response_id: str | None
    provider_usage: dict[str, int | dict[str, int]] | None
    provider_service_tier: str | None
    requested_model: Literal["gpt-5.6-luna"]
    returned_model: str | None
    response_status: str | None
    failure_code: str | None = None
    error_type: str | None = None
    provider_status: str | None = None
    provider_incomplete_reason: str | None = None
    reasoning_mode: Literal["standard"]
    reasoning_effort: Literal["none"]
    reasoning_context: Literal["current_turn"]
    previous_response_id: None
    store: Literal[False]
    tools: tuple[()]
    maximum_input_tokens: int = Field(ge=1)
    maximum_output_tokens: int = Field(ge=1)
    execution_envelope_id: Literal["CORE_EXECUTION_ENVELOPE_REGISTRY_V2"]
    execution_envelope_sha256: Sha256
    failure_contract_id: Literal["CORE_TRANSPORT_ATTEMPT_CONTRACT_V2"]
    failure_contract_sha256: Sha256
    terminal_failure_contract_id: Literal["CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"]
    terminal_failure_contract_sha256: Sha256
    raw_usage: dict[str, int | dict[str, int]] | None
    normalized_usage: dict[str, int]
    authoritative_provider_cost_usd: float | None = Field(ge=0)
    derived_cost_usd: float | None = Field(ge=0)
    cost_source: Literal[
        "AUTHORITATIVE_PROVIDER", "DERIVED_FROM_PROVIDER_USAGE"
    ] | None
    rate_card_sha256: Sha256
    source_spans: tuple[PromptSourceSpan, ...]


class RuntimeJoinEvidence(_FrozenModel):
    task: Literal["game24", "mmlu_pro_engineering", "mmlu_pro_physics"]
    baseline: Literal[
        "nomem", "fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"
    ]
    sample_id: str = Field(min_length=1)
    suffix_position: Literal[1]
    sample_order: Literal[1]
    trajectory_seed: Literal[0]
    concrete_seed_id: Literal["0"]
    execution_template_id: str = Field(min_length=1)
    ordered_sample_ids_sha256: Sha256
    checkpoint_registry_sha256: Sha256
    registration_packet_sha256: Sha256
    retrieval_query_sha256: Sha256 | None
    retrieval_candidates_sha256: Sha256 | None
    retrieval_source_span_sha256: Sha256
    retrieval_event_id: str | None
    retrieved_entry_ids: tuple[str, ...]
    retrieved_scores: tuple[float, ...]
    memory_before_sha256: Sha256
    memory_after_sha256: Sha256
    capacity_law_id: Literal["luna_common_visible_memory_capacity_v1"] | None
    capacity_tokens: Literal[8192] | None
    capacity_artifact_sha256: Sha256
    task_order_sha256: Sha256
    analysis_window_id: Literal["core_prefix_50"]
    analysis_window_registry_sha256: Sha256
    text_only: Literal[True]
    tool_execution_count: Literal[0]


__all__ = [
    "ProviderAuthorityContract",
    "ProviderCallEvidence",
    "ProviderRequestContract",
    "RuntimeJoinEvidence",
]
