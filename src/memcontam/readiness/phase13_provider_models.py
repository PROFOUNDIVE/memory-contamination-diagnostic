from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from memcontam.readiness.phase13_authority import JsonValue


JsonScalar: TypeAlias = None | bool | int | float | str


class StrictAccountingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OwnedDispatchConfig(StrictAccountingModel):
    execution_owner_id: str
    execution_template_id: str
    provider_config: Mapping[str, JsonScalar]


class TransportAttempt(StrictAccountingModel):
    attempt_id: str = Field(min_length=1)
    semantic_call_id: str = Field(min_length=1)
    execution_owner_id: str = Field(min_length=1)
    attempt_number: int = Field(gt=0)
    status: Literal["completed", "failed", "partial"]
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_microusd: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    storage_bytes: int = Field(ge=0)
    provider_error: str | None
    raw_evidence: Mapping[str, JsonScalar]


class ProviderTotals(StrictAccountingModel):
    transport_attempts: int = Field(ge=0)
    retries: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_microusd: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    storage_bytes: int = Field(ge=0)


class AccountingTotals(ProviderTotals):
    semantic_calls: int = Field(ge=0)
    dispatches: int = Field(ge=0)


class SettledCall(StrictAccountingModel):
    semantic_call_id: str
    execution_owner_id: str
    execution_template_id: str
    provider_error: str | None
    attempts: tuple[TransportAttempt, ...]
    totals: ProviderTotals


class OfflineAccounting(StrictAccountingModel):
    operation: Literal["prefix_derivation", "paired_seed_bootstrap", "report_rendering"]
    owner_id: str
    provider_calls: Literal[0]
    cost_microusd: Literal[0]


class AccountingReport(StrictAccountingModel):
    calls: tuple[SettledCall, ...]
    offline: tuple[OfflineAccounting, ...]
    totals: AccountingTotals


class ProviderAccountingError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProviderDispatchFailure(Exception):
    provider_error: str
    provider_attempts: tuple[Mapping[str, JsonValue], ...]
    provider_totals: Mapping[str, int]

    def __str__(self) -> str:
        return self.provider_error
