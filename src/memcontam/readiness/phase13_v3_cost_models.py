from __future__ import annotations

import hashlib
import json
from typing import Annotated, ClassVar, Final, Literal, Self, TypeVar

from pydantic import BaseModel, Field, model_validator

from memcontam.readiness.phase13_cost_policy_models import Budget, RateCard, Sha256
from memcontam.readiness.phase13_v3_authority_models import AuthoritySnapshotV3, FrozenModel


RATE_CARD: Final = RateCard(
    input_usd_per_million="0.20", cached_input_usd_per_million="0.02",
    output_usd_per_million="1.20", cache_write_planning_premium="1.25",
    cache_read_credit="none", long_context_threshold_tokens=272000,
    long_context_input_multiplier="2.0", long_context_output_multiplier="1.5",
    fx_planning_ceiling_krw_per_usd=1600,
)
BUDGET: Final = Budget(total_budget_ceiling_krw=500000, reserve_fraction="0.10", core_authorization_gate_krw=450000)
DecimalString = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")]
NonnegativeInt = Annotated[int, Field(ge=0)]


class ProviderUsage(FrozenModel):
    input_tokens: NonnegativeInt
    output_tokens: NonnegativeInt
    cached_input_tokens: NonnegativeInt = 0

    @model_validator(mode="after")
    def check_cache(self) -> Self:
        if self.cached_input_tokens > self.input_tokens:
            raise CostError("MAIN_COST_NUMERIC_INVALID")
        return self


class ProviderCostEvidence(FrozenModel):
    monetary_cost: str | None = None
    currency: str | None = None
    usage: ProviderUsage | None = None


class ActualCost(FrozenModel):
    evidence: ProviderCostEvidence
    derived_usd: DecimalString | None
    source: Literal["AUTHORITATIVE_PROVIDER", "DERIVED_FROM_PROVIDER_USAGE"]
    selected_usd: DecimalString
    realized_krw: NonnegativeInt


class CostError(ValueError):
    def __init__(self, code: str, evidence: ProviderCostEvidence | None = None, derived_usd: str | None = None) -> None:
        self.code, self.evidence, self.derived_usd = code, evidence, derived_usd
        super().__init__(code)


class RequestTokens(FrozenModel):
    input_tokens: NonnegativeInt
    output_tokens: NonnegativeInt
    cache_write_tokens: NonnegativeInt = 0

    @model_validator(mode="after")
    def check_cache(self) -> Self:
        if self.cache_write_tokens > self.input_tokens:
            raise CostError("MAIN_COST_NUMERIC_INVALID")
        return self


class CostArtifact(FrozenModel):
    hash_field: ClassVar[str]


class ActivatedPolicyV3(CostArtifact):
    hash_field: ClassVar[str] = "policy_hash"
    schema_version: Literal["phase13_main_activated_cost_policy_v3"] = "phase13_main_activated_cost_policy_v3"
    authority: AuthoritySnapshotV3
    model: Literal["gpt-5.6-luna"] = "gpt-5.6-luna"
    service_tier: Literal["default"] = "default"
    billing_mode: Literal["standard/default"] = "standard/default"
    currency: Literal["USD"] = "USD"
    rate_card: RateCard = RATE_CARD
    budget: Budget = BUDGET
    rate_card_source: Literal["https://developers.openai.com/api/docs/models/gpt-5.6-luna"] = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
    rate_card_retrieved_at: Literal["2026-08-27T09:03:29Z"] = "2026-08-27T09:03:29Z"
    rate_card_normalized_extract_sha256: Literal["83968b3ad89096fddb255fafc21e3da0576da67ab021f6fcfca40fae0851f916"] = "83968b3ad89096fddb255fafc21e3da0576da67ab021f6fcfca40fae0851f916"
    approval_evidence_id: Literal["phase13_450k_cost_policy_approval_2026-08-28"] = "phase13_450k_cost_policy_approval_2026-08-28"
    approval_evidence_sha256: Literal["daff466d7160e4ff50158edf3ac926a9859541aa267e38ec593dc6d49aff474f"] = "daff466d7160e4ff50158edf3ac926a9859541aa267e38ec593dc6d49aff474f"
    rounding: Literal["componentwise_stage_upward_with_per_unit_cumulative_ceiling_attribution"] = "componentwise_stage_upward_with_per_unit_cumulative_ceiling_attribution"
    final_upward_rounding: Literal["required"] = "required"
    prospective_cost_source: Literal["DETERMINISTIC_CONSERVATIVE_UPPER_BOUND"] = "DETERMINISTIC_CONSERVATIVE_UPPER_BOUND"
    policy_hash: Sha256


class PrefreezeBindings(FrozenModel):
    package_cells_hash: Sha256
    seed_checkpoint_hash: Sha256
    prefix_ownership_hash: Sha256
    stage_occurrences_hash: Sha256
    governed_source_hash: Sha256
    runtime_hash: Sha256
    request_compiler_hash: Sha256
    serializer_hash: Sha256
    tokenizer_hash: Sha256


class StageOccurrences(FrozenModel):
    stage_id: str = Field(min_length=1)
    calls: int = Field(gt=0)
    cache_write_tokens: NonnegativeInt = 0


class CostUnit(FrozenModel):
    unit_id: str = Field(min_length=1)
    stages: tuple[StageOccurrences, ...] = Field(min_length=1)


class BaseCostInputsV3(CostArtifact):
    hash_field: ClassVar[str] = "base_inputs_hash"
    schema_version: Literal["phase13_main_base_cost_inputs_v3"] = "phase13_main_base_cost_inputs_v3"
    policy: ActivatedPolicyV3
    bindings: PrefreezeBindings
    units: tuple[CostUnit, ...] = Field(min_length=1)
    base_inputs_hash: Sha256


class FinalOrder(FrozenModel):
    unit_ids: tuple[str, ...] = Field(min_length=1)
    runtime_hash: Sha256
    request_hash: Sha256
    tokenizer_hash: Sha256


class CompleteCostInputsV3(CostArtifact):
    hash_field: ClassVar[str] = "complete_inputs_hash"
    schema_version: Literal["phase13_main_complete_cost_inputs_v3"] = "phase13_main_complete_cost_inputs_v3"
    base: BaseCostInputsV3
    base_inputs_hash: Sha256
    final_order: FinalOrder
    final_order_hash: Sha256
    complete_inputs_hash: Sha256


class ExactStageCost(FrozenModel):
    stage_id: str
    semantic_calls: NonnegativeInt
    input_exact_krw: DecimalString
    output_exact_krw: DecimalString
    input_krw_ceiling: NonnegativeInt
    output_krw_ceiling: NonnegativeInt


class CostTotals(FrozenModel):
    stage_costs: tuple[ExactStageCost, ...]
    semantic_calls: NonnegativeInt
    cmax_main_krw: NonnegativeInt
    core_authorization_gate_krw: Literal[450000] = 450000
    gate_margin_krw: int
    gate_result: Literal["PASS", "FAIL"]


class CostWitnessV3(CostArtifact):
    hash_field: ClassVar[str] = "witness_hash"
    schema_version: Literal["phase13_main_cost_witness_v3"] = "phase13_main_cost_witness_v3"
    role: Literal["NON_AUTHORIZING_STAGE_FEASIBILITY_WITNESS"] = "NON_AUTHORIZING_STAGE_FEASIBILITY_WITNESS"
    base_inputs_hash: Sha256
    totals: CostTotals
    witness_hash: Sha256


class UnitProjection(FrozenModel):
    unit_id: str
    projected_krw: NonnegativeInt


class CostProofV3(CostArtifact):
    hash_field: ClassVar[str] = "proof_hash"
    schema_version: Literal["phase13_main_cost_proof_v3"] = "phase13_main_cost_proof_v3"
    proof_id: Literal["phase13-main-a-corrected-cost-proof-v3"] = "phase13-main-a-corrected-cost-proof-v3"
    role: Literal["DERIVED_RECOMPUTABLE_EXECUTION_ARTIFACT"] = "DERIVED_RECOMPUTABLE_EXECUTION_ARTIFACT"
    complete_inputs_hash: Sha256
    witness_hash: Sha256
    package_core_hash: Sha256
    totals: CostTotals
    projected_krw: tuple[UnitProjection, ...] = Field(min_length=1)
    proof_hash: Sha256


Artifact = ActivatedPolicyV3 | BaseCostInputsV3 | CompleteCostInputsV3 | CostWitnessV3 | CostProofV3
ArtifactT = TypeVar("ArtifactT", bound=CostArtifact)


def canonical_bytes(model: BaseModel, exclude: str | None = None) -> bytes:
    return (json.dumps(model.model_dump(mode="json", exclude=set() if exclude is None else {exclude}),
                       ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(model: BaseModel, exclude: str | None = None) -> str:
    return hashlib.sha256(canonical_bytes(model, exclude)).hexdigest()


def seal(model: ArtifactT) -> ArtifactT:
    return model.model_copy(update={model.hash_field: digest(model, model.hash_field)})


def check_hash(model: Artifact) -> None:
    if getattr(model, model.hash_field) != digest(model, model.hash_field):
        raise CostError("MAIN_COST_PROOF_MISMATCH")
