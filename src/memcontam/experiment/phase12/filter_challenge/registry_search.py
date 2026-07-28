from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BeforeValidator, model_validator

from memcontam.experiment.phase12.filter_challenge.registry_common import (
    NonNegativeInt,
    PositiveInt,
    RegistryValidationError,
    StrictRegistry,
    StringTuple,
    UnitInterval,
    parse_tuple,
    stable_hash,
    validate_ids,
)


class Stratum(StrictRegistry):
    task_family: Literal["game24", "math_equation_balancer", "word_sorting"]
    baseline: Literal["full_history", "rag_frozen", "bot_style", "reflexion_style"]


StratumTuple: TypeAlias = Annotated[tuple[Stratum, ...], BeforeValidator(parse_tuple)]


class SuiteCandidate(StrictRegistry):
    operational_probe_suite_id: str
    probe_ids: StringTuple
    replicates_per_probe: PositiveInt

    @model_validator(mode="after")
    def _validate_probe_ids(self) -> Self:
        validate_ids(self.probe_ids, "SUITE_PROBE_IDS_EMPTY", "DUPLICATE_SUITE_PROBE_ID")
        return self


class KappaCandidate(StrictRegistry):
    kappa_id: str
    min_total_evaluable_replicates: PositiveInt
    min_distinct_evaluable_probes: PositiveInt
    min_witness_replicates_per_probe: PositiveInt
    min_distinct_witness_probes: PositiveInt


class CoverageContractCandidate(StrictRegistry):
    coverage_contract_id: str
    strata: StratumTuple
    strict_clean_solvable_probe_count: NonNegativeInt
    candidate_final_context_inclusion_rate: UnitInterval
    paired_evaluability_rate: UnitInterval
    correct_control_false_quarantine_rate: UnitInterval
    irrelevant_control_false_quarantine_rate: UnitInterval
    ordinary_route_false_memory_coverage: UnitInterval


class ReplicateRetryCandidate(StrictRegistry):
    replicate_retry_id: str
    control_retry_limit: NonNegativeInt
    challenge_retry_limit: NonNegativeInt


class CanonicalizerCandidate(StrictRegistry):
    canonicalizer_id: str
    canonicalizer_version: str | None


class ToleranceCandidate(StrictRegistry):
    tolerance_id: str
    correct_false_quarantine_tolerance: UnitInterval
    irrelevant_false_quarantine_tolerance: UnitInterval
    repeatability_tolerance: UnitInterval


class RateCandidate(StrictRegistry):
    rate_id: str
    minimum_rate: UnitInterval


class BudgetCapCandidate(StrictRegistry):
    budget_cap_id: str
    call_cap: NonNegativeInt
    latency_cap_ms: NonNegativeInt
    cost_cap_label: Literal["not_estimated"]


class CiProcedureCandidate(StrictRegistry):
    ci_procedure_id: str
    procedure: Literal["paired-bootstrap"]


class TieBreakCandidate(StrictRegistry):
    tie_break_id: str
    order: StringTuple

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        validate_ids(self.order, "TIE_BREAK_ORDER_EMPTY", "DUPLICATE_TIE_BREAK_ORDER_ITEM")
        return self


SuiteCandidateTuple: TypeAlias = Annotated[tuple[SuiteCandidate, ...], BeforeValidator(parse_tuple)]
KappaCandidateTuple: TypeAlias = Annotated[tuple[KappaCandidate, ...], BeforeValidator(parse_tuple)]
CoverageContractTuple: TypeAlias = Annotated[
    tuple[CoverageContractCandidate, ...], BeforeValidator(parse_tuple)
]
ReplicateRetryTuple: TypeAlias = Annotated[tuple[ReplicateRetryCandidate, ...], BeforeValidator(parse_tuple)]
CanonicalizerTuple: TypeAlias = Annotated[tuple[CanonicalizerCandidate, ...], BeforeValidator(parse_tuple)]
ToleranceTuple: TypeAlias = Annotated[tuple[ToleranceCandidate, ...], BeforeValidator(parse_tuple)]
RateTuple: TypeAlias = Annotated[tuple[RateCandidate, ...], BeforeValidator(parse_tuple)]
BudgetCapTuple: TypeAlias = Annotated[tuple[BudgetCapCandidate, ...], BeforeValidator(parse_tuple)]
CiProcedureTuple: TypeAlias = Annotated[tuple[CiProcedureCandidate, ...], BeforeValidator(parse_tuple)]
TieBreakTuple: TypeAlias = Annotated[tuple[TieBreakCandidate, ...], BeforeValidator(parse_tuple)]


class SearchConfig(StrictRegistry):
    registry_id: str
    evidence_layer: Literal["build"]
    scientific_result: Literal[False]
    fixture_only: Literal[True]
    calibration_probe_inventory_id: str
    calibration_probe_inventory_manifest_hash: str
    operational_probe_suite_manifest_id: str
    operational_probe_suite_manifest_hash: str
    calibration_probe_ids: StringTuple
    required_strata: StratumTuple
    suite_candidates: SuiteCandidateTuple
    kappa_candidates: KappaCandidateTuple
    coverage_contract_candidates: CoverageContractTuple
    replicate_retry_candidates: ReplicateRetryTuple
    canonicalizer_candidates: CanonicalizerTuple
    tolerance_candidates: ToleranceTuple
    paired_evaluability_candidates: RateTuple
    inclusion_rate_candidates: RateTuple
    ordinary_route_coverage_candidates: RateTuple
    budget_cap_candidates: BudgetCapTuple
    ci_procedure_candidates: CiProcedureTuple
    constraint_order: StringTuple
    deterministic_tie_break_candidates: TieBreakTuple
    search_config_hash: str

    def stable_hash(self) -> str:
        return stable_hash(self, "search_config_hash")

    @model_validator(mode="after")
    def _validate_registry(self) -> Self:
        self._validate_candidate_sets()
        self._validate_coverage_contracts()
        self._validate_suite_and_kappa_candidates()
        if self.search_config_hash != self.stable_hash():
            raise RegistryValidationError("HASH_MISMATCH")
        return self

    def _validate_candidate_sets(self) -> None:
        for ids, empty_code, duplicate_code in (
            (self.calibration_probe_ids, "CALIBRATION_PROBE_IDS_EMPTY", "DUPLICATE_CALIBRATION_PROBE_ID"),
            (tuple(item.operational_probe_suite_id for item in self.suite_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_SUITE_CANDIDATE_ID"),
            (tuple(item.kappa_id for item in self.kappa_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_KAPPA_CANDIDATE_ID"),
            (tuple(item.coverage_contract_id for item in self.coverage_contract_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_COVERAGE_CONTRACT_ID"),
            (tuple(item.replicate_retry_id for item in self.replicate_retry_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_REPLICATE_RETRY_ID"),
            (tuple(item.canonicalizer_id for item in self.canonicalizer_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_CANONICALIZER_ID"),
            (tuple(item.tolerance_id for item in self.tolerance_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_TOLERANCE_ID"),
            (tuple(item.rate_id for item in self.paired_evaluability_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_PAIRED_EVALUABILITY_ID"),
            (tuple(item.rate_id for item in self.inclusion_rate_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_INCLUSION_RATE_ID"),
            (tuple(item.rate_id for item in self.ordinary_route_coverage_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_ORDINARY_ROUTE_COVERAGE_ID"),
            (tuple(item.budget_cap_id for item in self.budget_cap_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_BUDGET_CAP_ID"),
            (tuple(item.ci_procedure_id for item in self.ci_procedure_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_CI_PROCEDURE_ID"),
            (tuple(item.tie_break_id for item in self.deterministic_tie_break_candidates), "FINITE_CANDIDATE_SET_EMPTY", "DUPLICATE_TIE_BREAK_ID"),
        ):
            validate_ids(ids, empty_code, duplicate_code)
        validate_ids(
            self.constraint_order,
            "CONSTRAINT_ORDER_EMPTY",
            "DUPLICATE_CONSTRAINT_ORDER_ITEM",
        )

    def _validate_coverage_contracts(self) -> None:
        if not self.required_strata:
            raise RegistryValidationError("REQUIRED_STRATA_EMPTY")
        if len(set(self.required_strata)) != len(self.required_strata):
            raise RegistryValidationError("REQUIRED_STRATA_DUPLICATE")
        for contract in self.coverage_contract_candidates:
            if len(set(contract.strata)) != len(contract.strata):
                raise RegistryValidationError("DUPLICATE_COVERAGE_STRATUM")
            if set(contract.strata) != set(self.required_strata):
                raise RegistryValidationError("REQUIRED_STRATA_MISSING")

    def _validate_suite_and_kappa_candidates(self) -> None:
        for suite in self.suite_candidates:
            if not set(suite.probe_ids).issubset(self.calibration_probe_ids):
                raise RegistryValidationError("SUITE_PROBE_UNKNOWN")
        for kappa in self.kappa_candidates:
            if not any(
                kappa.min_distinct_witness_probes <= kappa.min_distinct_evaluable_probes
                <= len(suite.probe_ids)
                and kappa.min_witness_replicates_per_probe <= suite.replicates_per_probe
                and kappa.min_total_evaluable_replicates
                <= len(suite.probe_ids) * suite.replicates_per_probe
                for suite in self.suite_candidates
            ):
                raise RegistryValidationError("KAPPA_INCOHERENT")


class SelectedPolicy(StrictRegistry):
    registry_id: str
    evidence_layer: Literal["build"]
    scientific_result: Literal[False]
    fixture_only: Literal[True]
    search_config_id: str
    search_config_hash: str
    operational_probe_suite_id: str
    kappa_id: str
    coverage_contract_id: str
    replicate_retry_id: str
    canonicalizer_id: str
    tolerance_id: str
    paired_evaluability_rate_id: str
    inclusion_rate_id: str
    ordinary_route_coverage_id: str
    budget_cap_id: str
    ci_procedure_id: str
    tie_break_id: str
    selected_policy_hash: str

    def stable_hash(self) -> str:
        return stable_hash(self, "selected_policy_hash")

    @model_validator(mode="after")
    def _validate_hash(self) -> Self:
        if self.selected_policy_hash != self.stable_hash():
            raise RegistryValidationError("HASH_MISMATCH")
        return self
